from flask import Flask, render_template_string, request, jsonify
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d

app = Flask(__name__)

def calculate_curvature(points):
    points = np.array(points)
    if len(points) < 3:
        return np.zeros(len(points))
    curvatures = []
    for i in range(len(points)):
        if i == 0 or i == len(points) - 1:
            curvatures.append(0)
            continue
        p1, p2, p3 = points[i-1], points[i], points[i+1]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        a, b, c = np.linalg.norm(p2 - p1), np.linalg.norm(p3 - p2), np.linalg.norm(p3 - p1)
        curvatures.append(4 * area / (a * b * c) if a * b * c > 1e-10 else 0)
    return np.array(curvatures)

def detect_turning_points(points, threshold=0.3):
    curvatures = calculate_curvature(points)
    smoothed = gaussian_filter1d(curvatures, sigma=2)
    turning_indices = [0]
    for i in range(5, len(smoothed) - 5):
        if smoothed[i] > threshold and smoothed[i] > smoothed[i-5:i].max() and smoothed[i] > smoothed[i+1:i+6].max():
            if not turning_indices or i - turning_indices[-1] > 10:
                turning_indices.append(i)
    turning_indices.append(len(points) - 1)
    return turning_indices

def bezier_curve(t, p0, p1, p2, p3):
    return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3

def fit_bezier_segment(points):
    points = np.array(points)
    n = len(points)
    if n < 4:
        p0, p3 = points[0], points[-1]
        return [p0.tolist(), (p0 + (p3 - p0) / 3).tolist(), (p0 + 2 * (p3 - p0) / 3).tolist(), p3.tolist()]
    
    p0, p3 = points[0], points[-1]
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    t_values = np.concatenate([[0], np.cumsum(distances)])
    t_values /= t_values[-1]
    
    def residuals(params):
        p1, p2 = params[:2], params[2:]
        errors = [bezier_curve(t, p0, p1, p2, p3) - points[i] for i, t in enumerate(t_values)]
        return np.concatenate(errors)
    
    result = least_squares(residuals, np.concatenate([p0 + (p3 - p0) / 3, p0 + 2 * (p3 - p0) / 3]))
    return [p0.tolist(), result.x[:2].tolist(), result.x[2:].tolist(), p3.tolist()]

def fit_curve(points):
    if len(points) < 3:
        return {'segments': [], 'turning_points': []}
    turning_indices = detect_turning_points(points)
    segments = [{'controls': fit_bezier_segment(points[turning_indices[i]:turning_indices[i+1]+1]), 
                 'start_idx': turning_indices[i], 'end_idx': turning_indices[i+1]} 
                for i in range(len(turning_indices) - 1)]
    return {'segments': segments, 'turning_points': [points[i] for i in turning_indices]}

def identify_shape(points):
    points = np.array(points)
    start, end = points[0], points[-1]
    dist = np.linalg.norm(end - start)
    
    if dist < len(points) * 0.1:
        xs, ys = points[:, 0], points[:, 1]
        width, height = xs.max() - xs.min(), ys.max() - ys.min()
        return 'circle' if abs(width - height) < max(width, height) * 0.2 else 'ellipse'
    return 'parabola'

def fit_circle(points):
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]
    x_m, y_m = np.mean(x), np.mean(y)
    
    def residuals(params):
        xc, yc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r
    
    r_init = np.sqrt((x - x_m)**2 + (y - y_m)**2).mean()
    result = least_squares(residuals, [x_m, y_m, r_init])
    cx, cy, radius = result.x
    
    angle = np.arctan2(points[0][1] - cy, points[0][0] - cx)
    start_end = [cx + radius * np.cos(angle), cy + radius * np.sin(angle)]
    
    return {
        'type': 'circle', 
        'params': [cx, cy, radius],
        'center': [cx, cy],
        'radius': radius,
        'startEnd': start_end
    }

def fit_ellipse(points):
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]
    x_m, y_m = np.mean(x), np.mean(y)
    
    def residuals(params):
        h, k, a, b, theta = params
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xr = (x - h) * cos_t + (y - k) * sin_t
        yr = -(x - h) * sin_t + (y - k) * cos_t
        return (xr / max(abs(a), 0.1))**2 + (yr / max(abs(b), 0.1))**2 - 1
    
    a_init = (x.max() - x.min()) / 2
    b_init = (y.max() - y.min()) / 2
    result = least_squares(residuals, [x_m, y_m, a_init, b_init, 0])
    h, k, a, b, theta = result.x
    a, b = abs(a), abs(b)
    
    if a >= b:
        c = np.sqrt(max(a**2 - b**2, 0))
        f1 = [h + c * np.cos(theta), k + c * np.sin(theta)]
        f2 = [h - c * np.cos(theta), k - c * np.sin(theta)]
    else:
        c = np.sqrt(max(b**2 - a**2, 0))
        f1 = [h - c * np.sin(theta), k + c * np.cos(theta)]
        f2 = [h + c * np.sin(theta), k - c * np.cos(theta)]
    
    ang = np.arctan2(points[0][1] - k, points[0][0] - h) - theta
    start_end = [
        h + a * np.cos(ang) * np.cos(theta) - b * np.sin(ang) * np.sin(theta),
        k + a * np.cos(ang) * np.sin(theta) + b * np.sin(ang) * np.cos(theta)
    ]
    
    return {
        'type': 'ellipse', 
        'params': [h, k, a, b, theta],
        'foci': [f1, f2],
        'startEnd': start_end,
        'sumDist': 2 * max(a, b)
    }

def fit_parabola(points):
    points = np.array(points)
    start, end = points[0], points[-1]
    
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len > 1e-10:
        line_dir = line_vec / line_len
        perp = np.array([-line_dir[1], line_dir[0]])
        proj = np.abs(np.dot(points - start, perp))
        vertex_idx = np.argmax(proj)
    else:
        vertex_idx = len(points) // 2
    vertex = points[vertex_idx]
    
    try:
        A = np.array([
            [start[0]**2, start[0], 1],
            [end[0]**2, end[0], 1],
            [vertex[0]**2, vertex[0], 1]
        ])
        B = np.array([start[1], end[1], vertex[1]])
        coeffs = np.linalg.solve(A, B)
        a, b, c = coeffs
    except:
        def residuals(params):
            a, b, c = params
            y_fit = a * points[:, 0]**2 + b * points[:, 0] + c
            return y_fit - points[:, 1]
        result = least_squares(residuals, [0.001, 0, np.mean(points[:, 1])])
        a, b, c = result.x
    
    if abs(a) > 1e-10:
        vx = -b / (2 * a)
        vy = a * vx**2 + b * vx + c
    else:
        vx, vy = vertex[0], vertex[1]
    
    return {
        'type': 'parabola', 
        'params': [a, b, c],
        'vertex': [vx, vy],
        'start': start.tolist(),
        'end': end.tolist()
    }

def fit_geometry(points):
    shape = identify_shape(points)
    if shape == 'circle':
        return fit_circle(points)
    elif shape == 'ellipse':
        return fit_ellipse(points)
    else:
        return fit_parabola(points)

def douglas_peucker(points, epsilon):
    points = np.array(points)
    if len(points) < 3:
        return [0, len(points) - 1]
    
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-10:
        dists = np.sqrt(np.sum((points - start)**2, axis=1))
        max_idx = np.argmax(dists)
        max_dist = dists[max_idx]
    else:
        line_dir = line_vec / line_len
        vecs = points - start
        proj_lengths = np.dot(vecs, line_dir)
        proj_points = start + np.outer(proj_lengths, line_dir)
        dists = np.sqrt(np.sum((points - proj_points)**2, axis=1))
        max_idx = np.argmax(dists)
        max_dist = dists[max_idx]
    
    if max_dist > epsilon:
        left = douglas_peucker(points[:max_idx+1], epsilon)
        right = douglas_peucker(points[max_idx:], epsilon)
        return left[:-1] + [x + max_idx for x in right]
    else:
        return [0, len(points) - 1]

def find_corners_dp(points, tolerance_ratio=0.02):
    points = np.array(points)
    bbox_diag = np.sqrt((points[:,0].max() - points[:,0].min())**2 + 
                        (points[:,1].max() - points[:,1].min())**2)
    epsilon = bbox_diag * tolerance_ratio
    corner_indices = douglas_peucker(points, epsilon)
    return corner_indices

def identify_polygon(points):
    points = np.array(points)
    n = len(points)
    
    if n < 5:
        return 'line', [0, n-1]
    
    start, end = points[0], points[-1]
    segments = np.diff(points, axis=0)
    path_lengths = np.sqrt(np.sum(segments**2, axis=1))
    total_path = np.sum(path_lengths)
    direct_dist = np.linalg.norm(end - start)
    
    if total_path > 0 and direct_dist / total_path > 0.95:
        return 'line', [0, n-1]
    
    is_closed = direct_dist < total_path * 0.15
    corner_indices = find_corners_dp(points)
    
    if is_closed and len(corner_indices) >= 2:
        if corner_indices[-1] == n - 1:
            corner_indices = corner_indices[:-1]
    
    num_corners = len(corner_indices)
    
    if not is_closed and num_corners <= 2:
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        try:
            if x_range > y_range:
                coeffs = np.polyfit(points[:, 0], points[:, 1], 1)
                y_pred = np.polyval(coeffs, points[:, 0])
                ss_res = np.sum((points[:, 1] - y_pred)**2)
                ss_tot = np.sum((points[:, 1] - np.mean(points[:, 1]))**2)
            else:
                coeffs = np.polyfit(points[:, 1], points[:, 0], 1)
                x_pred = np.polyval(coeffs, points[:, 1])
                ss_res = np.sum((points[:, 0] - x_pred)**2)
                ss_tot = np.sum((points[:, 0] - np.mean(points[:, 0]))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 1
        except:
            r2 = 0
        if r2 > 0.9:
            return 'line', [0, n-1]
    
    if is_closed:
        cx, cy = np.mean(points[:, 0]), np.mean(points[:, 1])
        distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        mean_dist = np.mean(distances)
        circularity = np.std(distances) / (mean_dist + 1e-10)
        if circularity < 0.08:
            return 'circle', corner_indices
        if num_corners > 8 and circularity < 0.15:
            return 'circle', corner_indices
    
    if num_corners == 3:
        return 'triangle', corner_indices
    elif num_corners == 4:
        return 'square', corner_indices
    elif num_corners == 5:
        return 'pentagon', corner_indices
    elif num_corners == 6:
        return 'hexagon', corner_indices
    elif num_corners == 7:
        return 'heptagon', corner_indices
    elif num_corners == 8:
        return 'octagon', corner_indices
    elif num_corners > 8:
        return 'circle', corner_indices
    elif num_corners == 2 and is_closed:
        return 'circle', corner_indices
    else:
        return 'line', corner_indices

def fit_polygon(points, shape_type, corner_indices):
    points = np.array(points)
    
    if shape_type == 'line':
        return {
            'type': 'line',
            'start': points[0].tolist(),
            'end': points[-1].tolist()
        }
    
    if shape_type == 'circle':
        x, y = points[:, 0], points[:, 1]
        cx, cy = np.mean(x), np.mean(y)
        
        def residuals(params):
            xc, yc, r = params
            return np.sqrt((x - xc)**2 + (y - yc)**2) - r
        
        r_init = np.mean(np.sqrt((x - cx)**2 + (y - cy)**2))
        result = least_squares(residuals, [cx, cy, r_init])
        
        return {
            'type': 'circle',
            'center': [float(result.x[0]), float(result.x[1])],
            'radius': float(result.x[2])
        }
    
    n_sides = {
        'triangle': 3, 'square': 4, 'pentagon': 5,
        'hexagon': 6, 'heptagon': 7, 'octagon': 8
    }[shape_type]
    
    actual_corners = np.array([points[i] for i in corner_indices if i < len(points)])
    
    if len(actual_corners) >= 2:
        cx = np.mean(actual_corners[:, 0])
        cy = np.mean(actual_corners[:, 1])
    else:
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
    
    if len(actual_corners) >= 2:
        corner_dists = np.sqrt((actual_corners[:, 0] - cx)**2 + (actual_corners[:, 1] - cy)**2)
        R = np.mean(corner_dists)
    else:
        R = np.mean(np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2))
    
    if len(actual_corners) > 0:
        first_corner = actual_corners[0]
        theta = np.arctan2(first_corner[1] - cy, first_corner[0] - cx)
    else:
        theta = np.arctan2(points[0][1] - cy, points[0][0] - cx)
    
    vertices = []
    for i in range(n_sides):
        angle = theta + 2 * np.pi * i / n_sides
        vertices.append([float(cx + R * np.cos(angle)), float(cy + R * np.sin(angle))])
    
    return {
        'type': shape_type,
        'center': [float(cx), float(cy)],
        'radius': float(R),
        'rotation': float(theta),
        'vertices': vertices,
        'sides': n_sides,
        'detectedCorners': [points[i].tolist() for i in corner_indices if i < len(points)]
    }

def fit_polygon_shape(points):
    shape_type, corner_indices = identify_polygon(points)
    return fit_polygon(points, shape_type, corner_indices)

from pathlib import Path
html_code = Path("ui.html").read_text(encoding='utf-8')

@app.route('/')
def index():
    return render_template_string(html_code)

@app.route('/fit', methods=['POST'])
def fit():
    return jsonify(fit_curve(request.json['points']))

@app.route('/fit_geometry', methods=['POST'])
def fit_geo():
    return jsonify(fit_geometry(request.json['points']))

@app.route('/fit_polygon', methods=['POST'])
def fit_poly():
    return jsonify(fit_polygon_shape(request.json['points']))

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Server started: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)