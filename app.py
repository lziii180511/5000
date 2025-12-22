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

# ============ POLYGON RECOGNITION AND FITTING ============

def douglas_peucker(points, epsilon):
    """Douglas-Peucker algorithm to simplify curve and find key points"""
    points = np.array(points)
    if len(points) < 3:
        return [0, len(points) - 1]
    
    # Find point with maximum distance from line between start and end
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-10:
        # Start and end are the same point, find farthest point
        dists = np.sqrt(np.sum((points - start)**2, axis=1))
        max_idx = np.argmax(dists)
        max_dist = dists[max_idx]
    else:
        # Distance from each point to the line
        line_dir = line_vec / line_len
        vecs = points - start
        proj_lengths = np.dot(vecs, line_dir)
        proj_points = start + np.outer(proj_lengths, line_dir)
        dists = np.sqrt(np.sum((points - proj_points)**2, axis=1))
        max_idx = np.argmax(dists)
        max_dist = dists[max_idx]
    
    if max_dist > epsilon:
        # Recursively simplify
        left = douglas_peucker(points[:max_idx+1], epsilon)
        right = douglas_peucker(points[max_idx:], epsilon)
        # Combine results (avoid duplicate of max_idx)
        return left[:-1] + [x + max_idx for x in right]
    else:
        return [0, len(points) - 1]

def find_corners_dp(points, tolerance_ratio=0.02):
    """Find corners using Douglas-Peucker with adaptive tolerance"""
    points = np.array(points)
    
    # Calculate bounding box diagonal as reference size
    bbox_diag = np.sqrt((points[:,0].max() - points[:,0].min())**2 + 
                        (points[:,1].max() - points[:,1].min())**2)
    
    epsilon = bbox_diag * tolerance_ratio
    
    # Get simplified point indices
    corner_indices = douglas_peucker(points, epsilon)
    
    return corner_indices

def identify_polygon(points):
    """Identify polygon type based on Douglas-Peucker corner detection"""
    points = np.array(points)
    n = len(points)
    
    if n < 5:
        return 'line', [0, n-1]
    
    start, end = points[0], points[-1]
    
    # Calculate path length and direct distance
    segments = np.diff(points, axis=0)
    path_lengths = np.sqrt(np.sum(segments**2, axis=1))
    total_path = np.sum(path_lengths)
    direct_dist = np.linalg.norm(end - start)
    
    # Check if it's a straight line
    if total_path > 0 and direct_dist / total_path > 0.95:
        return 'line', [0, n-1]
    
    # Check if closed shape (start and end are close)
    is_closed = direct_dist < total_path * 0.15
    
    # Find corners using Douglas-Peucker
    corner_indices = find_corners_dp(points)
    
    # For closed shapes, we may need to merge start/end corner
    if is_closed and len(corner_indices) >= 2:
        # Remove the last index if it's essentially the same as start
        if corner_indices[-1] == n - 1:
            corner_indices = corner_indices[:-1]
    
    num_corners = len(corner_indices)
    
    # If not closed and only 2 corners (start/end), it's a line
    if not is_closed and num_corners <= 2:
        # Additional linearity check using R²
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
    
    # Check circularity for closed shapes
    if is_closed:
        cx, cy = np.mean(points[:, 0]), np.mean(points[:, 1])
        distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        mean_dist = np.mean(distances)
        circularity = np.std(distances) / (mean_dist + 1e-10)
        
        # Very circular = circle
        if circularity < 0.08:
            return 'circle', corner_indices
        
        # Many corners + somewhat circular = circle
        if num_corners > 8 and circularity < 0.15:
            return 'circle', corner_indices
    
    # Map corner count to polygon type
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
        # Closed with 2 corners might be an elongated shape, treat as ellipse-like
        return 'circle', corner_indices
    else:
        return 'line', corner_indices

def fit_polygon(points, shape_type, corner_indices):
    """Fit ideal polygon to detected corners"""
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
    
    # Regular polygon (triangle, square, pentagon, etc.)
    n_sides = {
        'triangle': 3, 'square': 4, 'pentagon': 5,
        'hexagon': 6, 'heptagon': 7, 'octagon': 8
    }[shape_type]
    
    # Get actual corner points
    actual_corners = np.array([points[i] for i in corner_indices if i < len(points)])
    
    # Calculate center from corner points
    if len(actual_corners) >= 2:
        cx = np.mean(actual_corners[:, 0])
        cy = np.mean(actual_corners[:, 1])
    else:
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
    
    # Calculate radius as average distance from center to corners
    if len(actual_corners) >= 2:
        corner_dists = np.sqrt((actual_corners[:, 0] - cx)**2 + (actual_corners[:, 1] - cy)**2)
        R = np.mean(corner_dists)
    else:
        R = np.mean(np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2))
    
    # Calculate rotation angle - align first vertex with first detected corner
    if len(actual_corners) > 0:
        first_corner = actual_corners[0]
        theta = np.arctan2(first_corner[1] - cy, first_corner[0] - cx)
    else:
        theta = np.arctan2(points[0][1] - cy, points[0][0] - cx)
    
    # Generate perfect regular polygon vertices
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
    """Main function: identify and fit polygon"""
    shape_type, corner_indices = identify_polygon(points)
    return fit_polygon(points, shape_type, corner_indices)

# ============ END POLYGON ============

HTML = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Curve Fitting Tool</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #f5f5f5; overflow: hidden; }
        #container { width: 100vw; height: 100vh; display: flex; flex-direction: column; }
        #toolbar { background: white; padding: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
        button { padding: 10px 20px; border: none; border-radius: 8px; background: #007AFF; color: white; font-size: 16px; cursor: pointer; }
        button:active { background: #0051D5; }
        #clear { background: #FF3B30; }
        .mode-btn { background: #34C759; }
        .mode-btn.active { background: #248A3D; }
        #polygon { background: #AF52DE; }
        #polygon.active { background: #8B3DB0; }
        #toggleStroke { background: #8E8E93; }
        #zoomIn, #zoomOut, #resetZoom { background: #5856D6; padding: 10px 15px; }
        #zoomLevel { font-size: 14px; color: #666; min-width: 80px; }
        #canvasContainer { flex: 1; position: relative; overflow: hidden; background: white; touch-action: none; }
        #canvas { position: absolute; top: 0; left: 0; touch-action: none; }
        #status { padding: 10px; background: #fff; border-top: 1px solid #ddd; text-align: center; font-size: 14px; color: #666; }
    </style>
</head>
<body>
    <div id="container">
        <div id="toolbar">
            <button id="clear">Clear</button>
            <button id="mode" class="mode-btn active">Curve</button>
            <button id="geometry" class="mode-btn">Conics</button>
            <button id="polygon">Polygon</button>
            <button id="toggleStroke">Hide Stroke</button>
            <button id="zoomOut">-</button>
            <span id="zoomLevel">100%</span>
            <button id="zoomIn">+</button>
            <button id="resetZoom">Reset</button>
        </div>
        <div id="canvasContainer">
            <canvas id="canvas"></canvas>
        </div>
        <div id="status">Draw a curve, hold 0.5s before releasing to fit</div>
    </div>
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const status = document.getElementById('status');
        const modeBtn = document.getElementById('mode');
        const geometryBtn = document.getElementById('geometry');
        const polygonBtn = document.getElementById('polygon');
        const toggleStrokeBtn = document.getElementById('toggleStroke');
        
        let currentTool = 'bezier';
        let editMode = false;
        let showStroke = true;
        let points = [], segments = [], geometryShape = null, polygonShape = null;
        let zoom = 1, offsetX = 0, offsetY = 0;
        let dragging = null, panning = false, lastX = 0, lastY = 0, lastMove = 0;
        let activePointerId = null;

        function updateButtons() {
            modeBtn.classList.toggle('active', currentTool === 'bezier');
            geometryBtn.classList.toggle('active', currentTool === 'geometry');
            polygonBtn.classList.toggle('active', currentTool === 'polygon');
            
            modeBtn.textContent = currentTool === 'bezier' && editMode ? 'Curve: Edit' : 'Curve';
            geometryBtn.textContent = currentTool === 'geometry' && editMode ? 'Conic: Edit' : 'Conic';
            polygonBtn.textContent = currentTool === 'polygon' && editMode ? 'Polygon: Edit' : 'Polygon';
        }

        function resize() {
            const r = canvas.parentElement.getBoundingClientRect();
            const scale = window.devicePixelRatio || 2;
            canvas.width = r.width * scale;
            canvas.height = r.height * scale;
            canvas.style.width = r.width + 'px';
            canvas.style.height = r.height + 'px';
            draw();
        }
        window.addEventListener('resize', resize);
        resize();

        function draw() {
            const scale = window.devicePixelRatio || 2;
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.scale(scale, scale);
            ctx.translate(offsetX / scale, offsetY / scale);
            ctx.scale(zoom, zoom);
            
            // Draw hand-drawn stroke
            if (showStroke && points.length > 0) {
                ctx.strokeStyle = '#999';
                ctx.lineWidth = 2 / zoom;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.beginPath();
                ctx.moveTo(points[0][0], points[0][1]);
                for (let i = 1; i < points.length; i++) {
                    ctx.lineTo(points[i][0], points[i][1]);
                }
                ctx.stroke();
            }
            
            // Draw Bezier curves
            if (currentTool === 'bezier' && segments.length > 0) {
                segments.forEach(seg => {
                    const [p0, p1, p2, p3] = seg.controls;
                    ctx.strokeStyle = '#007AFF';
                    ctx.lineWidth = 3 / zoom;
                    ctx.lineCap = 'round';
                    ctx.beginPath();
                    ctx.moveTo(p0[0], p0[1]);
                    ctx.bezierCurveTo(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]);
                    ctx.stroke();
                    
                    if (editMode) {
                        ctx.strokeStyle = '#999';
                        ctx.lineWidth = 1 / zoom;
                        ctx.setLineDash([5 / zoom, 5 / zoom]);
                        ctx.beginPath();
                        ctx.moveTo(p0[0], p0[1]);
                        ctx.lineTo(p1[0], p1[1]);
                        ctx.moveTo(p2[0], p2[1]);
                        ctx.lineTo(p3[0], p3[1]);
                        ctx.stroke();
                        ctx.setLineDash([]);
                        
                        [p0, p3].forEach(p => {
                            ctx.fillStyle = '#34C759';
                            ctx.beginPath();
                            ctx.arc(p[0], p[1], 8 / zoom, 0, Math.PI * 2);
                            ctx.fill();
                        });
                        [p1, p2].forEach(p => {
                            ctx.fillStyle = '#FF3B30';
                            ctx.beginPath();
                            ctx.arc(p[0], p[1], 8 / zoom, 0, Math.PI * 2);
                            ctx.fill();
                        });
                    }
                });
            }
            
            // Draw conic sections
            if (currentTool === 'geometry' && geometryShape) {
                ctx.strokeStyle = '#007AFF';
                ctx.lineWidth = 3 / zoom;
                const geo = geometryShape;
                
                if (geo.type === 'circle') {
                    const [cx, cy, r] = geo.params;
                    ctx.beginPath();
                    ctx.arc(cx, cy, r, 0, Math.PI * 2);
                    ctx.stroke();
                    
                    if (editMode) {
                        ctx.fillStyle = '#FF3B30';
                        ctx.beginPath();
                        ctx.arc(cx, cy, 8 / zoom, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.beginPath();
                        ctx.arc(geo.startEnd[0], geo.startEnd[1], 8 / zoom, 0, Math.PI * 2);
                        ctx.fill();
                    }
                } else if (geo.type === 'ellipse') {
                    const [h, k, a, b, theta] = geo.params;
                    ctx.save();
                    ctx.translate(h, k);
                    ctx.rotate(theta);
                    ctx.beginPath();
                    ctx.ellipse(0, 0, a, b, 0, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.restore();
                    
                    if (editMode) {
                        ctx.fillStyle = '#FF3B30';
                        geo.foci.forEach(f => {
                            ctx.beginPath();
                            ctx.arc(f[0], f[1], 8 / zoom, 0, Math.PI * 2);
                            ctx.fill();
                        });
                        ctx.beginPath();
                        ctx.arc(geo.startEnd[0], geo.startEnd[1], 8 / zoom, 0, Math.PI * 2);
                        ctx.fill();
                    }
                } else if (geo.type === 'parabola') {
                    const [a, b, c] = geo.params;
                    const xMin = Math.min(geo.start[0], geo.end[0]);
                    const xMax = Math.max(geo.start[0], geo.end[0]);
                    
                    ctx.beginPath();
                    for (let x = xMin; x <= xMax; x += 1) {
                        const y = a * x * x + b * x + c;
                        if (x === xMin) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    }
                    ctx.stroke();
                    
                    if (editMode) {
                        ctx.fillStyle = '#FF3B30';
                        ctx.beginPath();
                        ctx.arc(geo.vertex[0], geo.vertex[1], 8 / zoom, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.beginPath();
                        ctx.arc(geo.start[0], geo.start[1], 8 / zoom, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.beginPath();
                        ctx.arc(geo.end[0], geo.end[1], 8 / zoom, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
            }
            
            // Draw polygon
            if (currentTool === 'polygon' && polygonShape) {
                ctx.strokeStyle = '#AF52DE';
                ctx.lineWidth = 3 / zoom;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                
                const poly = polygonShape;
                
                if (poly.type === 'line') {
                    ctx.beginPath();
                    ctx.moveTo(poly.start[0], poly.start[1]);
                    ctx.lineTo(poly.end[0], poly.end[1]);
                    ctx.stroke();
                    
                    if (editMode) {
                        ctx.fillStyle = '#FF3B30';
                        ctx.beginPath();
                        ctx.arc(poly.start[0], poly.start[1], 8 / zoom, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.beginPath();
                        ctx.arc(poly.end[0], poly.end[1], 8 / zoom, 0, Math.PI * 2);
                        ctx.fill();
                    }
                } else if (poly.type === 'circle') {
                    ctx.beginPath();
                    ctx.arc(poly.center[0], poly.center[1], poly.radius, 0, Math.PI * 2);
                    ctx.stroke();
                    
                    if (editMode) {
                        ctx.fillStyle = '#FF3B30';
                        ctx.beginPath();
                        ctx.arc(poly.center[0], poly.center[1], 8 / zoom, 0, Math.PI * 2);
                        ctx.fill();
                    }
                } else if (poly.vertices) {
                    ctx.beginPath();
                    ctx.moveTo(poly.vertices[0][0], poly.vertices[0][1]);
                    for (let i = 1; i < poly.vertices.length; i++) {
                        ctx.lineTo(poly.vertices[i][0], poly.vertices[i][1]);
                    }
                    ctx.closePath();
                    ctx.stroke();
                    
                    if (editMode) {
                        ctx.fillStyle = '#FF3B30';
                        ctx.beginPath();
                        ctx.arc(poly.center[0], poly.center[1], 8 / zoom, 0, Math.PI * 2);
                        ctx.fill();
                        poly.vertices.forEach(v => {
                            ctx.beginPath();
                            ctx.arc(v[0], v[1], 6 / zoom, 0, Math.PI * 2);
                            ctx.fill();
                        });
                    }
                }
            }
        }

        function toCanvas(clientX, clientY) {
            const r = canvas.getBoundingClientRect();
            const scale = window.devicePixelRatio || 2;
            const sx = (clientX - r.left) * scale;
            const sy = (clientY - r.top) * scale;
            return [(sx - offsetX) / zoom / scale, (sy - offsetY) / zoom / scale];
        }

        function findControlPoint(x, y) {
            const r = 12 / zoom;
            
            if (currentTool === 'bezier' && editMode) {
                for (let i = 0; i < segments.length; i++) {
                    for (let j = 0; j < 4; j++) {
                        const [px, py] = segments[i].controls[j];
                        if (Math.hypot(x - px, y - py) < r) return {type: 'bezier', seg: i, pt: j};
                    }
                }
            }
            
            if (currentTool === 'geometry' && editMode && geometryShape) {
                const geo = geometryShape;
                if (geo.type === 'circle') {
                    if (Math.hypot(x - geo.params[0], y - geo.params[1]) < r) return {type: 'circle_center'};
                    if (Math.hypot(x - geo.startEnd[0], y - geo.startEnd[1]) < r) return {type: 'circle_startEnd'};
                } else if (geo.type === 'ellipse') {
                    for (let i = 0; i < 2; i++) {
                        if (Math.hypot(x - geo.foci[i][0], y - geo.foci[i][1]) < r) return {type: 'ellipse_focus', idx: i};
                    }
                    if (Math.hypot(x - geo.startEnd[0], y - geo.startEnd[1]) < r) return {type: 'ellipse_startEnd'};
                } else if (geo.type === 'parabola') {
                    if (Math.hypot(x - geo.vertex[0], y - geo.vertex[1]) < r) return {type: 'parabola_vertex'};
                    if (Math.hypot(x - geo.start[0], y - geo.start[1]) < r) return {type: 'parabola_start'};
                    if (Math.hypot(x - geo.end[0], y - geo.end[1]) < r) return {type: 'parabola_end'};
                }
            }
            
            if (currentTool === 'polygon' && editMode && polygonShape) {
                const poly = polygonShape;
                if (poly.type === 'line') {
                    if (Math.hypot(x - poly.start[0], y - poly.start[1]) < r) return {type: 'line_start'};
                    if (Math.hypot(x - poly.end[0], y - poly.end[1]) < r) return {type: 'line_end'};
                } else if (poly.type === 'circle') {
                    if (Math.hypot(x - poly.center[0], y - poly.center[1]) < r) return {type: 'poly_circle_center'};
                } else if (poly.vertices) {
                    if (Math.hypot(x - poly.center[0], y - poly.center[1]) < r) return {type: 'poly_center'};
                    for (let i = 0; i < poly.vertices.length; i++) {
                        if (Math.hypot(x - poly.vertices[i][0], y - poly.vertices[i][1]) < r) return {type: 'poly_vertex', idx: i};
                    }
                }
            }
            return null;
        }

        function updateParabola() {
            const geo = geometryShape;
            const s = geo.start, e = geo.end, v = geo.vertex;
            try {
                const A = [[s[0]*s[0], s[0], 1], [e[0]*e[0], e[0], 1], [v[0]*v[0], v[0], 1]];
                const B = [s[1], e[1], v[1]];
                const det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
                if (Math.abs(det) < 1e-10) return;
                const inv = [
                    [(A[1][1]*A[2][2]-A[1][2]*A[2][1])/det, (A[0][2]*A[2][1]-A[0][1]*A[2][2])/det, (A[0][1]*A[1][2]-A[0][2]*A[1][1])/det],
                    [(A[1][2]*A[2][0]-A[1][0]*A[2][2])/det, (A[0][0]*A[2][2]-A[0][2]*A[2][0])/det, (A[0][2]*A[1][0]-A[0][0]*A[1][2])/det],
                    [(A[1][0]*A[2][1]-A[1][1]*A[2][0])/det, (A[0][1]*A[2][0]-A[0][0]*A[2][1])/det, (A[0][0]*A[1][1]-A[0][1]*A[1][0])/det]
                ];
                geo.params = [inv[0][0]*B[0]+inv[0][1]*B[1]+inv[0][2]*B[2], inv[1][0]*B[0]+inv[1][1]*B[1]+inv[1][2]*B[2], inv[2][0]*B[0]+inv[2][1]*B[1]+inv[2][2]*B[2]];
            } catch(e) {}
        }

        function updateEllipse() {
            const geo = geometryShape;
            const [f1, f2] = geo.foci;
            const h = (f1[0] + f2[0]) / 2, k = (f1[1] + f2[1]) / 2;
            const c = Math.hypot(f1[0] - f2[0], f1[1] - f2[1]) / 2;
            const a = geo.sumDist / 2;
            const b = Math.sqrt(Math.max(a * a - c * c, 1));
            const theta = Math.atan2(f1[1] - f2[1], f1[0] - f2[0]);
            geo.params = [h, k, a, b, theta];
            const ang = Math.atan2(geo.startEnd[1] - k, geo.startEnd[0] - h) - theta;
            geo.startEnd = [h + a*Math.cos(ang)*Math.cos(theta) - b*Math.sin(ang)*Math.sin(theta), k + a*Math.cos(ang)*Math.sin(theta) + b*Math.sin(ang)*Math.cos(theta)];
        }

        function updatePolygon() {
            const poly = polygonShape;
            if (!poly.vertices) return;
            const cx = poly.center[0], cy = poly.center[1];
            const n = poly.sides;
            for (let i = 0; i < n; i++) {
                const angle = poly.rotation + 2 * Math.PI * i / n;
                poly.vertices[i] = [cx + poly.radius * Math.cos(angle), cy + poly.radius * Math.sin(angle)];
            }
        }

        canvas.addEventListener('pointerdown', e => {
            e.preventDefault();
            if (activePointerId !== null) return;
            activePointerId = e.pointerId;
            canvas.setPointerCapture(e.pointerId);
            
            const [x, y] = toCanvas(e.clientX, e.clientY);
            lastX = e.clientX;
            lastY = e.clientY;
            
            if (editMode) {
                if (e.shiftKey) panning = true;
                else {
                    dragging = findControlPoint(x, y);
                    if (!dragging) panning = true;
                }
            } else {
                points = [[x, y]];
                if (currentTool === 'bezier') segments = [];
                else if (currentTool === 'geometry') geometryShape = null;
                else if (currentTool === 'polygon') polygonShape = null;
                lastMove = Date.now();
                draw();
            }
        });

        canvas.addEventListener('pointermove', e => {
            e.preventDefault();
            if (activePointerId !== null && e.pointerId !== activePointerId) return;
            
            const [x, y] = toCanvas(e.clientX, e.clientY);
            
            if (panning) {
                offsetX += e.clientX - lastX;
                offsetY += e.clientY - lastY;
                lastX = e.clientX;
                lastY = e.clientY;
                draw();
            } else if (dragging) {
                const d = dragging;
                
                if (d.type === 'bezier') {
                    segments[d.seg].controls[d.pt] = [x, y];
                    if (d.pt === 3 && d.seg < segments.length - 1) segments[d.seg + 1].controls[0] = [x, y];
                    if (d.pt === 0 && d.seg > 0) segments[d.seg - 1].controls[3] = [x, y];
                } else if (d.type === 'circle_center') {
                    const geo = geometryShape;
                    const dx = x - geo.params[0], dy = y - geo.params[1];
                    geo.params[0] = x; geo.params[1] = y;
                    geo.center = [x, y];
                    geo.startEnd[0] += dx; geo.startEnd[1] += dy;
                } else if (d.type === 'circle_startEnd') {
                    const geo = geometryShape;
                    const cx = geo.params[0], cy = geo.params[1];
                    geo.params[2] = Math.hypot(x - cx, y - cy);
                    geo.radius = geo.params[2];
                    const ang = Math.atan2(y - cy, x - cx);
                    geo.startEnd = [cx + geo.radius * Math.cos(ang), cy + geo.radius * Math.sin(ang)];
                } else if (d.type === 'ellipse_focus') {
                    geometryShape.foci[d.idx] = [x, y];
                    updateEllipse();
                } else if (d.type === 'ellipse_startEnd') {
                    const geo = geometryShape;
                    const [h, k, a, b, theta] = geo.params;
                    const ang = Math.atan2(y - k, x - h) - theta;
                    geo.startEnd = [h + a*Math.cos(ang)*Math.cos(theta) - b*Math.sin(ang)*Math.sin(theta), k + a*Math.cos(ang)*Math.sin(theta) + b*Math.sin(ang)*Math.cos(theta)];
                } else if (d.type === 'parabola_vertex') {
                    geometryShape.vertex = [x, y];
                    updateParabola();
                } else if (d.type === 'parabola_start') {
                    geometryShape.start = [x, y];
                    updateParabola();
                } else if (d.type === 'parabola_end') {
                    geometryShape.end = [x, y];
                    updateParabola();
                } else if (d.type === 'line_start') {
                    polygonShape.start = [x, y];
                } else if (d.type === 'line_end') {
                    polygonShape.end = [x, y];
                } else if (d.type === 'poly_circle_center') {
                    polygonShape.center = [x, y];
                } else if (d.type === 'poly_center') {
                    const dx = x - polygonShape.center[0], dy = y - polygonShape.center[1];
                    polygonShape.center = [x, y];
                    for (let v of polygonShape.vertices) { v[0] += dx; v[1] += dy; }
                } else if (d.type === 'poly_vertex') {
                    const poly = polygonShape;
                    const cx = poly.center[0], cy = poly.center[1];
                    poly.radius = Math.hypot(x - cx, y - cy);
                    poly.rotation = Math.atan2(y - cy, x - cx) - 2 * Math.PI * d.idx / poly.sides;
                    updatePolygon();
                }
                draw();
            } else if (!editMode && points.length > 0 && activePointerId !== null) {
                const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
                for (const ce of events) {
                    const [cx, cy] = toCanvas(ce.clientX, ce.clientY);
                    points.push([cx, cy]);
                }
                lastMove = Date.now();
                draw();
            }
        });

        canvas.addEventListener('pointerup', async e => {
            e.preventDefault();
            if (e.pointerId !== activePointerId) return;
            
            if (!editMode && points.length > 2 && Date.now() - lastMove >= 500) {
                if (currentTool === 'bezier') {
                    const res = await fetch('/fit', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({points})});
                    segments = (await res.json()).segments;
                    status.textContent = `Detected ${segments.length} curve segment(s)`;
                } else if (currentTool === 'geometry') {
                    const res = await fetch('/fit_geometry', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({points})});
                    geometryShape = await res.json();
                    status.textContent = `Recognized: ${geometryShape.type}`;
                } else if (currentTool === 'polygon') {
                    const res = await fetch('/fit_polygon', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({points})});
                    polygonShape = await res.json();
                    const nameMap = {line: 'Line', circle: 'Circle', triangle: 'Triangle', square: 'Square', pentagon: 'Pentagon', hexagon: 'Hexagon', heptagon: 'Heptagon', octagon: 'Octagon'};
                    status.textContent = `Recognized: ${nameMap[polygonShape.type] || polygonShape.type}`;
                }
                draw();
            }
            
            dragging = null;
            panning = false;
            activePointerId = null;
            canvas.releasePointerCapture(e.pointerId);
        });

        canvas.addEventListener('pointercancel', e => {
            if (e.pointerId === activePointerId) { activePointerId = null; dragging = null; panning = false; }
        });

        canvas.addEventListener('touchstart', e => e.preventDefault(), {passive: false});
        canvas.addEventListener('touchmove', e => e.preventDefault(), {passive: false});
        canvas.addEventListener('touchend', e => e.preventDefault(), {passive: false});

        canvas.addEventListener('wheel', e => {
            e.preventDefault();
            const [ox, oy] = toCanvas(e.clientX, e.clientY);
            const scale = window.devicePixelRatio || 2;
            zoom = Math.max(0.1, Math.min(5, zoom * (e.deltaY > 0 ? 0.9 : 1.1)));
            offsetX = e.clientX * scale - ox * zoom * scale;
            offsetY = e.clientY * scale - oy * zoom * scale;
            document.getElementById('zoomLevel').textContent = Math.round(zoom * 100) + '%';
            draw();
        }, {passive: false});

        document.getElementById('clear').onclick = () => {
            points = []; segments = []; geometryShape = null; polygonShape = null;
            editMode = false; zoom = 1; offsetX = offsetY = 0;
            document.getElementById('zoomLevel').textContent = '100%';
            status.textContent = currentTool === 'bezier' ? 'Draw a curve, hold 0.5s before releasing to fit' : 
                                 currentTool === 'geometry' ? 'Draw a shape, hold 0.5s before releasing to recognize' :
                                 'Draw a polygon, hold 0.5s before releasing to recognize';
            updateButtons();
            draw();
        };

        modeBtn.onclick = () => {
            if (currentTool === 'bezier') {
                if (segments.length > 0) editMode = !editMode;
            } else {
                currentTool = 'bezier'; editMode = false;
                points = []; geometryShape = null; polygonShape = null;
                status.textContent = 'Draw a curve, hold 0.5s before releasing to fit';
            }
            updateButtons();
            draw();
        };

        geometryBtn.onclick = () => {
            if (currentTool === 'geometry') {
                if (geometryShape) editMode = !editMode;
            } else {
                currentTool = 'geometry'; editMode = false;
                points = []; segments = []; polygonShape = null;
                status.textContent = 'Draw a shape, hold 0.5s before releasing to recognize';
            }
            updateButtons();
            draw();
        };

        polygonBtn.onclick = () => {
            if (currentTool === 'polygon') {
                if (polygonShape) editMode = !editMode;
            } else {
                currentTool = 'polygon'; editMode = false;
                points = []; segments = []; geometryShape = null;
                status.textContent = 'Draw a polygon, hold 0.5s before releasing to recognize';
            }
            updateButtons();
            draw();
        };

        toggleStrokeBtn.onclick = () => {
            showStroke = !showStroke;
            toggleStrokeBtn.textContent = showStroke ? 'Hide Stroke' : 'Show Stroke';
            draw();
        };

        document.getElementById('zoomIn').onclick = () => {
            const cx = canvas.width / 2, cy = canvas.height / 2;
            const scale = window.devicePixelRatio || 2;
            const [ox, oy] = [(cx - offsetX) / zoom / scale, (cy - offsetY) / zoom / scale];
            zoom = Math.min(5, zoom * 1.3);
            offsetX = cx - ox * zoom * scale;
            offsetY = cy - oy * zoom * scale;
            document.getElementById('zoomLevel').textContent = Math.round(zoom * 100) + '%';
            draw();
        };

        document.getElementById('zoomOut').onclick = () => {
            const cx = canvas.width / 2, cy = canvas.height / 2;
            const scale = window.devicePixelRatio || 2;
            const [ox, oy] = [(cx - offsetX) / zoom / scale, (cy - offsetY) / zoom / scale];
            zoom = Math.max(0.1, zoom / 1.3);
            offsetX = cx - ox * zoom * scale;
            offsetY = cy - oy * zoom * scale;
            document.getElementById('zoomLevel').textContent = Math.round(zoom * 100) + '%';
            draw();
        };

        document.getElementById('resetZoom').onclick = () => {
            zoom = 1; offsetX = offsetY = 0;
            document.getElementById('zoomLevel').textContent = '100%';
            draw();
        };
    </script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML)

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
