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
        return np.concatenate([bezier_curve(t, p0, p1, p2, p3) - points[i] for i, t in enumerate(t_values)])
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
    return {'type': 'circle', 'params': [cx, cy, radius], 'center': [cx, cy], 'radius': radius,
            'startEnd': [cx + radius * np.cos(angle), cy + radius * np.sin(angle)]}

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
    a_init, b_init = (x.max() - x.min()) / 2, (y.max() - y.min()) / 2
    result = least_squares(residuals, [x_m, y_m, a_init, b_init, 0])
    h, k, a, b, theta = result.x
    a, b = abs(a), abs(b)
    if a >= b:
        c = np.sqrt(max(a**2 - b**2, 0))
        f1, f2 = [h + c * np.cos(theta), k + c * np.sin(theta)], [h - c * np.cos(theta), k - c * np.sin(theta)]
    else:
        c = np.sqrt(max(b**2 - a**2, 0))
        f1, f2 = [h - c * np.sin(theta), k + c * np.cos(theta)], [h + c * np.sin(theta), k - c * np.cos(theta)]
    ang = np.arctan2(points[0][1] - k, points[0][0] - h) - theta
    start_end = [h + a * np.cos(ang) * np.cos(theta) - b * np.sin(ang) * np.sin(theta),
                 k + a * np.cos(ang) * np.sin(theta) + b * np.sin(ang) * np.cos(theta)]
    return {'type': 'ellipse', 'params': [h, k, a, b, theta], 'foci': [f1, f2], 'startEnd': start_end, 'sumDist': 2 * max(a, b)}

def fit_parabola(points):
    points = np.array(points)
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len > 1e-10:
        line_dir = line_vec / line_len
        perp = np.array([-line_dir[1], line_dir[0]])
        vertex_idx = np.argmax(np.abs(np.dot(points - start, perp)))
    else:
        vertex_idx = len(points) // 2
    vertex = points[vertex_idx]
    try:
        A = np.array([[start[0]**2, start[0], 1], [end[0]**2, end[0], 1], [vertex[0]**2, vertex[0], 1]])
        coeffs = np.linalg.solve(A, np.array([start[1], end[1], vertex[1]]))
        a, b, c = coeffs
    except:
        def residuals(params):
            a, b, c = params
            return a * points[:, 0]**2 + b * points[:, 0] + c - points[:, 1]
        result = least_squares(residuals, [0.001, 0, np.mean(points[:, 1])])
        a, b, c = result.x
    vx = -b / (2 * a) if abs(a) > 1e-10 else vertex[0]
    vy = a * vx**2 + b * vx + c if abs(a) > 1e-10 else vertex[1]
    return {'type': 'parabola', 'params': [a, b, c], 'vertex': [vx, vy], 'start': start.tolist(), 'end': end.tolist()}

def fit_geometry(points):
    shape = identify_shape(points)
    return fit_circle(points) if shape == 'circle' else fit_ellipse(points) if shape == 'ellipse' else fit_parabola(points)

def douglas_peucker(points, epsilon):
    points = np.array(points)
    if len(points) < 3:
        return [0, len(points) - 1]
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-10:
        dists = np.sqrt(np.sum((points - start)**2, axis=1))
    else:
        line_dir = line_vec / line_len
        proj_points = start + np.outer(np.dot(points - start, line_dir), line_dir)
        dists = np.sqrt(np.sum((points - proj_points)**2, axis=1))
    max_idx, max_dist = np.argmax(dists), dists.max()
    if max_dist > epsilon:
        left = douglas_peucker(points[:max_idx+1], epsilon)
        right = douglas_peucker(points[max_idx:], epsilon)
        return left[:-1] + [x + max_idx for x in right]
    return [0, len(points) - 1]

def identify_polygon(points):
    points = np.array(points)
    n = len(points)
    if n < 5:
        return 'line', [0, n-1]
    start, end = points[0], points[-1]
    segments = np.diff(points, axis=0)
    total_path = np.sum(np.sqrt(np.sum(segments**2, axis=1)))
    direct_dist = np.linalg.norm(end - start)
    if total_path > 0 and direct_dist / total_path > 0.95:
        return 'line', [0, n-1]
    is_closed = direct_dist < total_path * 0.15
    bbox_diag = np.sqrt((points[:,0].max() - points[:,0].min())**2 + (points[:,1].max() - points[:,1].min())**2)
    corner_indices = douglas_peucker(points, bbox_diag * 0.02)
    if is_closed and len(corner_indices) >= 2 and corner_indices[-1] == n - 1:
        corner_indices = corner_indices[:-1]
    num_corners = len(corner_indices)
    if is_closed:
        cx, cy = np.mean(points[:, 0]), np.mean(points[:, 1])
        distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        circularity = np.std(distances) / (np.mean(distances) + 1e-10)
        if circularity < 0.08 or (num_corners > 8 and circularity < 0.15):
            return 'circle', corner_indices
    shapes = {3: 'triangle', 4: 'square', 5: 'pentagon', 6: 'hexagon', 7: 'heptagon', 8: 'octagon'}
    if num_corners in shapes:
        return shapes[num_corners], corner_indices
    return ('circle' if num_corners > 8 else 'line'), corner_indices

def fit_polygon(points, shape_type, corner_indices):
    points = np.array(points)
    if shape_type == 'line':
        return {'type': 'line', 'start': points[0].tolist(), 'end': points[-1].tolist()}
    if shape_type == 'circle':
        x, y = points[:, 0], points[:, 1]
        cx, cy = np.mean(x), np.mean(y)
        def residuals(params):
            return np.sqrt((x - params[0])**2 + (y - params[1])**2) - params[2]
        result = least_squares(residuals, [cx, cy, np.mean(np.sqrt((x - cx)**2 + (y - cy)**2))])
        return {'type': 'circle', 'center': [float(result.x[0]), float(result.x[1])], 'radius': float(result.x[2])}
    n_sides = {'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6, 'heptagon': 7, 'octagon': 8}[shape_type]
    actual_corners = np.array([points[i] for i in corner_indices if i < len(points)])
    cx, cy = (np.mean(actual_corners[:, 0]), np.mean(actual_corners[:, 1])) if len(actual_corners) >= 2 else (np.mean(points[:, 0]), np.mean(points[:, 1]))
    R = np.mean(np.sqrt((actual_corners[:, 0] - cx)**2 + (actual_corners[:, 1] - cy)**2)) if len(actual_corners) >= 2 else np.mean(np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2))
    theta = np.arctan2(actual_corners[0][1] - cy, actual_corners[0][0] - cx) if len(actual_corners) > 0 else np.arctan2(points[0][1] - cy, points[0][0] - cx)
    vertices = [[float(cx + R * np.cos(theta + 2 * np.pi * i / n_sides)), float(cy + R * np.sin(theta + 2 * np.pi * i / n_sides))] for i in range(n_sides)]
    return {'type': shape_type, 'center': [float(cx), float(cy)], 'radius': float(R), 'rotation': float(theta), 'vertices': vertices, 'sides': n_sides}

def fit_polygon_shape(points):
    shape_type, corner_indices = identify_polygon(points)
    return fit_polygon(points, shape_type, corner_indices)

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
        #toolbar { background: white; padding: 10px 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
        button { padding: 8px 16px; border: none; border-radius: 8px; background: #007AFF; color: white; font-size: 14px; cursor: pointer; }
        button:active { background: #0051D5; }
        #clear { background: #FF3B30; }
        .tool-btn { background: #34C759; }
        .tool-btn.active { background: #248A3D; box-shadow: 0 0 0 3px rgba(36,138,61,0.3); }
        #polygon { background: #AF52DE; }
        #polygon.active { background: #8B3DB0; box-shadow: 0 0 0 3px rgba(139,61,176,0.3); }
        #freehand { background: #000; }
        #freehand.active { background: #333; box-shadow: 0 0 0 3px rgba(0,0,0,0.3); }
        #toggleStroke { background: #8E8E93; }
        .zoom-btn { background: #5856D6; padding: 8px 12px; }
        #zoomLevel { font-size: 12px; color: #666; min-width: 50px; text-align: center; }
        #colorPicker { display: none; gap: 5px; }
        #colorPicker.show { display: flex; }
        .color-opt { width: 28px; height: 28px; border-radius: 50%; border: 3px solid transparent; cursor: pointer; }
        .color-opt:hover { transform: scale(1.1); }
        .color-opt.selected { border-color: #007AFF; }
        #canvasContainer { flex: 1; position: relative; overflow: hidden; background: white; touch-action: none; }
        #canvas { position: absolute; top: 0; left: 0; touch-action: none; }
        #status { padding: 8px; background: #fff; border-top: 1px solid #ddd; text-align: center; font-size: 13px; color: #666; }
    </style>
</head>
<body>
    <div id="container">
        <div id="toolbar">
            <button id="clear">Clear</button>
            <button id="mode" class="tool-btn active">Curve</button>
            <button id="geometry" class="tool-btn">Conic</button>
            <button id="polygon">Polygon</button>
            <button id="toggleStroke">Hide Stroke</button>
            <button id="zoomOut" class="zoom-btn">−</button>
            <span id="zoomLevel">100%</span>
            <button id="zoomIn" class="zoom-btn">+</button>
            <button id="resetZoom" class="zoom-btn">Reset</button>
            <button id="freehand">Freehand</button>
            <div id="colorPicker"></div>
        </div>
        <div id="canvasContainer"><canvas id="canvas"></canvas></div>
        <div id="status">Select a tool and draw</div>
    </div>
    <script>
        const canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d'), status = document.getElementById('status');
        const colorPickerDiv = document.getElementById('colorPicker');
        const colors = ['#000000','#FF3B30','#FF9500','#FFCC00','#34C759','#007AFF','#5856D6','#AF52DE','#FF2D55','#A2845E'];
        colors.forEach((c,i) => {
            const div = document.createElement('div');
            div.className = 'color-opt' + (i === 0 ? ' selected' : '');
            div.style.background = c;
            div.onclick = () => { document.querySelectorAll('.color-opt').forEach(d => d.classList.remove('selected')); div.classList.add('selected'); freehandColor = c; };
            colorPickerDiv.appendChild(div);
        });

        let currentTool = 'bezier', editMode = false, showStroke = true, freehandColor = '#000000';
        let strokes = [], currentPoints = [], editingIdx = -1;
        let zoom = 1, offsetX = 0, offsetY = 0, dragging = null, panning = false, lastX = 0, lastY = 0, lastMove = 0, activePointerId = null;

        const updateButtons = () => {
            ['mode','geometry','polygon','freehand'].forEach(id => document.getElementById(id).classList.remove('active'));
            const map = {bezier:'mode', geometry:'geometry', polygon:'polygon', freehand:'freehand'};
            document.getElementById(map[currentTool]).classList.add('active');
            document.getElementById('mode').textContent = currentTool === 'bezier' && editMode ? 'Curve ✎' : 'Curve';
            document.getElementById('geometry').textContent = currentTool === 'geometry' && editMode ? 'Conic ✎' : 'Conic';
            document.getElementById('polygon').textContent = currentTool === 'polygon' && editMode ? 'Polygon ✎' : 'Polygon';
        };

        const resize = () => {
            const r = canvas.parentElement.getBoundingClientRect(), scale = devicePixelRatio || 2;
            canvas.width = r.width * scale; canvas.height = r.height * scale;
            canvas.style.width = r.width + 'px'; canvas.style.height = r.height + 'px';
            draw();
        };
        window.addEventListener('resize', resize); resize();

        const draw = () => {
            const scale = devicePixelRatio || 2;
            ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,canvas.width,canvas.height);
            ctx.scale(scale,scale); ctx.translate(offsetX/scale, offsetY/scale); ctx.scale(zoom,zoom);

            strokes.forEach((s, idx) => {
                const isEdit = editMode && idx === editingIdx;
                if (showStroke && s.points.length > 0 && s.tool !== 'freehand') {
                    ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1.5/zoom; ctx.lineCap = ctx.lineJoin = 'round';
                    ctx.beginPath(); ctx.moveTo(s.points[0][0], s.points[0][1]);
                    s.points.slice(1).forEach(p => ctx.lineTo(p[0], p[1])); ctx.stroke();
                }
                if (s.tool === 'freehand') {
                    ctx.strokeStyle = s.color; ctx.lineWidth = 3/zoom; ctx.lineCap = ctx.lineJoin = 'round';
                    ctx.beginPath(); if(s.points.length) { ctx.moveTo(s.points[0][0], s.points[0][1]); s.points.slice(1).forEach(p => ctx.lineTo(p[0], p[1])); } ctx.stroke();
                } else if (s.tool === 'bezier' && s.result?.segments) {
                    s.result.segments.forEach(seg => {
                        const [p0,p1,p2,p3] = seg.controls;
                        ctx.strokeStyle = '#007AFF'; ctx.lineWidth = 3/zoom; ctx.lineCap = 'round';
                        ctx.beginPath(); ctx.moveTo(p0[0],p0[1]); ctx.bezierCurveTo(p1[0],p1[1],p2[0],p2[1],p3[0],p3[1]); ctx.stroke();
                        if (isEdit) {
                            ctx.strokeStyle = '#999'; ctx.lineWidth = 1/zoom; ctx.setLineDash([5/zoom,5/zoom]);
                            ctx.beginPath(); ctx.moveTo(p0[0],p0[1]); ctx.lineTo(p1[0],p1[1]); ctx.moveTo(p2[0],p2[1]); ctx.lineTo(p3[0],p3[1]); ctx.stroke(); ctx.setLineDash([]);
                            [[p0,'#34C759'],[p3,'#34C759'],[p1,'#FF3B30'],[p2,'#FF3B30']].forEach(([p,c]) => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(p[0],p[1],8/zoom,0,Math.PI*2); ctx.fill(); });
                        }
                    });
                } else if (s.tool === 'geometry' && s.result) {
                    const g = s.result; ctx.strokeStyle = '#007AFF'; ctx.lineWidth = 3/zoom;
                    if (g.type === 'circle') { ctx.beginPath(); ctx.arc(g.params[0],g.params[1],g.params[2],0,Math.PI*2); ctx.stroke(); if(isEdit){ctx.fillStyle='#FF3B30';ctx.beginPath();ctx.arc(g.params[0],g.params[1],8/zoom,0,Math.PI*2);ctx.fill();ctx.beginPath();ctx.arc(g.startEnd[0],g.startEnd[1],8/zoom,0,Math.PI*2);ctx.fill();} }
                    else if (g.type === 'ellipse') { ctx.save(); ctx.translate(g.params[0],g.params[1]); ctx.rotate(g.params[4]); ctx.beginPath(); ctx.ellipse(0,0,g.params[2],g.params[3],0,0,Math.PI*2); ctx.stroke(); ctx.restore(); if(isEdit){ctx.fillStyle='#FF3B30';g.foci.forEach(f=>{ctx.beginPath();ctx.arc(f[0],f[1],8/zoom,0,Math.PI*2);ctx.fill();});ctx.beginPath();ctx.arc(g.startEnd[0],g.startEnd[1],8/zoom,0,Math.PI*2);ctx.fill();} }
                    else if (g.type === 'parabola') { const [a,b,c]=g.params,xMin=Math.min(g.start[0],g.end[0]),xMax=Math.max(g.start[0],g.end[0]); ctx.beginPath(); for(let x=xMin;x<=xMax;x++){const y=a*x*x+b*x+c;x===xMin?ctx.moveTo(x,y):ctx.lineTo(x,y);} ctx.stroke(); if(isEdit){ctx.fillStyle='#FF3B30';[g.vertex,g.start,g.end].forEach(p=>{ctx.beginPath();ctx.arc(p[0],p[1],8/zoom,0,Math.PI*2);ctx.fill();});} }
                } else if (s.tool === 'polygon' && s.result) {
                    const p = s.result; ctx.strokeStyle = '#AF52DE'; ctx.lineWidth = 3/zoom; ctx.lineCap = ctx.lineJoin = 'round';
                    if (p.type === 'line') { ctx.beginPath(); ctx.moveTo(p.start[0],p.start[1]); ctx.lineTo(p.end[0],p.end[1]); ctx.stroke(); if(isEdit){ctx.fillStyle='#FF3B30';[p.start,p.end].forEach(pt=>{ctx.beginPath();ctx.arc(pt[0],pt[1],8/zoom,0,Math.PI*2);ctx.fill();});} }
                    else if (p.type === 'circle') { ctx.beginPath(); ctx.arc(p.center[0],p.center[1],p.radius,0,Math.PI*2); ctx.stroke(); if(isEdit){ctx.fillStyle='#FF3B30';ctx.beginPath();ctx.arc(p.center[0],p.center[1],8/zoom,0,Math.PI*2);ctx.fill();} }
                    else if (p.vertices) { ctx.beginPath(); ctx.moveTo(p.vertices[0][0],p.vertices[0][1]); p.vertices.slice(1).forEach(v=>ctx.lineTo(v[0],v[1])); ctx.closePath(); ctx.stroke(); if(isEdit){ctx.fillStyle='#FF3B30';ctx.beginPath();ctx.arc(p.center[0],p.center[1],8/zoom,0,Math.PI*2);ctx.fill();p.vertices.forEach(v=>{ctx.beginPath();ctx.arc(v[0],v[1],6/zoom,0,Math.PI*2);ctx.fill();});} }
                }
            });
            if (currentPoints.length > 0) {
                ctx.strokeStyle = currentTool === 'freehand' ? freehandColor : '#999';
                ctx.lineWidth = (currentTool === 'freehand' ? 3 : 2)/zoom; ctx.lineCap = ctx.lineJoin = 'round';
                ctx.beginPath(); ctx.moveTo(currentPoints[0][0], currentPoints[0][1]);
                currentPoints.slice(1).forEach(p => ctx.lineTo(p[0], p[1])); ctx.stroke();
            }
        };

        const toCanvas = (cx, cy) => { const r = canvas.getBoundingClientRect(), scale = devicePixelRatio || 2; return [((cx-r.left)*scale - offsetX)/zoom/scale, ((cy-r.top)*scale - offsetY)/zoom/scale]; };

        const findCtrl = (x, y) => {
            if (editingIdx < 0) return null;
            const s = strokes[editingIdx], r = 12/zoom;
            if (s.tool === 'bezier' && s.result?.segments) for(let i=0;i<s.result.segments.length;i++) for(let j=0;j<4;j++) { const [px,py]=s.result.segments[i].controls[j]; if(Math.hypot(x-px,y-py)<r) return {t:'b',i,j}; }
            if (s.tool === 'geometry' && s.result) { const g=s.result; if(g.type==='circle'){if(Math.hypot(x-g.params[0],y-g.params[1])<r)return{t:'cc'};if(Math.hypot(x-g.startEnd[0],y-g.startEnd[1])<r)return{t:'cs'};} else if(g.type==='ellipse'){for(let i=0;i<2;i++)if(Math.hypot(x-g.foci[i][0],y-g.foci[i][1])<r)return{t:'ef',i};if(Math.hypot(x-g.startEnd[0],y-g.startEnd[1])<r)return{t:'es'};} else if(g.type==='parabola'){if(Math.hypot(x-g.vertex[0],y-g.vertex[1])<r)return{t:'pv'};if(Math.hypot(x-g.start[0],y-g.start[1])<r)return{t:'ps'};if(Math.hypot(x-g.end[0],y-g.end[1])<r)return{t:'pe'};} }
            if (s.tool === 'polygon' && s.result) { const p=s.result; if(p.type==='line'){if(Math.hypot(x-p.start[0],y-p.start[1])<r)return{t:'ls'};if(Math.hypot(x-p.end[0],y-p.end[1])<r)return{t:'le'};} else if(p.type==='circle'){if(Math.hypot(x-p.center[0],y-p.center[1])<r)return{t:'lc'};} else if(p.vertices){if(Math.hypot(x-p.center[0],y-p.center[1])<r)return{t:'pc'};for(let i=0;i<p.vertices.length;i++)if(Math.hypot(x-p.vertices[i][0],y-p.vertices[i][1])<r)return{t:'pv',i};} }
            return null;
        };

        const updPara = g => { try { const s=g.start,e=g.end,v=g.vertex,A=[[s[0]*s[0],s[0],1],[e[0]*e[0],e[0],1],[v[0]*v[0],v[0],1]],B=[s[1],e[1],v[1]],det=A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])-A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])+A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]); if(Math.abs(det)<1e-10)return; const inv=[[(A[1][1]*A[2][2]-A[1][2]*A[2][1])/det,(A[0][2]*A[2][1]-A[0][1]*A[2][2])/det,(A[0][1]*A[1][2]-A[0][2]*A[1][1])/det],[(A[1][2]*A[2][0]-A[1][0]*A[2][2])/det,(A[0][0]*A[2][2]-A[0][2]*A[2][0])/det,(A[0][2]*A[1][0]-A[0][0]*A[1][2])/det],[(A[1][0]*A[2][1]-A[1][1]*A[2][0])/det,(A[0][1]*A[2][0]-A[0][0]*A[2][1])/det,(A[0][0]*A[1][1]-A[0][1]*A[1][0])/det]]; g.params=[inv[0][0]*B[0]+inv[0][1]*B[1]+inv[0][2]*B[2],inv[1][0]*B[0]+inv[1][1]*B[1]+inv[1][2]*B[2],inv[2][0]*B[0]+inv[2][1]*B[1]+inv[2][2]*B[2]]; } catch(e){} };
        const updEllipse = g => { const [f1,f2]=g.foci,h=(f1[0]+f2[0])/2,k=(f1[1]+f2[1])/2,c=Math.hypot(f1[0]-f2[0],f1[1]-f2[1])/2,a=g.sumDist/2,b=Math.sqrt(Math.max(a*a-c*c,1)),theta=Math.atan2(f1[1]-f2[1],f1[0]-f2[0]); g.params=[h,k,a,b,theta]; const ang=Math.atan2(g.startEnd[1]-k,g.startEnd[0]-h)-theta; g.startEnd=[h+a*Math.cos(ang)*Math.cos(theta)-b*Math.sin(ang)*Math.sin(theta),k+a*Math.cos(ang)*Math.sin(theta)+b*Math.sin(ang)*Math.cos(theta)]; };
        const updPoly = p => { if(!p.vertices)return; for(let i=0;i<p.sides;i++){const ang=p.rotation+2*Math.PI*i/p.sides;p.vertices[i]=[p.center[0]+p.radius*Math.cos(ang),p.center[1]+p.radius*Math.sin(ang)];} };

        canvas.addEventListener('pointerdown', e => {
            e.preventDefault(); if (activePointerId !== null) return;
            activePointerId = e.pointerId; canvas.setPointerCapture(e.pointerId);
            const [x,y] = toCanvas(e.clientX, e.clientY); lastX = e.clientX; lastY = e.clientY;
            if (editMode && editingIdx >= 0) { dragging = findCtrl(x,y); panning = !dragging; }
            else { currentPoints = [[x,y]]; lastMove = Date.now(); draw(); }
        });

        canvas.addEventListener('pointermove', e => {
            e.preventDefault(); if (activePointerId !== null && e.pointerId !== activePointerId) return;
            const [x,y] = toCanvas(e.clientX, e.clientY);
            if (panning) { offsetX += e.clientX - lastX; offsetY += e.clientY - lastY; lastX = e.clientX; lastY = e.clientY; draw(); }
            else if (dragging && editingIdx >= 0) {
                const s = strokes[editingIdx], d = dragging;
                if (d.t === 'b') { const segs = s.result.segments; segs[d.i].controls[d.j] = [x,y]; if(d.j===3 && d.i<segs.length-1) segs[d.i+1].controls[0]=[x,y]; if(d.j===0 && d.i>0) segs[d.i-1].controls[3]=[x,y]; }
                else if (d.t === 'cc') { const g=s.result, dx=x-g.params[0], dy=y-g.params[1]; g.params[0]=x; g.params[1]=y; g.center=[x,y]; g.startEnd[0]+=dx; g.startEnd[1]+=dy; }
                else if (d.t === 'cs') { const g=s.result, cx=g.params[0], cy=g.params[1]; g.params[2]=Math.hypot(x-cx,y-cy); g.radius=g.params[2]; const ang=Math.atan2(y-cy,x-cx); g.startEnd=[cx+g.radius*Math.cos(ang),cy+g.radius*Math.sin(ang)]; }
                else if (d.t === 'ef') { s.result.foci[d.i]=[x,y]; updEllipse(s.result); }
                else if (d.t === 'es') { const g=s.result,[h,k,a,b,theta]=g.params,ang=Math.atan2(y-k,x-h)-theta; g.startEnd=[h+a*Math.cos(ang)*Math.cos(theta)-b*Math.sin(ang)*Math.sin(theta),k+a*Math.cos(ang)*Math.sin(theta)+b*Math.sin(ang)*Math.cos(theta)]; }
                else if (d.t === 'pv') { s.result.vertex=[x,y]; updPara(s.result); }
                else if (d.t === 'ps') { s.result.start=[x,y]; updPara(s.result); }
                else if (d.t === 'pe') { s.result.end=[x,y]; updPara(s.result); }
                else if (d.t === 'ls') s.result.start=[x,y];
                else if (d.t === 'le') s.result.end=[x,y];
                else if (d.t === 'lc') s.result.center=[x,y];
                else if (d.t === 'pc') { const dx=x-s.result.center[0],dy=y-s.result.center[1]; s.result.center=[x,y]; s.result.vertices.forEach(v=>{v[0]+=dx;v[1]+=dy;}); }
                else if (d.t === 'pv') { const p=s.result; p.radius=Math.hypot(x-p.center[0],y-p.center[1]); p.rotation=Math.atan2(y-p.center[1],x-p.center[0])-2*Math.PI*d.i/p.sides; updPoly(p); }
                draw();
            } else if (!editMode && currentPoints.length > 0 && activePointerId !== null) {
                (e.getCoalescedEvents?.() || [e]).forEach(ce => currentPoints.push(toCanvas(ce.clientX, ce.clientY)));
                lastMove = Date.now(); draw();
            }
        });

        canvas.addEventListener('pointerup', async e => {
            e.preventDefault(); if (e.pointerId !== activePointerId) return;
            if (!editMode && currentPoints.length > 2) {
                if (currentTool === 'freehand') { strokes.push({tool:'freehand', points:[...currentPoints], result:null, color:freehandColor}); status.textContent = 'Freehand stroke added'; }
                else if (Date.now() - lastMove >= 500) {
                    let result = null;
                    if (currentTool === 'bezier') { result = await (await fetch('/fit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({points:currentPoints})})).json(); status.textContent = `Fitted ${result.segments.length} segment(s)`; }
                    else if (currentTool === 'geometry') { result = await (await fetch('/fit_geometry',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({points:currentPoints})})).json(); status.textContent = `Recognized: ${result.type}`; }
                    else if (currentTool === 'polygon') { result = await (await fetch('/fit_polygon',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({points:currentPoints})})).json(); status.textContent = `Recognized: ${result.type}`; }
                    if (result) strokes.push({tool:currentTool, points:[...currentPoints], result, color:null});
                }
            }
            currentPoints = []; dragging = null; panning = false; activePointerId = null;
            canvas.releasePointerCapture(e.pointerId); draw();
        });

        canvas.addEventListener('pointercancel', e => { if(e.pointerId===activePointerId){activePointerId=null;dragging=null;panning=false;currentPoints=[];} });
        ['touchstart','touchmove','touchend'].forEach(ev => canvas.addEventListener(ev, e => e.preventDefault(), {passive:false}));

        canvas.addEventListener('wheel', e => {
            e.preventDefault();
            const [ox,oy] = toCanvas(e.clientX, e.clientY), scale = devicePixelRatio || 2;
            zoom = Math.max(0.1, Math.min(5, zoom * (e.deltaY > 0 ? 0.9 : 1.1)));
            offsetX = e.clientX * scale - ox * zoom * scale; offsetY = e.clientY * scale - oy * zoom * scale;
            document.getElementById('zoomLevel').textContent = Math.round(zoom*100)+'%'; draw();
        }, {passive:false});

        const findLast = tool => { for(let i=strokes.length-1;i>=0;i--) if(strokes[i].tool===tool && strokes[i].result) return i; return -1; };

        document.getElementById('clear').onclick = () => { strokes=[]; currentPoints=[]; editMode=false; editingIdx=-1; zoom=1; offsetX=offsetY=0; document.getElementById('zoomLevel').textContent='100%'; colorPickerDiv.classList.remove('show'); status.textContent='Canvas cleared'; updateButtons(); draw(); };

        document.getElementById('mode').onclick = () => {
            if (currentTool === 'bezier') { const idx=findLast('bezier'); if(idx>=0){editMode=!editMode; editingIdx=editMode?idx:-1;} }
            else { currentTool='bezier'; editMode=false; editingIdx=-1; colorPickerDiv.classList.remove('show'); status.textContent='Curve: draw and hold 0.5s'; }
            updateButtons(); draw();
        };
        document.getElementById('geometry').onclick = () => {
            if (currentTool === 'geometry') { const idx=findLast('geometry'); if(idx>=0){editMode=!editMode; editingIdx=editMode?idx:-1;} }
            else { currentTool='geometry'; editMode=false; editingIdx=-1; colorPickerDiv.classList.remove('show'); status.textContent='Conic: draw and hold 0.5s'; }
            updateButtons(); draw();
        };
        document.getElementById('polygon').onclick = () => {
            if (currentTool === 'polygon') { const idx=findLast('polygon'); if(idx>=0){editMode=!editMode; editingIdx=editMode?idx:-1;} }
            else { currentTool='polygon'; editMode=false; editingIdx=-1; colorPickerDiv.classList.remove('show'); status.textContent='Polygon: draw and hold 0.5s'; }
            updateButtons(); draw();
        };
        document.getElementById('freehand').onclick = () => {
            if (currentTool === 'freehand') colorPickerDiv.classList.toggle('show');
            else { currentTool='freehand'; editMode=false; editingIdx=-1; status.textContent='Freehand: draw freely, click again for colors'; }
            updateButtons(); draw();
        };
        document.getElementById('toggleStroke').onclick = () => { showStroke=!showStroke; document.getElementById('toggleStroke').textContent=showStroke?'Hide Stroke':'Show Stroke'; draw(); };
        document.getElementById('zoomIn').onclick = () => { const cx=canvas.width/2,cy=canvas.height/2,scale=devicePixelRatio||2,[ox,oy]=[(cx-offsetX)/zoom/scale,(cy-offsetY)/zoom/scale]; zoom=Math.min(5,zoom*1.3); offsetX=cx-ox*zoom*scale; offsetY=cy-oy*zoom*scale; document.getElementById('zoomLevel').textContent=Math.round(zoom*100)+'%'; draw(); };
        document.getElementById('zoomOut').onclick = () => { const cx=canvas.width/2,cy=canvas.height/2,scale=devicePixelRatio||2,[ox,oy]=[(cx-offsetX)/zoom/scale,(cy-offsetY)/zoom/scale]; zoom=Math.max(0.1,zoom/1.3); offsetX=cx-ox*zoom*scale; offsetY=cy-oy*zoom*scale; document.getElementById('zoomLevel').textContent=Math.round(zoom*100)+'%'; draw(); };
        document.getElementById('resetZoom').onclick = () => { zoom=1; offsetX=offsetY=0; document.getElementById('zoomLevel').textContent='100%'; draw(); };
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
    print("Server: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False) 