import os, json, numpy as np, laspy, rasterio
from collections import defaultdict
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from Tree_Segmentation.config import load_config

# ---------------- I/O ----------------
config = load_config(None)
tif_folder    = os.path.join(config['paths']['result_path'], 'Building Candidate Images', 'building images')
json_folder   = os.path.join(config['paths']['result_path'], 'building box')
las_folder    = os.path.join(config['paths']['result_path'], 'LAS')
output_folder = os.path.join(config['paths']['result_path'], 'Building 3D model')
os.makedirs(output_folder, exist_ok=True)

# ---------------- KNOBS ----------------
Z_BIN            = 0.02
WEIGHT_GAMMA     = 2.0
FALLBACK_STEPS   = 3
FALLBACK_STEP_FR = 0.25
NN_BBOX_MARGIN   = 1.0
NN_PERC          = 90.0

PER_POLY_BOTTOM_Z = False    
ID_WIDTH          = 8         
ROUND_DECIMALS    = 6        

# ---------------- 
def polygon_area2(coords_xy):
    a = 0.0
    for i in range(len(coords_xy)):
        x1, y1 = coords_xy[i]
        x2, y2 = coords_xy[(i+1) % len(coords_xy)]
        a += x1*y2 - x2*y1
    return 0.5*a

def ensure_ccw(coords_xy):
    if len(coords_xy) < 3: return coords_xy
    return coords_xy if polygon_area2(coords_xy) > 0 else list(reversed(coords_xy))

# ---------------- JSON/OBJ helpers ----------------
def write_obj_file_with_highlighted_corners(filename, vertices, faces, corner_vertices):
    with open(filename, 'w', encoding='utf-8') as file:
        for v in vertices: file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for v in corner_vertices: file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in faces: file.write(f"f {' '.join(map(str, f))}\n")

def create_faces(vertices_bottom, vertices_top):
    n = len(vertices_bottom)
    if n < 3: return []
    faces = []
    faces.append([i + 1 for i in range(n)])             # bottom
    faces.append([i + n + 1 for i in range(n)])         # roof
    for i in range(n):                                  # walls (quads)
        ni = (i + 1) % n
        faces.append([i + 1, ni + 1, ni + n + 1, i + n + 1])
    return faces

def save_model_to_named_json(filename, bottoms_list, roofs_list, walls_list):

    data = {"type": "json", "points": {}, "buildings": [{"geometry": {"roof": {}, "walls": {}, "bottom": {}}}]}
    building = data["buildings"][0]
    points   = data["points"]
    point_id_counter = 1
    point_lookup = {}

    def get_vid(pt):
        nonlocal point_id_counter
        key = (round(float(pt[0]), ROUND_DECIMALS),
               round(float(pt[1]), ROUND_DECIMALS),
               round(float(pt[2]), ROUND_DECIMALS))
        vid = point_lookup.get(key)
        if vid is None:
            vid = f"V{point_id_counter:0{ID_WIDTH}d}"
            point_lookup[key] = vid
            points[vid] = [float(pt[0]), float(pt[1]), float(pt[2])]
            point_id_counter += 1
        return vid

    def loop_ids(pts3d):
        ids = [get_vid(p) for p in pts3d]
        if len(ids) >= 3 and ids[0] != ids[-1]:
            ids.append(ids[0])        
        return ids

    # bottoms (b1, b2, ...)
    for i, bottom in enumerate(bottoms_list, 1):
        building["geometry"]["bottom"][f"b{i}"] = {"faces": [loop_ids(bottom)]}

    # roofs (r1, r2, ...)
    for i, roof in enumerate(roofs_list, 1):
        building["geometry"]["roof"][f"r{i}"] = {"faces": [loop_ids(roof)]}

    # walls (w1, w2, ...)
    for i, wall in enumerate(walls_list, 1):
        building["geometry"]["walls"][f"w{i}"] = {"faces": [loop_ids(wall)]}

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# ----------------
def angle_between_vectors(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12: return np.pi
    cos_val = float(np.dot(v1, v2) / (n1 * n2))
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.arccos(cos_val))

def filter_non_collinear_corners(polygon, collinear_tol_deg=5.0):
    coords = list(polygon.exterior.coords[:-1])
    if len(coords) <= 3: return coords
    out = []; T = np.radians(collinear_tol_deg)
    for i in range(len(coords)):
        p1, p2, p3 = coords[i - 1], coords[i], coords[(i + 1) % len(coords)]
        ang = angle_between_vectors(p1, p2, p3)
        if abs(np.pi - ang) > T: out.append(p2)
    if len(out) < 3: out = coords
    return out

# ----------------
def points_in_polygon_strict(polygon, points_xyz):
    minx, miny, maxx, maxy = polygon.bounds
    cand = points_xyz[(points_xyz[:,0] >= minx) & (points_xyz[:,0] <= maxx) &
                      (points_xyz[:,1] >= miny) & (points_xyz[:,1] <= maxy)]
    if cand.shape[0] == 0: return cand
    inside = []
    for x, y, z in cand:
        if polygon.contains(Point(x, y)): inside.append([x, y, z])
    return np.array(inside, dtype=float)

def weighted_mode_from_cells(z_list, w_list, z_bin=Z_BIN):
    z_arr = np.array(z_list, dtype=float)
    w_arr = np.array(w_list, dtype=float)
    mask = ~np.isnan(z_arr)
    z_arr = z_arr[mask]; w_arr = w_arr[mask]
    if z_arr.size == 0: return None
    if z_bin <= 0:
        order = np.argsort(z_arr); z_s, w_s = z_arr[order], w_arr[order]
        c = np.cumsum(w_s); cutoff = 0.5 * c[-1]
        return float(z_s[np.searchsorted(c, cutoff)])
    bins = np.floor(z_arr / z_bin).astype(np.int64)
    sumw = defaultdict(float); sumzw = defaultdict(float)
    for b, z, w in zip(bins, z_arr, w_arr):
        if w <= 0: continue
        sumw[b]  += w
        sumzw[b] += w * z
    if not sumw: return None
    b_star = max(sumw, key=lambda b: sumw[b])
    z_hat = sumzw[b_star] / max(sumw[b_star], 1e-12)
    return float(z_hat)

def estimate_box_height_weighted(box, points_xyz,
                                 grid_size_large=(10,10), grid_size_small=(3,3),
                                 area_threshold=4.0):
    coords = [(p['x'], p['y']) for p in box]
    poly = Polygon(coords)
    area = float(poly.area)
    if area <= 0: return None

    grid_size = grid_size_large if area > area_threshold else grid_size_small
    min_x, min_y, max_x, max_y = poly.bounds
    cx, cy = poly.centroid.x, poly.centroid.y
    dmax = max(np.hypot(max_x - cx, max_y - cy), 1e-6)

    def one_pass(polygon):
        pts = points_in_polygon_strict(polygon, points_xyz)
        if pts.shape[0] == 0:
            return None
        z_cells = []; w_cells = []
        min_x2, min_y2, max_x2, max_y2 = polygon.bounds
        gx2 = (max_x2 - min_x2) / grid_size[0]; gy2 = (max_y2 - min_y2) / grid_size[1]
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                cell = Polygon([
                    (min_x2 + i*gx2,     min_y2 + j*gy2),
                    (min_x2 + (i+1)*gx2, min_y2 + j*gy2),
                    (min_x2 + (i+1)*gx2, min_y2 + (j+1)*gy2),
                    (min_x2 + i*gx2,     min_y2 + (j+1)*gy2),
                ])
                x0, y0, x1, y1 = cell.bounds
                mask = (pts[:,0] >= x0) & (pts[:,0] <= x1) & (pts[:,1] >= y0) & (pts[:,1] <= y1)
                cand = pts[mask]
                if cand.shape[0] == 0:
                    z_cells.append(np.nan); w_cells.append(0.0); continue
                zs = [z for x, y, z in cand if cell.contains(Point(x, y))]
                if len(zs) == 0:
                    z_cells.append(np.nan); w_cells.append(0.0); continue
                ccx = 0.5*(x0+x1); ccy = 0.5*(y0+y1)
                d   = np.hypot(ccx - cx, ccy - cy)
                base = max(0.0, 1.0 - d / dmax)
                w    = base ** WEIGHT_GAMMA
                z_cells.append(max(zs))
                w_cells.append(w)
        return weighted_mode_from_cells(z_cells, w_cells, z_bin=Z_BIN)

    z0 = one_pass(poly)
    if z0 is not None: return float(z0)

    step = FALLBACK_STEP_FR * np.sqrt(area)
    for k in range(1, FALLBACK_STEPS+1):
        pb = poly.buffer(k * step)
        z_b = one_pass(pb)
        if z_b is not None: return float(z_b)

    x0, y0, x1, y1 = min_x - NN_BBOX_MARGIN, min_y - NN_BBOX_MARGIN, max_x + NN_BBOX_MARGIN, max_y + NN_BBOX_MARGIN
    cand = points_xyz[(points_xyz[:,0] >= x0) & (points_xyz[:,0] <= x1) &
                      (points_xyz[:,1] >= y0) & (points_xyz[:,1] <= y1)]
    if cand.shape[0] == 0: return float(np.nan)
    cx, cy = poly.centroid.x, poly.centroid.y
    d = np.hypot(cand[:,0]-cx, cand[:,1]-cy); d = np.maximum(d, 1e-3)
    w = 1.0 / d
    order = np.argsort(cand[:,2]); z_s, w_s = cand[order,2], w[order]
    cw = np.cumsum(w_s); thr = 0.01 * NN_PERC * cw[-1]
    idx = np.searchsorted(cw, thr)
    return float(z_s[min(idx, len(z_s)-1)])

# ---------------- MAIN ----------------
for filename in os.listdir(json_folder):
    if not filename.endswith(".json"): continue

    parts = os.path.splitext(filename)[0].split('_')
    if len(parts) < 2:
        print(f"Skip {filename}: unexpected name."); continue

    building_id = parts[1]
    json_path = os.path.join(json_folder, filename)
    tif_path  = os.path.join(tif_folder, f"building_{building_id}.tif")
    las_path  = os.path.join(las_folder, f"{building_id}_point_cloud.las")

    if not (os.path.exists(tif_path) and os.path.exists(las_path)):
        print(f"Skipping {building_id}: Missing TIFF or LAS file."); continue

    with open(json_path, 'r', encoding='utf-8') as f:
        boxes_data = json.load(f)
    if not boxes_data:
        print(f"Skipping {building_id}: Empty JSON file."); continue

    with rasterio.open(tif_path) as tif:
        T = tif.transform

    # pixel -> geo
    geo_boxes = []
    for box in boxes_data:
        gb = [{'x': (T * (p['x'], p['y']))[0], 'y': (T * (p['x'], p['y']))[1]} for p in box]
        geo_boxes.append(gb)

    las = laspy.read(las_path)
    pts_xyz = np.column_stack([las.x, las.y, las.z]).astype(float)

    min_z_value_global = float(np.min(pts_xyz[:,2]))

    # --- z box ---
    z_values_per_box = []
    areas_per_box    = []
    for box in geo_boxes:
        z_i = estimate_box_height_weighted(box, pts_xyz)
        if (z_i is None) or (np.isnan(z_i)):
            z_i = float(np.percentile(pts_xyz[:,2], NN_PERC))
        z_values_per_box.append(float(z_i))
        poly = Polygon([(p['x'], p['y']) for p in box]); areas_per_box.append(float(poly.area))

    z_values_per_box = np.array(z_values_per_box, dtype=float)
    areas_per_box    = np.array(areas_per_box, dtype=float)

    # --- DBSCAN z ---
    db = DBSCAN(eps=1.0, min_samples=2).fit(z_values_per_box.reshape(-1,1))
    labels_full = db.labels_

    # peak z 
    cluster_peak_z = {}
    for lab in np.unique(labels_full):
        if lab < 0: continue
        idxs = np.where(labels_full == lab)[0]
        if idxs.size == 0: continue
        zs  = z_values_per_box[idxs]
        ars = areas_per_box[idxs]
        cluster_peak_z[lab] = float(zs[np.argmax(ars)])

    # --- Merge  ---
    merged_polygons = []
    for lab in np.unique(labels_full):
        if lab >= 0:
            polys = [Polygon([(p['x'], p['y']) for p in geo_boxes[i]])
                     for i in range(len(geo_boxes)) if labels_full[i] == lab]
            if not polys: continue
            merged = unary_union(polys)
            z = cluster_peak_z[lab]
            if isinstance(merged, MultiPolygon):
                for poly in merged.geoms:
                    if not poly.is_empty and poly.area > 0:
                        merged_polygons.append((poly, z))
            else:
                if merged.area > 0: merged_polygons.append((merged, z))
        else:
            for i in np.where(labels_full == -1)[0]:
                poly = Polygon([(p['x'], p['y']) for p in geo_boxes[i]])
                if not poly.is_empty and poly.area > 0:
                    merged_polygons.append((poly, float(z_values_per_box[i])))

    # --- Build 3D ---
    filename_obj  = os.path.join(output_folder, f"building_{building_id}_3d_frame.obj")
    filename_json = os.path.join(output_folder, f"building_{building_id}_3d_frame.json")

    vertices_all, faces_all, corner_vertices = [], [], []
    vertex_offset = 0
    bottoms_list, roofs_list, walls_list = [], [], []

    for polygon, max_z_value in merged_polygons:
        if polygon.is_empty or polygon.geom_type != 'Polygon': continue


        coords = filter_non_collinear_corners(polygon)
        if len(coords) < 3: continue
        coords = ensure_ccw(coords)


        if PER_POLY_BOTTOM_Z:
            poly2 = Polygon(coords)
            pts_local = points_in_polygon_strict(poly2, pts_xyz)
            min_z_value = float(np.percentile(pts_local[:,2], 10.0)) if pts_local.shape[0] > 0 else min_z_value_global
        else:
            min_z_value = min_z_value_global

        vertices_bottom = [(x, y, min_z_value) for x, y in coords]
        vertices_top    = [(x, y, max_z_value) for x, y in coords]
        faces           = create_faces(vertices_bottom, vertices_top)
        if not faces: continue


        vertices = vertices_bottom + vertices_top
        vertices_all.extend(vertices)
        faces_all.extend([[vertex_offset + vi for vi in face] for face in faces])
        corner_vertices.extend(vertices_bottom)
        vertex_offset += len(vertices)

        # JSON lists
        bottoms_list.append(vertices_bottom)
        roofs_list.append(vertices_top)

        n = len(vertices_bottom)
        for i in range(n):
            ni = (i + 1) % n

            walls_list.append([vertices_top[i], vertices_bottom[i], vertices_bottom[ni], vertices_top[ni]])


    write_obj_file_with_highlighted_corners(filename_obj, vertices_all, faces_all, corner_vertices)
    print(f"3D frame OBJ file created: {filename_obj}")

    save_model_to_named_json(filename_json, bottoms_list, roofs_list, walls_list)
    print(f"Structured JSON model saved: {filename_json}")
