import json
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
from collections import defaultdict

import laspy
import numpy as np
import rasterio
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN

from Tree_Segmentation.config import load_config


# ---------------- I/O ----------------
config = load_config(None)
tif_folder = os.path.join(config["paths"]["result_path"], "Building Candidate Images", "building images")
json_folder = os.path.join(config["paths"]["result_path"], "building box")
las_folder = os.path.join(config["paths"]["result_path"], "LAS")
output_folder = os.path.join(config["paths"]["result_path"], "Building CityGML")
os.makedirs(output_folder, exist_ok=True)


# ---------------- KNOBS ----------------
Z_BIN = 0.02
WEIGHT_GAMMA = 2.0
FALLBACK_STEPS = 3
FALLBACK_STEP_FR = 0.25
NN_BBOX_MARGIN = 1.0
NN_PERC = 90.0

PER_POLY_BOTTOM_Z = False
ROUND_DECIMALS = 6
CITYGML_VERSION = "2.0"


# ---------------- GEOMETRY HELPERS ----------------
def polygon_area2(coords_xy):
    area = 0.0
    for i in range(len(coords_xy)):
        x1, y1 = coords_xy[i]
        x2, y2 = coords_xy[(i + 1) % len(coords_xy)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def ensure_ccw(coords_xy):
    if len(coords_xy) < 3:
        return coords_xy
    return coords_xy if polygon_area2(coords_xy) > 0 else list(reversed(coords_xy))


def angle_between_vectors(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return np.pi
    cos_val = float(np.dot(v1, v2) / (n1 * n2))
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.arccos(cos_val))


def filter_non_collinear_corners(polygon, collinear_tol_deg=5.0):
    coords = list(polygon.exterior.coords[:-1])
    if len(coords) <= 3:
        return coords

    out = []
    tolerance = np.radians(collinear_tol_deg)
    for i in range(len(coords)):
        p1 = coords[i - 1]
        p2 = coords[i]
        p3 = coords[(i + 1) % len(coords)]
        angle = angle_between_vectors(p1, p2, p3)
        if abs(np.pi - angle) > tolerance:
            out.append(p2)

    return out if len(out) >= 3 else coords


def points_in_polygon_strict(polygon, points_xyz):
    minx, miny, maxx, maxy = polygon.bounds
    cand = points_xyz[
        (points_xyz[:, 0] >= minx)
        & (points_xyz[:, 0] <= maxx)
        & (points_xyz[:, 1] >= miny)
        & (points_xyz[:, 1] <= maxy)
    ]
    if cand.shape[0] == 0:
        return cand

    inside = []
    for x, y, z in cand:
        if polygon.contains(Point(x, y)):
            inside.append([x, y, z])
    return np.array(inside, dtype=float)


def weighted_mode_from_cells(z_list, w_list, z_bin=Z_BIN):
    z_arr = np.array(z_list, dtype=float)
    w_arr = np.array(w_list, dtype=float)
    mask = ~np.isnan(z_arr)
    z_arr = z_arr[mask]
    w_arr = w_arr[mask]
    if z_arr.size == 0:
        return None

    if z_bin <= 0:
        order = np.argsort(z_arr)
        z_s = z_arr[order]
        w_s = w_arr[order]
        c = np.cumsum(w_s)
        cutoff = 0.5 * c[-1]
        return float(z_s[np.searchsorted(c, cutoff)])

    bins = np.floor(z_arr / z_bin).astype(np.int64)
    sumw = defaultdict(float)
    sumzw = defaultdict(float)
    for b, z, w in zip(bins, z_arr, w_arr):
        if w <= 0:
            continue
        sumw[b] += w
        sumzw[b] += w * z

    if not sumw:
        return None
    b_star = max(sumw, key=lambda b: sumw[b])
    return float(sumzw[b_star] / max(sumw[b_star], 1e-12))


def estimate_box_height_weighted(
    box,
    points_xyz,
    grid_size_large=(10, 10),
    grid_size_small=(3, 3),
    area_threshold=4.0,
):
    coords = [(p["x"], p["y"]) for p in box]
    poly = Polygon(coords)
    area = float(poly.area)
    if area <= 0:
        return None

    grid_size = grid_size_large if area > area_threshold else grid_size_small
    min_x, min_y, max_x, max_y = poly.bounds
    cx = poly.centroid.x
    cy = poly.centroid.y
    dmax = max(np.hypot(max_x - cx, max_y - cy), 1e-6)

    def one_pass(polygon):
        pts = points_in_polygon_strict(polygon, points_xyz)
        if pts.shape[0] == 0:
            return None

        z_cells = []
        w_cells = []
        min_x2, min_y2, max_x2, max_y2 = polygon.bounds
        gx2 = (max_x2 - min_x2) / grid_size[0]
        gy2 = (max_y2 - min_y2) / grid_size[1]

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                cell = Polygon(
                    [
                        (min_x2 + i * gx2, min_y2 + j * gy2),
                        (min_x2 + (i + 1) * gx2, min_y2 + j * gy2),
                        (min_x2 + (i + 1) * gx2, min_y2 + (j + 1) * gy2),
                        (min_x2 + i * gx2, min_y2 + (j + 1) * gy2),
                    ]
                )
                x0, y0, x1, y1 = cell.bounds
                mask = (
                    (pts[:, 0] >= x0)
                    & (pts[:, 0] <= x1)
                    & (pts[:, 1] >= y0)
                    & (pts[:, 1] <= y1)
                )
                cand = pts[mask]
                if cand.shape[0] == 0:
                    z_cells.append(np.nan)
                    w_cells.append(0.0)
                    continue

                zs = [z for x, y, z in cand if cell.contains(Point(x, y))]
                if len(zs) == 0:
                    z_cells.append(np.nan)
                    w_cells.append(0.0)
                    continue

                ccx = 0.5 * (x0 + x1)
                ccy = 0.5 * (y0 + y1)
                d = np.hypot(ccx - cx, ccy - cy)
                base = max(0.0, 1.0 - d / dmax)
                w = base**WEIGHT_GAMMA
                z_cells.append(max(zs))
                w_cells.append(w)

        return weighted_mode_from_cells(z_cells, w_cells, z_bin=Z_BIN)

    z0 = one_pass(poly)
    if z0 is not None:
        return float(z0)

    step = FALLBACK_STEP_FR * np.sqrt(area)
    for k in range(1, FALLBACK_STEPS + 1):
        z_b = one_pass(poly.buffer(k * step))
        if z_b is not None:
            return float(z_b)

    x0 = min_x - NN_BBOX_MARGIN
    y0 = min_y - NN_BBOX_MARGIN
    x1 = max_x + NN_BBOX_MARGIN
    y1 = max_y + NN_BBOX_MARGIN
    cand = points_xyz[
        (points_xyz[:, 0] >= x0)
        & (points_xyz[:, 0] <= x1)
        & (points_xyz[:, 1] >= y0)
        & (points_xyz[:, 1] <= y1)
    ]
    if cand.shape[0] == 0:
        return float(np.nan)

    cx = poly.centroid.x
    cy = poly.centroid.y
    d = np.hypot(cand[:, 0] - cx, cand[:, 1] - cy)
    d = np.maximum(d, 1e-3)
    w = 1.0 / d
    order = np.argsort(cand[:, 2])
    z_s = cand[order, 2]
    w_s = w[order]
    cw = np.cumsum(w_s)
    thr = 0.01 * NN_PERC * cw[-1]
    idx = np.searchsorted(cw, thr)
    return float(z_s[min(idx, len(z_s) - 1)])


def build_merged_polygons(geo_boxes, points_xyz):
    min_z_value_global = float(np.min(points_xyz[:, 2]))

    z_values_per_box = []
    areas_per_box = []
    for box in geo_boxes:
        z_i = estimate_box_height_weighted(box, points_xyz)
        if (z_i is None) or np.isnan(z_i):
            z_i = float(np.percentile(points_xyz[:, 2], NN_PERC))
        z_values_per_box.append(float(z_i))
        poly = Polygon([(p["x"], p["y"]) for p in box])
        areas_per_box.append(float(poly.area))

    z_values_per_box = np.array(z_values_per_box, dtype=float)
    areas_per_box = np.array(areas_per_box, dtype=float)

    db = DBSCAN(eps=1.0, min_samples=2).fit(z_values_per_box.reshape(-1, 1))
    labels_full = db.labels_

    cluster_peak_z = {}
    for lab in np.unique(labels_full):
        if lab < 0:
            continue
        idxs = np.where(labels_full == lab)[0]
        zs = z_values_per_box[idxs]
        areas = areas_per_box[idxs]
        cluster_peak_z[lab] = float(zs[np.argmax(areas)])

    merged_polygons = []
    for lab in np.unique(labels_full):
        if lab >= 0:
            polys = [
                Polygon([(p["x"], p["y"]) for p in geo_boxes[i]])
                for i in range(len(geo_boxes))
                if labels_full[i] == lab
            ]
            if not polys:
                continue
            merged = unary_union(polys)
            z = cluster_peak_z[lab]
            if isinstance(merged, MultiPolygon):
                for poly in merged.geoms:
                    if not poly.is_empty and poly.area > 0:
                        merged_polygons.append((poly, z))
            elif merged.area > 0:
                merged_polygons.append((merged, z))
        else:
            for i in np.where(labels_full == -1)[0]:
                poly = Polygon([(p["x"], p["y"]) for p in geo_boxes[i]])
                if not poly.is_empty and poly.area > 0:
                    merged_polygons.append((poly, float(z_values_per_box[i])))

    solids = []
    for polygon, roof_z in merged_polygons:
        if polygon.is_empty or polygon.geom_type != "Polygon":
            continue
        coords = filter_non_collinear_corners(polygon)
        if len(coords) < 3:
            continue
        coords = ensure_ccw(coords)

        if PER_POLY_BOTTOM_Z:
            poly2 = Polygon(coords)
            pts_local = points_in_polygon_strict(poly2, points_xyz)
            if pts_local.shape[0] > 0:
                bottom_z = float(np.percentile(pts_local[:, 2], 10.0))
            else:
                bottom_z = min_z_value_global
        else:
            bottom_z = min_z_value_global

        bottom = [(x, y, bottom_z) for x, y in coords]
        roof = [(x, y, roof_z) for x, y in coords]
        walls = []
        for i in range(len(coords)):
            ni = (i + 1) % len(coords)
            walls.append([bottom[i], bottom[ni], roof[ni], roof[i]])

        solids.append({"bottom": bottom, "roof": roof, "walls": walls})

    return solids


# ---------------- CITYGML HELPERS ----------------
NS = {
    "core": "http://www.opengis.net/citygml/2.0",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "gml": "http://www.opengis.net/gml",
    "xlink": "http://www.w3.org/1999/xlink",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def qn(prefix, tag):
    return f"{{{NS[prefix]}}}{tag}"


def fmt(v):
    return f"{float(v):.{ROUND_DECIMALS}f}".rstrip("0").rstrip(".")


def pos_list(points):
    closed = list(points)
    if closed[0] != closed[-1]:
        closed.append(closed[0])
    return " ".join(" ".join(fmt(v) for v in pt) for pt in closed)


def add_polygon(parent, polygon_id, points):
    member = ET.SubElement(parent, qn("gml", "surfaceMember"))
    polygon = ET.SubElement(member, qn("gml", "Polygon"), {qn("gml", "id"): polygon_id})
    exterior = ET.SubElement(polygon, qn("gml", "exterior"))
    ring = ET.SubElement(exterior, qn("gml", "LinearRing"))
    ET.SubElement(ring, qn("gml", "posList"), {"srsDimension": "3"}).text = pos_list(points)


def add_surface(building_el, surface_tag, surface_id, points):
    bounded_by = ET.SubElement(building_el, qn("bldg", "boundedBy"))
    surface = ET.SubElement(bounded_by, qn("bldg", surface_tag), {qn("gml", "id"): surface_id})
    lod2_multi_surface = ET.SubElement(surface, qn("bldg", "lod2MultiSurface"))
    multi_surface = ET.SubElement(lod2_multi_surface, qn("gml", "MultiSurface"))
    polygon_id = f"{surface_id}_poly"
    add_polygon(multi_surface, polygon_id, points)
    return polygon_id


def add_solid(building_el, building_id, polygon_ids):
    lod2_solid = ET.SubElement(building_el, qn("bldg", "lod2Solid"))
    solid = ET.SubElement(lod2_solid, qn("gml", "Solid"), {qn("gml", "id"): f"{building_id}_solid"})
    exterior = ET.SubElement(solid, qn("gml", "exterior"))
    composite_surface = ET.SubElement(exterior, qn("gml", "CompositeSurface"))
    for polygon_id in polygon_ids:
        ET.SubElement(
            composite_surface,
            qn("gml", "surfaceMember"),
            {qn("xlink", "href"): f"#{polygon_id}"},
        )


def compute_envelope(solids):
    pts = []
    for solid in solids:
        pts.extend(solid["bottom"])
        pts.extend(solid["roof"])
        for wall in solid["walls"]:
            pts.extend(wall)
    arr = np.array(pts, dtype=float)
    return arr.min(axis=0), arr.max(axis=0)


def crs_to_srs_name(crs):
    epsg = crs.to_epsg() if crs is not None else None
    if epsg is not None:
        return f"urn:ogc:def:crs:EPSG::{epsg}"
    return str(crs) if crs is not None else "urn:ogc:def:crs:EPSG::0"


def write_citygml_file(filename, building_id, solids, srs_name):
    root = ET.Element(
        qn("core", "CityModel"),
        {
            qn("xsi", "schemaLocation"): (
                "http://www.opengis.net/citygml/2.0 "
                "http://schemas.opengis.net/citygml/2.0/cityGMLBase.xsd "
                "http://www.opengis.net/citygml/building/2.0 "
                "http://schemas.opengis.net/citygml/building/2.0/building.xsd"
            )
        },
    )

    lower, upper = compute_envelope(solids)
    bounded_by = ET.SubElement(root, qn("gml", "boundedBy"))
    envelope = ET.SubElement(bounded_by, qn("gml", "Envelope"), {"srsName": srs_name, "srsDimension": "3"})
    ET.SubElement(envelope, qn("gml", "lowerCorner")).text = " ".join(fmt(v) for v in lower)
    ET.SubElement(envelope, qn("gml", "upperCorner")).text = " ".join(fmt(v) for v in upper)

    city_object_member = ET.SubElement(root, qn("core", "cityObjectMember"))
    building_gml_id = f"building_{building_id}"
    building_el = ET.SubElement(
        city_object_member,
        qn("bldg", "Building"),
        {qn("gml", "id"): building_gml_id},
    )
    ET.SubElement(building_el, qn("gml", "name")).text = building_gml_id

    polygon_ids = []
    for solid_idx, solid in enumerate(solids, start=1):
        base = f"{building_gml_id}_s{solid_idx}"

        # The bottom ring is reversed so its normal points downward.
        bottom_poly_id = add_surface(
            building_el,
            "GroundSurface",
            f"{base}_ground",
            list(reversed(solid["bottom"])),
        )
        roof_poly_id = add_surface(building_el, "RoofSurface", f"{base}_roof", solid["roof"])
        polygon_ids.extend([bottom_poly_id, roof_poly_id])

        for wall_idx, wall in enumerate(solid["walls"], start=1):
            wall_poly_id = add_surface(building_el, "WallSurface", f"{base}_wall_{wall_idx}", wall)
            polygon_ids.append(wall_poly_id)

    add_solid(building_el, building_gml_id, polygon_ids)

    rough = ET.tostring(root, encoding="utf-8")
    pretty = xml.dom.minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    with open(filename, "wb") as f:
        f.write(pretty)


def write_citygml_collection(filename, buildings, srs_name):
    all_solids = []
    for _, solids in buildings:
        all_solids.extend(solids)

    root = ET.Element(
        qn("core", "CityModel"),
        {
            qn("xsi", "schemaLocation"): (
                "http://www.opengis.net/citygml/2.0 "
                "http://schemas.opengis.net/citygml/2.0/cityGMLBase.xsd "
                "http://www.opengis.net/citygml/building/2.0 "
                "http://schemas.opengis.net/citygml/building/2.0/building.xsd"
            )
        },
    )

    lower, upper = compute_envelope(all_solids)
    bounded_by = ET.SubElement(root, qn("gml", "boundedBy"))
    envelope = ET.SubElement(bounded_by, qn("gml", "Envelope"), {"srsName": srs_name, "srsDimension": "3"})
    ET.SubElement(envelope, qn("gml", "lowerCorner")).text = " ".join(fmt(v) for v in lower)
    ET.SubElement(envelope, qn("gml", "upperCorner")).text = " ".join(fmt(v) for v in upper)

    for building_id, solids in buildings:
        city_object_member = ET.SubElement(root, qn("core", "cityObjectMember"))
        building_gml_id = f"building_{building_id}"
        building_el = ET.SubElement(
            city_object_member,
            qn("bldg", "Building"),
            {qn("gml", "id"): building_gml_id},
        )
        ET.SubElement(building_el, qn("gml", "name")).text = building_gml_id

        polygon_ids = []
        for solid_idx, solid in enumerate(solids, start=1):
            base = f"{building_gml_id}_s{solid_idx}"
            bottom_poly_id = add_surface(
                building_el,
                "GroundSurface",
                f"{base}_ground",
                list(reversed(solid["bottom"])),
            )
            roof_poly_id = add_surface(building_el, "RoofSurface", f"{base}_roof", solid["roof"])
            polygon_ids.extend([bottom_poly_id, roof_poly_id])

            for wall_idx, wall in enumerate(solid["walls"], start=1):
                wall_poly_id = add_surface(building_el, "WallSurface", f"{base}_wall_{wall_idx}", wall)
                polygon_ids.append(wall_poly_id)

        add_solid(building_el, building_gml_id, polygon_ids)

    rough = ET.tostring(root, encoding="utf-8")
    pretty = xml.dom.minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    with open(filename, "wb") as f:
        f.write(pretty)


# ---------------- MAIN ----------------
def main():
    json_files = sorted(f for f in os.listdir(json_folder) if f.endswith(".json"))
    exported_buildings = []
    collection_srs_name = None

    for filename in json_files:
        parts = os.path.splitext(filename)[0].split("_")
        if len(parts) < 2:
            print(f"Skip {filename}: unexpected name.")
            continue

        building_id = parts[1]
        json_path = os.path.join(json_folder, filename)
        tif_path = os.path.join(tif_folder, f"building_{building_id}.tif")
        las_path = os.path.join(las_folder, f"{building_id}_point_cloud.las")

        if not (os.path.exists(tif_path) and os.path.exists(las_path)):
            print(f"Skipping {building_id}: Missing TIFF or LAS file.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            boxes_data = json.load(f)
        if not boxes_data:
            print(f"Skipping {building_id}: Empty JSON file.")
            continue

        with rasterio.open(tif_path) as tif:
            transform = tif.transform
            srs_name = crs_to_srs_name(tif.crs)
            if collection_srs_name is None:
                collection_srs_name = srs_name

        geo_boxes = []
        for box in boxes_data:
            geo_box = []
            for p in box:
                x, y = transform * (p["x"], p["y"])
                geo_box.append({"x": x, "y": y})
            geo_boxes.append(geo_box)

        las = laspy.read(las_path)
        pts_xyz = np.column_stack([las.x, las.y, las.z]).astype(float)

        solids = build_merged_polygons(geo_boxes, pts_xyz)
        if not solids:
            print(f"Skipping {building_id}: no valid solids.")
            continue

        out_gml = os.path.join(output_folder, f"building_{building_id}_citygml_lod2.gml")
        write_citygml_file(out_gml, building_id, solids, srs_name)
        print(f"CityGML file created: {out_gml}")
        exported_buildings.append((building_id, solids))

    if exported_buildings:
        out_collection = os.path.join(output_folder, "all_buildings_citygml_lod2.gml")
        write_citygml_collection(out_collection, exported_buildings, collection_srs_name)
        print(f"Combined CityGML file created: {out_collection}")


if __name__ == "__main__":
    main()
