# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import laspy
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from keras.models import load_model
from PIL import Image

from Tree_Segmentation.config import load_config
from data import normalize_images
from model import dice_coefficient

Image.MAX_IMAGE_PIXELS = None


def split_image(tiff_path, output_dir, tile_size, width, height):
    os.makedirs(output_dir, exist_ok=True)
    with Image.open(tiff_path) as img:
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
                tile = img.crop(box)
                tile.save(os.path.join(output_dir, f"tile_{x}_{y}.png"))


def predict_mask(image_bgr, model):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = normalize_images([image])[0]
    image_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    image_resized = np.expand_dims(image_resized, axis=0)
    predicted = model.predict(image_resized)
    predicted = (predicted > 0.5).astype(np.uint8)
    predicted = np.squeeze(predicted)
    predicted = cv2.resize(predicted, (image_bgr.shape[1], image_bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    return predicted  # 0/1


# --------
def build_mask_geotiff_from_tiles(tiff_path, tile_mask_dir, out_mask_tif, threshold=127):
    with rasterio.open(tiff_path) as src:
        profile = src.profile.copy()
        profile.update(driver='GTiff', dtype='uint8', count=1, nodata=0, compress='LZW')
        width, height = src.width, src.height

        files = sorted([f for f in os.listdir(tile_mask_dir)
                        if f.lower().endswith('.png') or f.lower().endswith('.jpg')])
        if not files:
            print("[WARN] Không tìm thấy tile mask trong:", tile_mask_dir)

        with rasterio.open(out_mask_tif, 'w', **profile) as dst:
            for fname in files:
                name = os.path.splitext(fname)[0]  # tile_{x}_{y}
                parts = name.split('_')
                if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
                    continue

                x_off = int(parts[1])
                y_off = int(parts[2])

                p = os.path.join(tile_mask_dir, fname)
                tile = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if tile is None:
                    continue

                tile_bin01 = (tile > threshold).astype(np.uint8)
                tile_bin255 = tile_bin01 * 255

                h, w = tile_bin255.shape
                if x_off >= width or y_off >= height:
                    continue
                w = min(w, width - x_off)
                h = min(h, height - y_off)
                if w <= 0 or h <= 0:
                    continue

                dst.write(tile_bin255[:h, :w][np.newaxis, :, :],
                          window=Window(col_off=x_off, row_off=y_off, width=w, height=h))


def las_chunk_iterator(las_path, points_per_chunk=2_000_000):
    try:
        with laspy.open(las_path) as reader:
            for chunk in reader.chunk_iterator(points_per_chunk):
                xs = chunk.x
                ys = chunk.y
                zs = chunk.z
                return_obj = (chunk, xs, ys, zs)
                yield return_obj
    except Exception:
        las = laspy.read(las_path)
        n = len(las.x)
        for start in range(0, n, points_per_chunk):
            stop = min(n, start + points_per_chunk)
            sl = slice(start, stop)
            yield (None, las.x[sl], las.y[sl], las.z[sl])


def sample_mask_fast(xs, ys, ds):
    rows, cols = rowcol(ds.transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    H, W = ds.height, ds.width
    inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    mask_val = np.zeros(xs.shape[0], dtype=np.uint8)
    if not np.any(inside):
        return mask_val

    r_in = rows[inside]
    c_in = cols[inside]
    r0, r1 = int(r_in.min()), int(r_in.max())
    c0, c1 = int(c_in.min()), int(c_in.max())
    win_h = r1 - r0 + 1
    win_w = c1 - c0 + 1
    if win_h <= 0 or win_w <= 0:
        return mask_val

    window = Window(col_off=c0, row_off=r0, width=win_w, height=win_h)
    sub = ds.read(1, window=window, boundless=True, fill_value=0)
    mask_val[inside] = sub[r_in - r0, c_in - c0]
    return mask_val  

def pass1_accumulate_cell_stats(las_path, mask_tif_path, cell_size=0.75, bin_size=0.15):
    stats = {}
    with rasterio.open(mask_tif_path) as ds:
        bounds = ds.bounds
        x0 = bounds.left
        y0 = bounds.bottom

        for chunk_obj, xs, ys, zs in las_chunk_iterator(las_path):
            m = sample_mask_fast(xs, ys, ds)
            sel = (m > 0)  
            if not np.any(sel):
                continue

            xs1 = np.asarray(xs)[sel]
            ys1 = np.asarray(ys)[sel]
            zs1 = np.asarray(zs)[sel]  

            ci = np.floor((xs1 - x0) / cell_size).astype(np.int64)
            cj = np.floor((ys1 - y0) / cell_size).astype(np.int64)

            cell_pairs = np.stack([ci, cj], axis=1)
            cell_view = cell_pairs.view([('i', np.int64), ('j', np.int64)]).ravel()
            uniq_cells, inv = np.unique(cell_view, return_inverse=True)

            for k, cell_rec in enumerate(uniq_cells):
                idx_k = (inv == k)
                z_k = zs1[idx_k]
                z_k_len = int(len(z_k))
                if z_k_len == 0:
                    continue

                z_k = np.asarray(z_k)
                zmin = float(np.min(z_k))
                zmax = float(np.max(z_k))

                bin_idx = np.floor(z_k / bin_size).astype(np.int64)
                u_bins, counts = np.unique(bin_idx, return_counts=True)

                i_val = int(cell_rec['i'])
                j_val = int(cell_rec['j'])
                key = (i_val, j_val)

                cell = stats.get(key)
                if cell is None:
                    cell = {'n': 0, 'zmin': zmin, 'zmax': zmax, 'hist': {}}
                    stats[key] = cell
                cell['n'] += z_k_len
                cell['zmin'] = min(cell['zmin'], zmin)
                cell['zmax'] = max(cell['zmax'], zmax)
                hist = cell['hist']
                for b, c in zip(u_bins, counts):
                    hist[int(b)] = hist.get(int(b), 0) + int(c)

    return stats


def finalize_cell_decisions(stats, bin_size=0.15):
    info = {}
    for key, st in stats.items():
        n = st['n']
        z_range = float(st['zmax'] - st['zmin'])
        if st['hist']:
            mode_bin, mode_count = max(st['hist'].items(), key=lambda kv: kv[1])
            dominance = float(mode_count) / max(1, n)
            z_mode = (mode_bin + 0.5) * bin_size
        else:
            dominance = 0.0
            z_mode = float('nan')
        info[key] = {'n': n, 'z_range': z_range, 'dominance': dominance, 'z_mode': z_mode}
    return info


def pass2_filter_and_write(las_path, out_las_path, mask_tif_path, cell_info,
                           cell_size=0.75, z_range_thin=0.30, delta_mode=0.40,
                           dominance_thr=0.45, min_points_cell=30):
    with rasterio.open(mask_tif_path) as ds:
        bounds = ds.bounds
        x0 = bounds.left
        y0 = bounds.bottom

        with laspy.open(las_path) as reader, laspy.open(out_las_path, mode="w", header=reader.header) as writer:
            for chunk in reader.chunk_iterator(2_000_000):
                xs = chunk.x
                ys = chunk.y
                zs = chunk.z

                m = sample_mask_fast(xs, ys, ds)
                keep = np.ones(len(xs), dtype=bool)

                inside_mask = (m > 0)  
                if np.any(inside_mask):
                    xs1 = np.asarray(xs)[inside_mask]
                    ys1 = np.asarray(ys)[inside_mask]
                    zs1 = np.asarray(zs)[inside_mask]

                    ci = np.floor((xs1 - x0) / cell_size).astype(np.int64)
                    cj = np.floor((ys1 - y0) / cell_size).astype(np.int64)

                    sub_keep = np.empty(len(xs1), dtype=bool)
                    for idx in range(len(xs1)):
                        key = (int(ci[idx]), int(cj[idx]))
                        inf = cell_info.get(key)
                        if inf is None:
                            sub_keep[idx] = True
                            continue

                        n = inf['n']
                        z_range = inf['z_range']
                        dominance = inf['dominance']
                        z_mode = inf['z_mode']
                        z_here = float(zs1[idx])

                        if (n < min_points_cell) or (dominance < dominance_thr):
                            sub_keep[idx] = (z_range <= z_range_thin)
                        else:
                            cond_thin = (z_range <= z_range_thin)
                            cond_mode = (abs(z_here - z_mode) <= delta_mode)
                            sub_keep[idx] = (cond_thin or cond_mode)

                    keep[inside_mask] = sub_keep

                if not np.any(keep):
                    continue
                writer.write_points(chunk[keep])


def main(config_path=None):
    config = load_config(config_path)
    print("[DEBUG] Config:", config)

    tiff_path = config['paths']['tiff_path']
    las_file = config['paths']['las_path']
    model_path = config['paths']['model_path']

    tiff_name = os.path.splitext(os.path.basename(tiff_path))[0]
    base_dir = os.path.dirname(tiff_path)

    tiles_dir = os.path.join(base_dir, "Tiff_to_Images")
    pred_dir = os.path.join(base_dir, f"{tiff_name}_Predicted_Masks")
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    tile_size = config.get('model_params', {}).get('tile_size', 512)

    filt = config.get('filter_params', {})
    cell_size = float(filt.get('cell_size', 0.75))
    bin_size = float(filt.get('bin_size', 0.15))
    dominance_thr = float(filt.get('dominance', 0.45))
    z_range_thin = float(filt.get('z_range_thin', 0.30))
    delta_mode = float(filt.get('delta_mode', 0.40))
    min_points_cell = int(filt.get('min_points_cell', 30))

    with Image.open(tiff_path) as img:
        width, height = img.size

    split_image(tiff_path, tiles_dir, tile_size, width, height)

    model = load_model(model_path, custom_objects={'dice_coefficient': dice_coefficient})
    tile_files = sorted([os.path.join(tiles_dir, f)
                         for f in os.listdir(tiles_dir)
                         if f.lower().endswith('.png') or f.lower().endswith('.jpg')])

    for p in tile_files:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        m = predict_mask(im, model)  
        out = os.path.join(pred_dir, os.path.basename(p))
        cv2.imwrite(out, (m.astype(np.uint8) * 255))  

    mask_tif = os.path.join(base_dir, f"{tiff_name}_mask.tif")
    build_mask_geotiff_from_tiles(tiff_path, pred_dir, mask_tif, threshold=127)

    # Pass 1 & 2
    stats = pass1_accumulate_cell_stats(las_file, mask_tif, cell_size=cell_size, bin_size=bin_size)
    cell_info = finalize_cell_decisions(stats, bin_size=bin_size)

    out_las = os.path.join(os.path.dirname(las_file), f"{tiff_name}_filtered.las")
    pass2_filter_and_write(
        las_file, out_las, mask_tif, cell_info,
        cell_size=cell_size,
        z_range_thin=z_range_thin,
        delta_mode=delta_mode,
        dominance_thr=dominance_thr,
        min_points_cell=min_points_cell
    )
    print("[Done] LAS saved into:", out_las)


if __name__ == "__main__":
    main()
