import os
import numpy as np
import rasterio
import laspy
import pickle
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from shapely.geometry import Polygon
from shapely.affinity import scale

from Tree_Segmentation.config import load_config
import shutil
import time
start_time = time.time()

# Load configuration file
config_path = None    
if config_path is None:
    config = load_config(config_path)

# Input paths from YAML
tiff_path = config['paths']['tiff_path']
las_path = config['paths']['las_path']
pkl_path = os.path.join(config['paths']['result_path'], 'output_new_npy', 'cluster_building.pkl')  # Path to pkl file from another script
result_path = config['paths']['result_path']

# Automatically create output directories in result_path
output_npy_dir = os.path.join(result_path, 'npy stp2')
os.makedirs(output_npy_dir, exist_ok=True)
output_npy_path = os.path.join(output_npy_dir, 'point_cloud_grid_suseo.npy')

building_images_dir = os.path.join(result_path, 'Building Candidate Images')
cropped_images_folder = os.path.join(building_images_dir, 'building images')
bounding_box_folder = os.path.join(building_images_dir, 'bounding coordinates')
os.makedirs(cropped_images_folder, exist_ok=True)
os.makedirs(bounding_box_folder, exist_ok=True)

# Load TIFF file
with rasterio.open(tiff_path) as src:
    tiff_data = src.read()
    transform = src.transform
    width, height = src.width, src.height
    bounds, crs = src.bounds, src.crs

# Load LAS file and extract coordinates
las = laspy.read(las_path)
points = np.vstack((las.x, las.y, las.z)).transpose()

# Create an array to hold Z values
z_array = np.full((height, width), np.nan, dtype=np.float32)

# Convert geo-coordinates to pixel indices
def coord_to_pixel(x, y, transform):
    col, row = ~transform * (x, y)
    return int(col), int(row)

# Assign Z values to corresponding pixels
valid_points = 0
for point in points:
    x, y, z = point
    if bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top:
        col, row = coord_to_pixel(x, y, transform)
        if 0 <= col < width and 0 <= row < height:
            if np.isnan(z_array[row, col]):
                z_array[row, col] = z
            else:
                z_array[row, col] = (z_array[row, col] + z) / 2
            valid_points += 1

# Save Z values as npy file
np.save(output_npy_path, z_array)
print(f"Output npy saved to {output_npy_path}")




''' fixing (2025.08.01)
# Load clusters from pkl file
with open(pkl_path, 'rb') as file:
    clusters = pickle.load(file)
'''
#to
from joblib import load
clusters = load(pkl_path)






# Function to calculate expanded convex hull
def calculate_expanded_convex_hull(cluster, expansion_distance=1):
    if len(cluster) < 3:
        return cluster
    hull = ConvexHull(cluster)
    hull_points = cluster[hull.vertices]
    hull_polygon = Polygon(hull_points)
    expanded_polygon = scale(hull_polygon, xfact=1 + expansion_distance, yfact=1 + expansion_distance, origin='center')
    expanded_hull_points = np.array(expanded_polygon.exterior.coords)
    return expanded_hull_points

# Calculate expanded convex hull for each cluster and save bounding boxes
expanded_convex_hulls = {}
for label, cluster in clusters.items():
    expanded_hull = calculate_expanded_convex_hull(cluster, expansion_distance=0.1)  # Expansion distance in pixel units
    expanded_convex_hulls[label] = expanded_hull
    bounding_box_file = os.path.join(bounding_box_folder, f"building_{label}.txt")
    with open(bounding_box_file, 'w') as f:
        f.write(f"Building {label} Expanded Convex Hull: {expanded_hull.tolist()}\n")

# Crop images based on expanded convex hulls
def crop_and_save_images(tiff_data, expanded_convex_hulls, folder):
    os.makedirs(folder, exist_ok=True)
    for label, hull in expanded_convex_hulls.items():
        min_row, min_col = np.floor(np.min(hull, axis=0)).astype(int)
        max_row, max_col = np.ceil(np.max(hull, axis=0)).astype(int)
        
        # Ensure indices are within image bounds
        min_row = max(min_row, 0)
        min_col = max(min_col, 0)
        max_row = min(max_row, tiff_data.shape[1] - 1)
        max_col = min(max_col, tiff_data.shape[2] - 1)
        
        # Crop region of interest from the TIFF data
        cropped_image = tiff_data[:, min_row:max_row+1, min_col:max_col+1]
        
        # Create a mask for the expanded convex hull
        mask = np.zeros((cropped_image.shape[1], cropped_image.shape[2]), dtype=bool)
        hull[:, 0] -= min_row
        hull[:, 1] -= min_col
        rr, cc = polygon(hull[:, 0], hull[:, 1], mask.shape)
        mask[rr, cc] = True
        
        # Apply mask to the cropped image
        masked_cropped_image = np.zeros_like(cropped_image)
        for i in range(cropped_image.shape[0]):
            masked_cropped_image[i] = np.where(mask, cropped_image[i], 0)
        
        cropped_image_path = os.path.join(folder, f"building_{label}.tif")
        with rasterio.open(
            cropped_image_path,
            'w',
            driver='GTiff',
            height=masked_cropped_image.shape[1],
            width=masked_cropped_image.shape[2],
            count=masked_cropped_image.shape[0],
            dtype=masked_cropped_image.dtype,
            crs=crs,
            transform=rasterio.transform.from_bounds(
                bounds.left + min_col * transform.a,
                bounds.top + max_row * transform.e,
                bounds.left + max_col * transform.a,
                bounds.top + min_row * transform.e,
                masked_cropped_image.shape[2],
                masked_cropped_image.shape[1]
            )
        ) as dst:
            dst.write(masked_cropped_image)

# Crop and save images
crop_and_save_images(tiff_data, expanded_convex_hulls, cropped_images_folder)
print(f"Cropped images saved in folder {cropped_images_folder}")




las_folder = os.path.join(result_path, 'LAS')
removed_dir = os.path.join(las_folder, 'LAS_removed')
os.makedirs(removed_dir, exist_ok=True)

removed_labels = set()
las_files = [f for f in os.listdir(las_folder) if f.endswith('.las')]

for las_file in las_files:
    las_path_file = os.path.join(las_folder, las_file)
    try:
        las_data = laspy.read(las_path_file)
        num_points = len(las_data.points)
        if num_points < 30000:
            shutil.move(las_path_file, os.path.join(removed_dir, las_file))
            print(f"Moved {las_file} to LAS_removed/ (points: {num_points})")
            label = las_file.split('_')[0]  # e.g., '0_point_cloud.las' → '0'
            removed_labels.add(label)
    except Exception as e:
        print(f"Error reading {las_file}: {e}")

for label in removed_labels:
    tif_path = os.path.join(cropped_images_folder, f"building_{label}.tif")
    if os.path.exists(tif_path):

        removed_tif_dir = os.path.join(cropped_images_folder, 'TIF_removed')
        os.makedirs(removed_tif_dir, exist_ok=True)

        for label in removed_labels:
            tif_path = os.path.join(cropped_images_folder, f"building_{label}.tif")
            if os.path.exists(tif_path):
                shutil.move(tif_path, os.path.join(removed_tif_dir, f"building_{label}.tif"))
                print(f"Moved cropped image to TIF_removed/: building_{label}.tif")

        print(f"Removed cropped image: building_{label}.tif")
        
        
print(f"Processing is completed. Processing time: {time.time() - start_time:.2f} seconds.") 