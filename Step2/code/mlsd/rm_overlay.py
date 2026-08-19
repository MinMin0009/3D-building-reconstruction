import torch
import os
import sys
import cv2
from osgeo import gdal
import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
import random

from models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny
from models.mbv2_mlsd_large import MobileV2_MLSD_Large

from utils import pred_lines

def rotate_point(center, point, angle):
    """Rotate a point around a center with given angle."""
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    x_shifted = point[0] - center[0]
    y_shifted = point[1] - center[1]
    x_rotated = x_shifted * cos_angle - y_shifted * sin_angle
    y_rotated = x_shifted * sin_angle + y_shifted * cos_angle
    return (int(center[0] + x_rotated), int(center[1] + y_rotated))

def extend_line(pt1, pt2, length=1000):
    """Extend a line segment by a given length in both directions."""
    line_vector = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])
    line_length = np.linalg.norm(line_vector)
    if line_length == 0:
        return pt1, pt2
    line_unit_vector = line_vector / line_length
    pt1_extended = (pt1[0] - line_unit_vector[0] * length, pt1[1] - line_unit_vector[1] * length)
    pt2_extended = (pt2[0] + line_unit_vector[0] * length, pt2[1] + line_unit_vector[1] * length)
    return pt1_extended, pt2_extended

def main():
    current_dir = os.path.dirname(__file__)
    if current_dir == "":
        current_dir = "./"

    model_path = current_dir + '/models/mlsd_large_512_fp32.pth'
    model = MobileV2_MLSD_Large().cuda().eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    img_fn = current_dir + '/data/building_building_72.tif'

    # Read the image using GDAL to preserve metadata
    dataset = gdal.Open(img_fn, gdal.GA_ReadOnly)
    metadata = dataset.GetMetadata()
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    img = cv2.imread(img_fn)
    original_size = img.shape[:2]  # Store the original size

    # Resize to 512x512 for model input
    resized_img = cv2.resize(img, (512, 512))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    lines = pred_lines(resized_img, model, [512, 512], 0.2, 10)

    # Scale lines back to original image size
    scale_x = original_size[1] / 512.0
    scale_y = original_size[0] / 512.0
    scaled_lines = [(int(l[0] * scale_x), int(l[1] * scale_y), int(l[2] * scale_x), int(l[3] * scale_y)) for l in lines]

    line_lengths = []
    line_angles = []
    line_points = []

    # Collect line data
    for l in scaled_lines:
        pt1 = (int(l[0]), int(l[1]))
        pt2 = (int(l[2]), int(l[3]))
        
        # Calculate the length of the line
        length = np.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2)
        line_lengths.append(length)
        
        # Calculate the angle of the line in degrees
        angle = np.degrees(np.arctan2(l[3] - l[1], l[2] - l[0])) % 180
        line_angles.append(angle)
        
        line_points.append((pt1, pt2))

    # Create a histogram of line lengths by angle
    angle_bins = np.arange(0, 180, 1)
    total_lengths = np.zeros_like(angle_bins, dtype=np.float64)

    for angle, length in zip(line_angles, line_lengths):
        bin_index = int(angle)
        total_lengths[bin_index] += length

    # Prepare data for Mean-Shift clustering
    X = np.array([[angle] for angle in angle_bins])
    y = total_lengths

    # Expand the dataset by replicating each point according to its weight
    weighted_X = np.repeat(X, y.astype(int), axis=0)

    # Apply Mean-Shift clustering
    bandwidth = 2  # Adjust this parameter as necessary
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    mean_shift.fit(weighted_X)

    mean_shift_labels = mean_shift.predict(X)
    unique_labels = np.unique(mean_shift_labels)

    # Find the angle with the highest value in each cluster
    peak_angles = {}
    valid_clusters = []
    for label in unique_labels:
        cluster_indices = np.where(mean_shift_labels == label)[0]
        if len(cluster_indices) > 0:
            peak_index = cluster_indices[np.argmax(total_lengths[cluster_indices])]
            peak_angle = angle_bins[peak_index]
            if total_lengths[peak_index] > 300:  # Threshold for filtering significant clusters
                peak_angles[label] = peak_angle
                valid_clusters.append(label)

    # Print peak angles for each valid cluster
    for label, peak_angle in peak_angles.items():
        print(f"Cluster {label + 1}: Peak Angle = {peak_angle} degrees")

    # Filter lines based on valid clusters
    filtered_lines = []
    for i in range(len(line_points)):
        angle = line_angles[i]
        bin_index = int(angle)
        cluster_label = mean_shift_labels[bin_index]
        if cluster_label in peak_angles:  # Ensure the cluster has a peak angle
            filtered_lines.append(line_points[i])

    # Expand lines and create shapely LineStrings
    shapely_lines = []
    for pt1, pt2 in filtered_lines:
        pt1_extended, pt2_extended = extend_line(pt1, pt2)
        shapely_lines.append(LineString([pt1_extended, pt2_extended]))

    # Find intersections and create polygons
    lines_union = unary_union(shapely_lines)
    polygons = list(polygonize(lines_union))

    # Draw polygons with random colors
    color_img = img.copy()
    for polygon in polygons:
        if not polygon.is_empty and isinstance(polygon, Polygon):
            exterior_coords = np.array(polygon.exterior.coords, dtype=np.int32)
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.fillPoly(color_img, [exterior_coords], random_color)

    # Save the output image using GDAL to retain metadata
    output_fn = current_dir + '/results/building_segmented_building(72).tif'
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_fn, img.shape[1], img.shape[0], 3, gdal.GDT_Byte)

    out_dataset.SetGeoTransform(geo_transform)
    out_dataset.SetProjection(projection)
    out_dataset.SetMetadata(metadata)

    # Write the image data to the output dataset
    for i in range(3):
        out_dataset.GetRasterBand(i + 1).WriteArray(color_img[:, :, i])

    # Close the datasets
    out_dataset.FlushCache()
    out_dataset = None
    dataset = None

    # Plot the results with valid clusters and filtered angles
    plt.figure(figsize=(12, 6))
    for label in valid_clusters:
        peak_angle = peak_angles[label]
        cluster_indices = np.where(mean_shift_labels == label)[0]
        filtered_indices = [idx for idx in cluster_indices if abs(angle_bins[idx] - peak_angle) <= 5]
        plt.bar(angle_bins[filtered_indices], total_lengths[filtered_indices], width=1, edgecolor='black', label=f'Cluster {label + 1}')

    plt.xlabel('Angle (degrees)')
    plt.ylabel('Total Length of Lines')
    plt.title('Total Length of Lines by Angle with Mean-Shift Clusters (Filtered)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
