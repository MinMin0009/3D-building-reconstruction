import torch
import os
import cv2
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.ops import unary_union
import random
from sklearn.cluster import MeanShift
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
    return (center[0] + x_rotated, center[1] + y_rotated)

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

def angle_difference(angle1, angle2):
    """Calculate the absolute difference between two angles, considering the circular nature of angles."""
    diff = abs(angle1 - angle2)
    return min(diff, 180 - diff)

def rotate_line_to_angle(pt1, pt2, target_angle):
    """Rotate a line to a target angle."""
    mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
    current_angle = np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])) % 180
    angle_diff = target_angle - current_angle
    pt1_rotated = rotate_point(mid_point, pt1, angle_diff)
    pt2_rotated = rotate_point(mid_point, pt2, angle_diff)
    return pt1_rotated, pt2_rotated

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
    bandwidth = 4  # Adjust this parameter as necessary
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
            if total_lengths[peak_index] > 200:  # Threshold for filtering significant clusters
                peak_angles[label] = peak_angle
                valid_clusters.append(label)

    # Filter clusters based on the distance between peak angles and their total lengths
    to_remove = set()
    for i, label1 in enumerate(valid_clusters):
        for label2 in valid_clusters[i+1:]:
            if angle_difference(peak_angles[label1], peak_angles[label2]) < 25:
                if total_lengths[peak_angles[label1]] < total_lengths[peak_angles[label2]]:
                    to_remove.add(label1)
                else:
                    to_remove.add(label2)

    valid_clusters = [label for label in valid_clusters if label not in to_remove]

    # Print peak angles for each valid cluster
    for label in valid_clusters:
        print(f"Cluster {label + 1}: Peak Angle = {peak_angles[label]} degrees")

    # Filter lines based on valid clusters
    filtered_lines = []
    for i in range(len(line_points)):
        angle = line_angles[i]
        bin_index = int(angle)
        cluster_label = mean_shift_labels[bin_index]
        if cluster_label in valid_clusters:  # Ensure the cluster is valid
            filtered_lines.append((line_points[i], line_lengths[i], cluster_label))

    # Sort and rank lines within each cluster by length
    cluster_lines = {label: [] for label in valid_clusters}
    for line, length, label in filtered_lines:
        cluster_lines[label].append((line, length))

    for label in cluster_lines:
        cluster_lines[label].sort(key=lambda x: x[1], reverse=True)

    # Rotate lines based on valid clusters and save a TIF file for each cluster
    for label in cluster_lines:
        line_img = img.copy()
        for idx, (line, length) in enumerate(cluster_lines[label]):
            pt1, pt2 = line
            mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            current_angle = np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])) % 180
            target_angle = peak_angles[label]
            angle_diff = target_angle - current_angle
            pt1_rotated = rotate_point(mid_point, pt1, angle_diff)
            pt2_rotated = rotate_point(mid_point, pt2, angle_diff)
            
            # Draw the rotated line on the image and label it with index and length
            cv2.line(line_img, tuple(map(int, pt1_rotated)), tuple(map(int, pt2_rotated)), (0, 255, 0), 2)
            label_text = f"{idx + 1}: {int(length)}"
            text_position = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.putText(line_img, label_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save the output image using GDAL to retain metadata
        output_fn = current_dir + f'/results/building_cluster_{label + 1}_lines_overlayed.tif'
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(output_fn, img.shape[1], img.shape[0], 3, gdal.GDT_Byte)

        out_dataset.SetGeoTransform(geo_transform)
        out_dataset.SetProjection(projection)
        out_dataset.SetMetadata(metadata)

        # Write the image data to the output dataset
        for i in range(3):
            out_dataset.GetRasterBand(i + 1).WriteArray(line_img[:, :, i])

        # Close the datasets
        out_dataset.FlushCache()
        out_dataset = None

        # Plot the ranked line lengths for this cluster
        plt.figure(figsize=(10, 6))
        lengths = [length for _, length in cluster_lines[label]]
        indices = range(1, len(lengths) + 1)
        plt.bar(indices, lengths, color='b', alpha=0.7)
        plt.xlabel('Line Index')
        plt.ylabel('Length of Line')
        plt.title(f'Cluster {label + 1}: Line Lengths Ranked')
        plt.grid(True)
        plt.show()

    dataset = None

if __name__ == '__main__':
    main()
