
import json
import numpy as np
import laspy
from PIL import Image, ImageDraw
import os
from osgeo import gdal
import alphashape
from shapely.geometry import Polygon, Point

# Kích hoạt ngoại lệ trong GDAL
gdal.UseExceptions()

# Đường dẫn tới các file
json_file_path = "F:/Project_2024_MJU/step2/Step2_1(mlsd_cluster)/code/mlsd/results/building_25.json"
tif_file_path = "F:/Project_2024_MJU/step2/Step2_1(mlsd_cluster)/code/mlsd/data/building_25.tif"
las_file_path = "D:/project/suseou_data2/LAS/output/LAS/25_point_cloud.las"  #"D:/project/suseou_data2/LAS/output/LAS/13_point_cloud.las"  #"E:/project/2024.06.12/pointcloud/LAS_Files/building_23_point_cloud.las"

# Tham số lambda
lambda_factor = 0.9  # Bạn có thể thay đổi giá trị lambda tại đây

# Tải ảnh TIFF
image = Image.open(tif_file_path)

# Load the TIFF file using GDAL to access geotransform
tif_dataset = gdal.Open(tif_file_path)
geotransform = tif_dataset.GetGeoTransform()

# Hàm chuyển đổi từ tọa độ địa lý sang tọa độ pixel
def geo_to_pixel(x, y, geotransform):
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    
    pixel_x = int((x - origin_x) / pixel_width)
    pixel_y = int((y - origin_y) / pixel_height)
    
    return pixel_x, pixel_y

# Tải dữ liệu từ JSON
with open(json_file_path, 'r') as f:
    boxes_data = json.load(f)

# Tải dữ liệu point cloud
las = laspy.read(las_file_path)
points_geo = np.vstack([las.x, las.y]).transpose()

# Chuyển đổi tất cả các points từ geo-coordinates sang pixel coordinates
points_pixel = np.array([geo_to_pixel(x, y, geotransform) for x, y in points_geo])

# Hàm đếm số lượng points trong polygon
def points_in_polygon(polygon, points):
    from matplotlib.path import Path
    path = Path(polygon)
    return np.sum(path.contains_points(points))

# Hàm tính toán model complexity với lambda
def calculate_model_complexity(point_counts, a=0.001, lambda_factor=1.0):
    if len(point_counts) == 0:
        return np.nan
    return lambda_factor * np.mean([1 / (count + a) for count in point_counts])

# Hàm tính toán point coverage cho từng box
def calculate_point_coverage(box, points_pixel, alpha=0.5, coverage_threshold=0.3):
    """
    Tính toán tỷ lệ phủ điểm của mỗi box dựa trên alpha shape.
    Nếu tỷ lệ nhỏ hơn ngưỡng threshold thì loại bỏ box.
    """
    # get coordinate of boxes and calculate boxés area
    coordinates = [(point['x'], point['y']) for point in box]
    box_polygon = Polygon(coordinates)
    box_area = box_polygon.area
    
    # check point cloud inside box
    points_in_box = [p for p in points_pixel if box_polygon.contains(Point(p))]
    # If no point, point coverage = 0
    if len(points_in_box) == 0:
        return 0
    
    # calculate alpha shape for point in box
    alpha_shape_polygon = alphashape.alphashape(points_in_box, alpha)
    alpha_shape_area = alpha_shape_polygon.area
    
    # calculate point coverage following equation
    point_coverage = alpha_shape_area / box_area
    return point_coverage

# Tính toán số lượng points trong mỗi box
point_counts = []
for box in boxes_data:
    coordinates = [(point['x'], point['y']) for point in box]
    count = points_in_polygon(coordinates, points_pixel)
    point_counts.append(count)

# Xử lý trường hợp không có điểm nào trong bất kỳ box nào
if len(point_counts) == 0 or all(count == 0 for count in point_counts):
    print("There is no point in boxes, or all boxes are empty")
else:
    # Tính toán model complexity ban đầu với lambda
    initial_complexity = calculate_model_complexity(point_counts, lambda_factor=lambda_factor)
    print(f"Initial boxes value: {initial_complexity}")

    # Quá trình tối ưu hóa từng box
    filtered_boxes = boxes_data.copy()
    filtered_point_counts = point_counts.copy()
    removal_done = False
    threshold = 0.0008  # change threshold to stop removing

    while not removal_done:
        removal_done = True
        for i in range(len(filtered_boxes)):
            # Giả sử loại bỏ box thứ i
            temp_boxes = filtered_boxes[:i] + filtered_boxes[i+1:]
            temp_counts = filtered_point_counts[:i] + filtered_point_counts[i+1:]
            
            new_complexity = calculate_model_complexity(temp_counts, lambda_factor=lambda_factor)
            complexity_change = initial_complexity - new_complexity
            
            # Kiểm tra sự thay đổi của model complexity so với threshold
            if complexity_change > threshold:
                # Loại bỏ box này nếu sự thay đổi đáng kể
                filtered_boxes.pop(i)
                filtered_point_counts.pop(i)
                initial_complexity = new_complexity
                removal_done = False
                print(f"Remove box number {i}, Current boxes value: {initial_complexity}")
                break  # Restart loop after removal

    # threshold point coverage
    coverage_threshold = 0.5
    alpha_value = 0.2  # alpha value dèfault value

    # remove boxes base on coverage value
    final_filtered_boxes = []
    for i, box in enumerate(filtered_boxes):
        coverage = calculate_point_coverage(box, points_pixel, alpha=alpha_value, coverage_threshold=coverage_threshold)
        if coverage >= coverage_threshold:
            final_filtered_boxes.append(box)
        else:
            print(f"Remove box {i} with point coverage: {coverage:.2f}")

    # Vẽ các boxes còn lại lên ảnh
    draw = ImageDraw.Draw(image)
    for box in final_filtered_boxes:
        coordinates = [(point['x'], point['y']) for point in box]
        draw.polygon(coordinates, outline="red", width=3)

    # Lưu ảnh TIFF sau khi lọc
    output_tif_path = os.path.join(os.path.dirname(json_file_path), "building25_boxes_visualization.tif")
    image.save(output_tif_path)
    print(f"Filtered image saved to: {output_tif_path}")
    
    # Lưu các boxes sau khi lọc vào file JSON mới
    output_json_path = os.path.join(os.path.dirname(json_file_path), "building25_boxes_data.json")
    with open(output_json_path, 'w') as f:
        json.dump(final_filtered_boxes, f, indent=4)
    print(f"Filtered boxes saved to: {output_json_path}")
