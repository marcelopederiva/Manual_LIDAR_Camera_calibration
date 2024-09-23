import numpy as np
import open3d as o3d
import cv2
import tkinter as tk
from tkinter import scrolledtext

# ===========================
# 1. Initial Configuration
# ===========================

# --- Camera Intrinsics ---
camera_matrix = np.array([
    [2072.5190899420822, 0., 939.74694935751722],
    [0., 2228.2457986304757, 531.09465306781908],
    [0., 0., 1.]
])

# --- Distortion Coefficients ---
dist_coeffs = np.array([
    -4.5918644909485867e-01, 8.0596999894475399e-02,
    8.0714770089564975e-03, -4.0825911977209186e-03,
    -1.0014134711781530e-01
])

# --- Thresholds for Filtering LIDAR Points ---
lidar_thresholds = {
    'x_min': -15,
    'x_max': 15,
    'y_min': 2,
    'y_max': 15,
    'z_min': None,
    'z_max': None
}

# --- File Number ---
lidar_file = "data/lidar/lidar_hishort_1.pcd"
image_file = "data/image/image_hishort_1.jpg"

# ===========================
# 2. Matrix Modification Functions
# ===========================

def update_transformation_matrix():
    global H_lidar_to_cam
    # Display updated matrix and update visualization
    info_text.config(state='normal')
    info_text.delete('1.0', tk.END)
    info_content = f"Updated LIDAR to Camera Transformation Matrix:\n{H_lidar_to_cam}\n"
    info_text.insert(tk.END, info_content)
    info_text.config(state='disabled')
    update_3d_view()
    update_and_show_image()

def modify_H_value(row, col, step_entry, is_positive=True):
    global H_lidar_to_cam
    try:
        step = float(step_entry.get())
    except ValueError:
        step = 0.0  # Default to 0 if invalid input
    if not is_positive:
        step = -step  # Subtract if the button is for negative
    H_lidar_to_cam[row, col] += step
    update_transformation_matrix()

# ===========================
# 3. Processing Functions
# ===========================

def transform_lidar_points():
    cam_homogeneous_points = H_lidar_to_cam @ lidar_points_homogeneous.T
    cam_homogeneous_points = cam_homogeneous_points.T
    cam_points = cam_homogeneous_points[:, :3]
    cam_points_camref = np.copy(cam_points)
    cam_points_camref[:, 0] = cam_points[:, 0]  # x remains the same
    cam_points_camref[:, 1] = -cam_points[:, 2]  # y is the inverse of Z
    cam_points_camref[:, 2] = cam_points[:, 1]
    return cam_points_camref

def project_points(lidar_points_cam_coords):
    projected_points = np.dot(lidar_points_cam_coords, camera_matrix.T)
    projected_points /= projected_points[:, 2:3]  # Normalize by the z value
    return projected_points

def overlay_points(image_to_overlay, projected_points, distances):
    min_dist, max_dist = distances.min(), distances.max()
    colors = cv2.applyColorMap(((distances - min_dist) / (max_dist - min_dist) * 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    for point, color in zip(projected_points, colors):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image_to_overlay.shape[1] and 0 <= y < image_to_overlay.shape[0]:
            color = (int(color[0][0]), int(color[0][1]), int(color[0][2]))
            cv2.circle(image_to_overlay, (x, y), radius=3, color=color, thickness=-1)

def update_and_show_image():
    lidar_points_cam_coords = transform_lidar_points()
    projected_points = project_points(lidar_points_cam_coords)

    valid_points = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < image_width) & \
                   (projected_points[:, 1] >= 0) & (projected_points[:, 1] < image_height)
    projected_points = projected_points[valid_points]
    if projected_points.size == 0:
        print("No valid points after filtering by image bounds.")
        return
    distances = np.linalg.norm(lidar_points_cam_coords, axis=1)[valid_points]
    overlay_image = image.copy()
    overlay_points(overlay_image, projected_points, distances)
    resized_image = cv2.resize(overlay_image, (0, 0), fx=0.7, fy=0.7)
    cv2.imshow("LIDAR points on image", resized_image)
    cv2.waitKey(1)

def update_3d_view():
    global vis, lidar_points
    vis.clear_geometries()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

def newimg(image):
    height_nusc, width_nusc = 900, 1600
    horizon_line_nusc = int(height_nusc * 0.5)
    horizon_line_image4 = int(image.shape[0] * 0.35)
    crop_height = image.shape[0] - 400
    cropped_image4 = image[:crop_height, :]
    vertical_translation = horizon_line_image4 - horizon_line_nusc
    M = np.float32([[1, 0, 0], [0, 1, -vertical_translation]])
    shifted_cropped_image4 = cv2.warpAffine(cropped_image4, M, (cropped_image4.shape[1], cropped_image4.shape[0]))
    resized_cropped_image = cv2.resize(shifted_cropped_image4, (width_nusc, height_nusc), interpolation=cv2.INTER_LINEAR)
    return resized_cropped_image


# ===========================
# 4. GUI Configuration
# ===========================

# Creating the GUI first to ensure 'info_text' is defined before 'update_transformation_matrix' is called
root = tk.Tk()
root.title("Transformation Controls")

# --- Frame for transformation matrix ---
transformation_frame = tk.LabelFrame(root, text="Transformation Matrix H")
transformation_frame.pack(padx=10, pady=5, fill="both", expand="yes")

# --- Frame for transformation information ---
info_frame = tk.LabelFrame(root, text="Transformation Information")
info_frame.pack(padx=10, pady=5, fill="both", expand="yes")

# --- ScrolledText Widget to display information ---
info_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, width=50, height=10, state='disabled')
info_text.pack(padx=5, pady=5, fill="both", expand=True)

# --- Exit button ---
exit_button = tk.Button(root, text="Exit", command=root.quit)
exit_button.pack(pady=10)

# Creating buttons for modifying the H_lidar_to_cam matrix and setting step size
step_entries = {}
for i in range(4):
    for j in range(4):
        label = tk.Label(transformation_frame, text=f"H[{i}][{j}]")
        label.grid(row=i*2, column=j*3, padx=5, pady=2, sticky='w')

        step_label = tk.Label(transformation_frame, text="Step:")
        step_label.grid(row=i*2+1, column=j*3, padx=5, pady=2, sticky='w')

        step_entry = tk.Entry(transformation_frame, width=5)
        step_entry.grid(row=i*2+1, column=j*3+1, padx=5, pady=2)
        step_entry.insert(0, "0.1")  # Default step value
        step_entries[(i, j)] = step_entry

        button_inc = tk.Button(transformation_frame, text="+", width=5, command=lambda r=i, c=j: modify_H_value(r, c, step_entries[(r, c)], is_positive=True))
        button_inc.grid(row=i*2, column=j*3+1, padx=5, pady=2)

        button_dec = tk.Button(transformation_frame, text="-", width=5, command=lambda r=i, c=j: modify_H_value(r, c, step_entries[(r, c)], is_positive=False))
        button_dec.grid(row=i*2, column=j*3+2, padx=5, pady=2)

# ===========================
# 5. Data Loading and Initialization
# ===========================

# Load data
pcd = o3d.io.read_point_cloud(lidar_file)
lidar_points = np.asarray(pcd.points)

# Filter LIDAR points based on thresholds
lidar_points = lidar_points[lidar_points[:, 0] > lidar_thresholds['x_min']]
lidar_points = lidar_points[lidar_points[:, 0] < lidar_thresholds['x_max']]
lidar_points = lidar_points[lidar_points[:, 1] > lidar_thresholds['y_min']]
lidar_points = lidar_points[lidar_points[:, 1] < lidar_thresholds['y_max']]

# Convert LIDAR points to homogeneous coordinates
ones = np.ones((lidar_points.shape[0], 1))
lidar_points_homogeneous = np.hstack((lidar_points, ones))

# Load and process image
image = cv2.imread(image_file)
image = cv2.undistort(image, camera_matrix, dist_coeffs)
image = newimg(image)

image_width = image.shape[1]
image_height = image.shape[0]

# Initialize Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D LIDAR view", width=800, height=600)

# ===========================
# 6. Initial Transformation Matrix
# ===========================

# Initialize the transformation matrix H_lidar_to_cam
H_lidar_to_cam = np.eye(4)
H_lidar_to_cam = np.array([[0.0 ,  0.0 ,-0.0 ,  -0],
                           [ -0.0, 0 , -0.0 ,  0.0],
                           [ 0 ,  0.0, 0 , 0.0],
                           [         0 ,         0  ,        0  ,        0 ] ])
# Update visualization and image initially
update_3d_view()
update_transformation_matrix()

# Initialize OpenCV window
cv2.namedWindow("LIDAR points on image")

# ===========================
# 7. Run the Main GUI Loop
# ===========================

root.mainloop()
