import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import cv2
import tkinter as tk
from tkinter import scrolledtext

# ===========================
# 1. Initial Configuration
# ===========================

# Configuration of angular parameters
ANGULAR_PARAMS = {
    'pitch_cam', 'roll_cam', 'yaw_cam',
    'pitch_lidar', 'roll_lidar', 'yaw_lidar'
}

# --- Camera Extrinsics ---
pitch_cam_deg = 8.0
roll_cam_deg = 0.0
yaw_cam_deg = -1.0
x_cam, y_cam, z_cam = 0.15, -1.4, 1.46

# --- LIDAR Extrinsics ---
pitch_lidar_deg = 0.5
roll_lidar_deg = 0.0
yaw_lidar_deg = -5.5
x_lidar, y_lidar, z_lidar = 0, -2.74, 1.83

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
    'y_max': 30,
    'z_min': None,  # If necessary, add. Remember that LIDAR is at Z ~ 1.83
    'z_max': None   # If necessary, add. Remember that LIDAR is at Z ~ 1.83
}

# --- File Number ---
lidar_file = "data/lidar/lidar_hishort_2.pcd"
image_file = "data/image/image_hishort_2.jpg"


# =================================
# =================================
# 2. ALL THE HARD WORK GOES HERE
# =================================
# =================================


def degrees_to_radians(deg):
    return deg * np.pi / 180.0

def rad2deg(rad):
    return rad * 180.0 / np.pi

pitch_cam = degrees_to_radians(pitch_cam_deg) 
roll_cam = degrees_to_radians(roll_cam_deg) 
yaw_cam = degrees_to_radians(yaw_cam_deg)

pitch_lidar = degrees_to_radians(pitch_lidar_deg) 
roll_lidar = degrees_to_radians(roll_lidar_deg)
yaw_lidar = degrees_to_radians(yaw_lidar_deg)


def m_Rx(angle):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    return Rx

def m_Ry(angle):
    Ry = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    return Ry

def m_Rz(angle):
    Rz = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return Rz

def create_homogeneous_matrix(R_matrix, T_vector):
    H = np.eye(4)
    H[:3, :3] = R_matrix
    H[:3, 3] = T_vector
    return H

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

# ===========================
# 4. Update and Interaction Functions
# ===========================

def update_parameter(param_name, operation):
    global x_cam, y_cam, z_cam, pitch_cam, roll_cam, yaw_cam
    global x_lidar, y_lidar, z_lidar, pitch_lidar, roll_lidar, yaw_lidar

    # Mapping of parameters to their global variables
    params = {
        'x_cam': 'x_cam',
        'y_cam': 'y_cam',
        'z_cam': 'z_cam',
        'pitch_cam': 'pitch_cam',
        'roll_cam': 'roll_cam',
        'yaw_cam': 'yaw_cam',
        'x_lidar': 'x_lidar',
        'y_lidar': 'y_lidar',
        'z_lidar': 'z_lidar',
        'pitch_lidar': 'pitch_lidar',
        'roll_lidar': 'roll_lidar',
        'yaw_lidar': 'yaw_lidar',
    }

    var_name = params[param_name]
    step_var = globals()[f"{param_name}_step_entry"]
    try:
        step_input = float(step_var.get())
    except ValueError:
        print(f"Invalid step size for {param_name}. Using step=0.")
        step_input = 0.0

    # Check if the parameter is angular to convert degrees to radians
    if param_name in ANGULAR_PARAMS:
        step = degrees_to_radians(step_input)
    else:
        step = step_input

    if operation == 'increase':
        globals()[var_name] += step
    elif operation == 'decrease':
        globals()[var_name] -= step
    update_transformation()

def update_transformation():
    global R_cam, R_lidar, H_lidar_to_cam

    R_cam = m_Rz(yaw_cam) @ m_Ry(roll_cam) @ m_Rx(pitch_cam)
    R_lidar = m_Rz(yaw_lidar) @ m_Ry(roll_lidar) @ m_Rx(pitch_lidar)

    R = R_cam @ R_lidar.T

    T = np.array([x_lidar - x_cam,
                  y_lidar - y_cam,
                  z_lidar - z_cam])

    H_lidar_to_cam = create_homogeneous_matrix(R, T)

    # Console printing
    print("\nUpdated LIDAR to Camera Transformation Matrix:\n", H_lidar_to_cam)
    print("\nUpdated LIDAR Position:\n", x_lidar, y_lidar, z_lidar)
    print("Roll:", round(rad2deg(roll_lidar), ndigits=1), "Pitch:", round(rad2deg(pitch_lidar), ndigits=1), "Yaw:", round(rad2deg(yaw_lidar), ndigits=1))
    print("\nUpdated Camera Position:\n", x_cam, y_cam, z_cam)
    print("Roll:", round(rad2deg(roll_cam), ndigits=1), "Pitch:", round(rad2deg(pitch_cam), ndigits=1), "Yaw:", round(rad2deg(yaw_cam), ndigits=1))

    # Update the information area in the GUI
    info_text.config(state='normal')
    info_text.delete('1.0', tk.END)
    info_content = (
        f"Updated LIDAR to Camera Transformation Matrix:\n{H_lidar_to_cam}\n\n"
        f"Updated LIDAR Position:\nX: {x_lidar:.3f}, Y: {y_lidar:.3f}, Z: {z_lidar:.3f}\n"
        f"Roll: {rad2deg(roll_lidar):.1f}°, Pitch: {rad2deg(pitch_lidar):.1f}°, Yaw: {rad2deg(yaw_lidar):.1f}°\n\n"
        f"Updated Camera Position:\nX: {x_cam:.3f}, Y: {y_cam:.3f}, Z: {z_cam:.3f}\n"
        f"Roll: {rad2deg(roll_cam):.1f}°, Pitch: {rad2deg(pitch_cam):.1f}°, Yaw: {rad2deg(yaw_cam):.1f}°\n"
    )
    info_text.insert(tk.END, info_content)
    info_text.config(state='disabled')

    update_3d_view()
    update_and_show_image()

def close_program():
    root.destroy()
    cv2.destroyAllWindows()
    vis.destroy_window()

# ===========================
# 5. GUI Configuration
# ===========================

# Creating the GUI first to ensure 'info_text' is defined before 'update_transformation' is called
root = tk.Tk()
root.title("Transformation Controls")

# --- Frame for camera parameters ---
camera_frame = tk.LabelFrame(root, text="Camera Parameters")
camera_frame.pack(padx=10, pady=5, fill="both", expand="yes")

# --- Frame for LIDAR parameters ---
lidar_frame = tk.LabelFrame(root, text="LIDAR Parameters")
lidar_frame.pack(padx=10, pady=5, fill="both", expand="yes")

# --- Frame for transformation information ---
info_frame = tk.LabelFrame(root, text="Transformation Information")
info_frame.pack(padx=10, pady=5, fill="both", expand="yes")

# --- ScrolledText Widget to display information ---
info_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, width=50, height=15, state='disabled')
info_text.pack(padx=5, pady=5, fill="both", expand=True)

# --- Exit button ---
exit_button = tk.Button(root, text="Exit", command=close_program)
exit_button.pack(pady=10)

# --- Camera parameters with step size inputs ---
params_cam = [
    'x_cam',
    'y_cam',
    'z_cam',
    'pitch_cam',
    'roll_cam',
    'yaw_cam',
]

for idx, param_name in enumerate(params_cam):
    label = tk.Label(camera_frame, text=param_name)
    label.grid(row=idx, column=0, padx=5, pady=2, sticky='w')
    
    button_inc = tk.Button(camera_frame, text="+", width=5, command=lambda p=param_name: update_parameter(p, 'increase'))
    button_inc.grid(row=idx, column=1, padx=5, pady=2)
    
    button_dec = tk.Button(camera_frame, text="-", width=5, command=lambda p=param_name: update_parameter(p, 'decrease'))
    button_dec.grid(row=idx, column=2, padx=5, pady=2)
    
    # Add input field for step size with unit indication
    if param_name in ANGULAR_PARAMS:
        step_label = tk.Label(camera_frame, text="Step (deg):")
        default_step = "1"  # 1 degree
    else:
        step_label = tk.Label(camera_frame, text="Step (m):")
        # Set default values for step sizes
        if 'z_cam' in param_name:
            default_step = "0.3"
        else:
            default_step = "0.01"
    
    step_label.grid(row=idx, column=3, padx=5, pady=2)
    
    step_entry = tk.Entry(camera_frame, width=10)
    step_entry.grid(row=idx, column=4, padx=5, pady=2)
    step_entry.insert(0, default_step)
    
    # Store the Entry reference for use in the functions
    globals()[f"{param_name}_step_entry"] = step_entry

# --- LIDAR parameters with step size inputs ---
params_lidar = [
    'x_lidar',
    'y_lidar',
    'z_lidar',
    'pitch_lidar',
    'roll_lidar',
    'yaw_lidar',
]

for idx, param_name in enumerate(params_lidar):
    label = tk.Label(lidar_frame, text=param_name)
    label.grid(row=idx, column=0, padx=5, pady=2, sticky='w')
    
    button_inc = tk.Button(lidar_frame, text="+", width=5, command=lambda p=param_name: update_parameter(p, 'increase'))
    button_inc.grid(row=idx, column=1, padx=5, pady=2)
    
    button_dec = tk.Button(lidar_frame, text="-", width=5, command=lambda p=param_name: update_parameter(p, 'decrease'))
    button_dec.grid(row=idx, column=2, padx=5, pady=2)
    
    # Add input field for step size with unit indication
    if param_name in ANGULAR_PARAMS:
        step_label = tk.Label(lidar_frame, text="Step (deg):")
        default_step = "1"  # 1 degree
    else:
        step_label = tk.Label(lidar_frame, text="Step (m):")
        # Set default values for step sizes
        if 'z_lidar' in param_name:
            default_step = "0.3"
        else:
            default_step = "0.01"
    
    step_label.grid(row=idx, column=3, padx=5, pady=2)
    
    step_entry = tk.Entry(lidar_frame, width=10)
    step_entry.grid(row=idx, column=4, padx=5, pady=2)
    step_entry.insert(0, default_step)
    
    # Store the Entry reference for use in the functions
    globals()[f"{param_name}_step_entry"] = step_entry

# ===========================
# 6. Data Loading and Visualization Initialization
# ===========================

# Load data
pcd = o3d.io.read_point_cloud(lidar_file)
lidar_points = np.asarray(pcd.points)

# Filtering LIDAR points based on thresholds
lidar_points = lidar_points[lidar_points[:, 0] > lidar_thresholds['x_min']]
lidar_points = lidar_points[lidar_points[:, 0] < lidar_thresholds['x_max']]
lidar_points = lidar_points[lidar_points[:, 1] > lidar_thresholds['y_min']]
lidar_points = lidar_points[lidar_points[:, 1] < lidar_thresholds['y_max']]
# If you want to add filters for the Z-axis, uncomment and adjust:
if ((lidar_thresholds['z_min'] != None) and (lidar_thresholds['z_max']!= None)):
    lidar_points = lidar_points[lidar_points[:, 2] > lidar_thresholds['z_min']]
    lidar_points = lidar_points[lidar_points[:, 2] < lidar_thresholds['z_max']]

# Convert LIDAR points to homogeneous coordinates
ones = np.ones((lidar_points.shape[0], 1))
lidar_points_homogeneous = np.hstack((lidar_points, ones))

# Load and process image
image = cv2.imread(image_file)
image = cv2.undistort(image, camera_matrix, dist_coeffs)
image_width = image.shape[1]
image_height = image.shape[0]

# Initialize Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D LIDAR view", width=800, height=600)

# Initialize transformation matrices before any updates
R_cam = m_Rz(yaw_cam) @ m_Ry(roll_cam) @ m_Rx(pitch_cam)
R_lidar = m_Rz(yaw_lidar) @ m_Ry(roll_lidar) @ m_Rx(pitch_lidar)
R = R_cam @ R_lidar.T
T = np.array([x_lidar - x_cam,
              y_lidar - y_cam,
              z_lidar - z_cam])
H_lidar_to_cam = create_homogeneous_matrix(R, T)

# Update visualization and image initially
update_3d_view()
update_transformation()
update_and_show_image()

# Initialize OpenCV window
cv2.namedWindow("LIDAR points on image")

# ===========================
# 7. Run the Main GUI Loop
# ===========================

root.mainloop()
