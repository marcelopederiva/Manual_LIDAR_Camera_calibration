import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import cv2

np.set_printoptions(suppress=True, precision=2)

def degrees_to_radians(deg):
    return deg * np.pi / 180.0
  
# Make a gross ajust to start
# Camera extrinsics
pitch_cam = degrees_to_radians(0.0)
roll_cam = degrees_to_radians(0.0)
yaw_cam = degrees_to_radians(0.0)
x_cam, y_cam, z_cam = 0, 0, 0

# LIDAR extrinsics
pitch_lidar = degrees_to_radians(0)
roll_lidar = degrees_to_radians(0)
yaw_lidar = degrees_to_radians(-4.2)
x_lidar, y_lidar, z_lidar = 0, 0, 0

def create_homogeneous_matrix(R, T):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = T
    return H

def update_transformation():
    global R_cam, R_lidar, T_cam, T_lidar, H_cam, H_lidar, H_lidar_to_cam
    R_cam = R.from_euler('xyz', [roll_cam, pitch_cam, yaw_cam]).as_matrix()
    R_lidar = R.from_euler('xyz', [roll_lidar, pitch_lidar, yaw_lidar]).as_matrix()
    T_cam = np.array([x_cam, y_cam, z_cam])
    T_lidar = np.array([x_lidar, y_lidar, z_lidar])
    H_cam = create_homogeneous_matrix(R_cam, T_cam)
    H_lidar = create_homogeneous_matrix(R_lidar, T_lidar)
    H_lidar_to_cam = H_cam @ np.linalg.inv(H_lidar)
    H_lidar_to_cam = np.asarray([[-0.1747, -0.9584,  0.2256, -2.4362],
 [ 0.9837, -0.18 ,  -0.0033 , 2.7897],
 [ 0.0438,  0.2213,  0.9742,  0.2141],
 [ 0. ,     0.  ,    0.    ,  1.    ]])
    print("\nUpdated LIDAR to Camera Transformation Matrix:\n", H_lidar_to_cam)
    print("\nUpdated LIDAR:\n", x_lidar, y_lidar, z_lidar)
    print(roll_lidar, pitch_lidar, yaw_lidar)
    print("\nUpdated Camera:\n", x_cam, y_cam, z_cam)
    print(roll_cam, pitch_cam, yaw_cam)
    update_3d_view()

def transform_lidar_points():
    lidar_points_cam = (H_lidar_to_cam @ lidar_points_homogeneous.T).T
    lidar_points_cam_coords = lidar_points_cam[:, :3]
    lidar_points_cam_coords = lidar_points_cam_coords[:, [1, -1, 0]]  # Swap y and z, and negate z
    # lidar_points_cam_coords[:, 1] *= -1
    lidar_points_cam_coords[:, 0] *= -1
    # lidar_points_cam = lidar_points_cam[lidar_points_cam[:, 2] > 0]
    # lidar_points_cam = lidar_points_cam[lidar_points_cam[:, 2] < 5]
    return lidar_points_cam_coords, lidar_points_cam

def project_points(lidar_points_cam_coords):
    projected_points = np.dot(lidar_points_cam_coords, camera_matrix.T)
    projected_points /= projected_points[:, 2:3]  # Normalize by the z value
    return projected_points

def overlay_points(image, projected_points, distances):
    min_dist, max_dist = distances.min(), distances.max()
    # min_dist, max_dist = 0, 10
    # colors = cv2.applyColorMap(((distances - min_dist) / (max_dist - min_dist) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colors = cv2.applyColorMap(((distances - min_dist) / (max_dist - min_dist) * 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    for point, color in zip(projected_points, colors):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Check if the point is within image bounds
            color = (int(color[0][0]), int(color[0][1]), int(color[0][2]))
            cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)

def update_and_show_image():
    lidar_points_cam_coords, lidar_points_cam = transform_lidar_points()
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

def handle_key_event(key):
    global pitch_cam, roll_cam, yaw_cam, x_cam, y_cam, z_cam, pitch_lidar, roll_lidar, yaw_lidar, x_lidar, y_lidar, z_lidar
    if key == ord('w'):
        z_cam += 0.01
    elif key == ord('s'):
        z_cam -= 0.01
    elif key == ord('a'):
        x_cam -= 0.01
    elif key == ord('d'):
        x_cam += 0.01
    elif key == ord('q'):
        yaw_cam += degrees_to_radians(1)
    elif key == ord('e'):
        yaw_cam -= degrees_to_radians(1)
    elif key == ord('z'):
        pitch_cam += degrees_to_radians(1)
    elif key == ord('c'):
        pitch_cam -= degrees_to_radians(1)

    elif key == ord('i'):
        z_lidar += 0.3
    elif key == ord('k'):
        z_lidar -= 0.3
    elif key == ord('j'):
        x_lidar -= 0.01
    elif key == ord('l'):
        x_lidar += 0.01
    elif key == ord('u'):
        yaw_lidar += degrees_to_radians(1)
    elif key == ord('o'):
        yaw_lidar -= degrees_to_radians(1)
    elif key == ord('b'):
        pitch_lidar += degrees_to_radians(1)
    elif key == ord('m'):
        pitch_lidar -= degrees_to_radians(1)
    else:
        return False  # Key not handled
    return True  # Key handled

def update_3d_view():
    global vis, lidar_points
    vis.clear_geometries()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

# number_file = '4692'
number_file = '6572'
# Load LIDAR points
pcd = o3d.io.read_point_cloud("lidar_data/lidar_" + number_file + ".pcd")
lidar_points = np.asarray(pcd.points)
# lidar_points[:, 2] = -lidar_points[:, 2]
lidar_points = lidar_points[lidar_points[:, 0] > -8]
lidar_points = lidar_points[lidar_points[:, 0] < 8]
lidar_points = lidar_points[lidar_points[:, 1] > -8]
lidar_points = lidar_points[lidar_points[:, 1] < 8]
# lidar_points = lidar_points[lidar_points[:, 1] < 10]
# Add a column of ones to the LIDAR points to make them homogeneous
ones = np.ones((lidar_points.shape[0], 1))
lidar_points_homogeneous = np.hstack((lidar_points, ones))

# Load the image
image = cv2.imread("images/image_" + number_file + ".jpg")

# Camera matrix Intrinsics // Below Matrix is Generic
camera_matrix = np.array([
    [1, 0., 1],
    [0., 1, 1],
    [0., 0., 1.]
])

# Image dimensions
image_width = image.shape[1]
image_height = image.shape[0]

# Initialize Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D LIDAR view", width=800, height=600)
update_3d_view()

# Initialize transformations
update_transformation()
update_and_show_image()

cv2.namedWindow("LIDAR points on image")

while True:
    key = cv2.waitKey(0) & 0xFF  # Wait for key event
    if key == 27:  # Escape key to exit
        break
    if handle_key_event(key):  # If key is handled, update the transformation and image
        update_transformation()
        update_and_show_image()

cv2.destroyAllWindows()
vis.destroy_window()
