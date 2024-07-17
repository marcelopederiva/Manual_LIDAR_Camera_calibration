import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import cv2

np.set_printoptions(suppress=True, precision=4)

def degrees_to_radians(deg):
    return deg * np.pi / 180.0

# Camera extrinsics
pitch_cam = degrees_to_radians(0.0)
roll_cam = degrees_to_radians(0.0)
yaw_cam = degrees_to_radians(0.0)
x_cam, y_cam, z_cam = 0.0, 0.0, 0.0

# LIDAR extrinsics
pitch_lidar = degrees_to_radians(0.0)
roll_lidar = degrees_to_radians(0.0)
yaw_lidar = degrees_to_radians(0.0)
x_lidar, y_lidar, z_lidar = 0.0, 0.0, 0.0


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
    print("\nUpdated LIDAR to Camera Transformation Matrix:\n", H_lidar_to_cam)
    print("\nUpdated LIDAR:\n", x_lidar, y_lidar, z_lidar)
    print(roll_lidar, pitch_lidar, yaw_lidar)
    print("\nUpdated Camera:\n", x_cam, y_cam, z_cam)
    print(roll_cam, pitch_cam, yaw_cam)
    update_3d_view()

def transform_lidar_points(lidar_points_homogeneous):
    lidar_points_cam = (H_lidar_to_cam @ lidar_points_homogeneous.T).T
    lidar_points_cam_coords = lidar_points_cam[:, :3]
    lidar_points_cam_coords = lidar_points_cam_coords[:, [1, -1, 0]]  # Swap y and z, and negate z
    lidar_points_cam_coords[:, 0] *= -1
    return lidar_points_cam_coords, lidar_points_cam

def project_points(lidar_points_cam_coords, camera_matrix):
    projected_points = np.dot(lidar_points_cam_coords, camera_matrix.T)
    projected_points /= projected_points[:, 2:3]  # Normalize by the z value
    return projected_points

def overlay_points(image, projected_points, distances):
    min_dist, max_dist = distances.min(), distances.max()
    colors = cv2.applyColorMap(((distances - min_dist) / (max_dist - min_dist) * 255).astype(np.uint8), cv2.COLORMAP_RAINBOW)
    for point, color in zip(projected_points, colors):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Check if the point is within image bounds
            color = (int(color[0][0]), int(color[0][1]), int(color[0][2]))
            cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)

def update_and_show_image(lidar_points_homogeneous, image, camera_matrix, window_name):
    lidar_points_cam_coords, lidar_points_cam = transform_lidar_points(lidar_points_homogeneous)
    projected_points = project_points(lidar_points_cam_coords, camera_matrix)
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
    cv2.imshow(window_name, resized_image)
    # cv2.imwrite(window_name,resized_image)
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
        z_lidar += 0.01
    elif key == ord('k'):
        z_lidar -= 0.01
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
    global vis, lidar_points1, lidar_points2
    # vis.clear_geometries()
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(lidar_points1)
    # vis.add_geometry(pcd1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(lidar_points2)
    vis.add_geometry(pcd2)
    vis.poll_events()
    vis.update_renderer()
  
# Exemple of number file
number_file1 = '4692'
number_file2 = '6572'
# Load LIDAR points
pcd1 = o3d.io.read_point_cloud("lidar_data/lidar_" + number_file1 + ".pcd")
lidar_points1 = np.asarray(pcd1.points)
# lidar_points1 = lidar_points1[lidar_points1[:, 0] > -8]
# lidar_points1 = lidar_points1[lidar_points1[:, 0] < 8]
# lidar_points1 = lidar_points1[lidar_points1[:, 1] > -8]
# lidar_points1 = lidar_points1[lidar_points1[:, 1] < 8]

pcd2 = o3d.io.read_point_cloud("lidar_data/lidar_" + number_file2 + ".pcd")
lidar_points2 = np.asarray(pcd2.points)
# lidar_points2 = lidar_points2[lidar_points2[:, 0] > -8]
# lidar_points2 = lidar_points2[lidar_points2[:, 0] < 8]
# lidar_points2 = lidar_points2[lidar_points2[:, 1] > -8]
# lidar_points2 = lidar_points2[lidar_points2[:, 1] < 8]

# Add a column of ones to the LIDAR points to make them homogeneous
ones1 = np.ones((lidar_points1.shape[0], 1))
lidar_points_homogeneous1 = np.hstack((lidar_points1, ones1))

ones2 = np.ones((lidar_points2.shape[0], 1))
lidar_points_homogeneous2 = np.hstack((lidar_points2, ones2))

# Load the images
image1 = cv2.imread("images/image_" + number_file1 + ".jpg")
image2 = cv2.imread("images/image_" + number_file2 + ".jpg")

# Camera matrix Generic
camera_matrix = np.array([
    [1., 0., 1.],
    [0., 1., 1.],
    [0., 0., 1.]
])

# Image dimensions
image_width = image1.shape[1]
image_height = image1.shape[0]

# Initialize Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D LIDAR view", width=800, height=600)
update_3d_view()

# Initialize transformations
update_transformation()
update_and_show_image(lidar_points_homogeneous1, image1, camera_matrix, "LIDAR points on image 1")
update_and_show_image(lidar_points_homogeneous2, image2, camera_matrix, "LIDAR points on image 2")

cv2.namedWindow("LIDAR points on image 1")
cv2.namedWindow("LIDAR points on image 2")

while True:
    key = cv2.waitKey(0) & 0xFF  # Wait for key event
    if key == 27:  # Escape key to exit
        break
    if handle_key_event(key):  # If key is handled, update the transformation and image
        update_transformation()
        update_and_show_image(lidar_points_homogeneous1, image1, camera_matrix, "LIDAR points on image 1")
        update_and_show_image(lidar_points_homogeneous2, image2, camera_matrix, "LIDAR points on image 2")

cv2.destroyAllWindows()
vis.destroy_window()
