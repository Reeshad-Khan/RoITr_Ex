import pcl
import numpy as np
import open3d as o3d

def load_kitti_point_cloud(bin_path):
    """Load KITTI point cloud from a binary file."""
    scan = np.fromfile(bin_path, dtype=np.float32)
    points = scan.reshape((-1, 4))[:, :3]  # We only take the x, y, z components
    return points

def create_pcl_point_cloud(points):
    """Convert Numpy array to PCL Point Cloud."""
    cloud = pcl.PointCloud()
    cloud.from_array(np.array(points, dtype=np.float32))
    return cloud

def remove_outliers(point_cloud, mean_k=50, std_dev=1.0):
    """Remove outliers from the point cloud using Statistical Outlier Removal."""
    sor = point_cloud.make_statistical_outlier_filter()
    sor.set_mean_k(mean_k)
    sor.set_std_dev_mul_thresh(std_dev)
    return sor.filter()

def compute_normals(point_cloud, radius=0.1):
    """Compute normal vectors for the point cloud using a radius search."""
    ne = point_cloud.make_NormalEstimation()
    tree = point_cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(radius)
    normals = ne.compute()
    return normals

def visualize_point_cloud(points, normals):
    """Visualize the point cloud with normals."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)

def main():
    bin_path = 'path/to/your/kitti_file.bin'  # Update this path depending on your file structure
    points = load_kitti_point_cloud(bin_path)
    pcl_point_cloud = create_pcl_point_cloud(points)
    
    # Preprocess: Remove outliers
    filtered_cloud = remove_outliers(pcl_point_cloud)
    
    # Compute normals using a more appropriate radius based on the expected scale of objects in KITTI
    normals = compute_normals(filtered_cloud, radius=0.3)  # Adjust the radius based on point cloud density
    
    # Extract points and normals for visualization
    filtered_points = np.array(filtered_cloud)
    normals_array = np.array(normals.to_array())
    
    # Visualize
    visualize_point_cloud(filtered_points, normals_array)

if __name__ == '__main__':
    main()
