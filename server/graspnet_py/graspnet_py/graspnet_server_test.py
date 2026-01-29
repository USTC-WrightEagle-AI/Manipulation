import rclpy
from rclpy.node import Node
import torch
import torch.nn as nn
import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image
import open3d as o3d
import glob
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Point, Quaternion
from grasp_srv_interface.srv import Graspnet             
from pathlib import Path
import json


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Set graspnet API path
GraspNetDIR = '/root/host_home/ros2_ws/graspnet-baseline'
sys.path.append(os.path.join(GraspNetDIR, 'models'))
sys.path.append(os.path.join(GraspNetDIR, 'dataset'))
sys.path.append(os.path.join(GraspNetDIR, 'utils'))


# Import graspnet module
from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image



# Define server node
class GraspnetServer(Node):
    def __init__(self):
        super().__init__('graspnet_server')
        self.get_logger().info('Graspnet service node starting')

        # Graspnet model parameters
        # Register parameter values
        self.declare_parameter('checkpoint_path', 
                               '/root/host_home/ros2_ws/graspnet-baseline/logs/log_kn/checkpoint.tar')
        self.declare_parameter('num_point', 20000)
        self.declare_parameter('num_view', 300)
        self.declare_parameter('collision_thresh', -0.01)
        self.declare_parameter('voxel_size', 0.01)

        # Get parameter values
        self.checkpoint_path = self.get_parameter('checkpoint_path').get_parameter_value().string_value
        self.num_point = self.get_parameter('num_point').get_parameter_value().integer_value
        self.num_view = self.get_parameter('num_view').get_parameter_value().integer_value
        self.collision_thresh = self.get_parameter('collision_thresh').get_parameter_value().double_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
     
        # Create srv service
        self.srv = self.create_service(
            Graspnet,
            'graspnet_service',
            self.handle_graspnet_request
        )
        self.get_logger().info('Graspnet service started, waiting for requests...')

        # Load computing device
        self.get_logger().info(f"Computing device: {device}")
        print("CUDA version:", torch.version.cuda)

        # Graspnet model loading
        self.get_logger().info("Loading Graspnet model...")
        self.graspnet_net = self.graspnet_get_net()
        self.get_logger().info("Graspnet model loaded.")

    # Graspnet model loading function
    def graspnet_get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net
    
    def get_and_process_data(self,data_dir):
    # load data
        color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
        depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
        workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
        with open(os.path.join(data_dir, 'camera.json')) as f:
            params = json.load(f)
        intrinsic = np.array(params['camera_matrix'])
        print(intrinsic)
        # factor_depth = meta['factor_depth']
        factor_depth = [[1000.]]
        print(factor_depth)

        # generate cloud
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        mask = (workspace_mask & (depth > 0))
        mask = mask.astype(bool)
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # sample points
        print(len(cloud_masked))
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud_ = o3d.geometry.PointCloud()
        # cloud_.points = o3d.utility.Vector3dVector(cloud_sampled.astype(np.float32))  #cloud_masked -> cloud_sampled
        # cloud_.colors = o3d.utility.Vector3dVector(color_sampled.astype(np.float32))  #cloud_masked -> cloud_sampled
        cloud_.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud_.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32)) 
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud_
    
    def find_ply_file(self, input_path):
        """
        Find .ply file in specified directory
        """
        ply_file = glob.glob(os.path.join(input_path, '*.ply'))
        if len(ply_file) == 0:
            self.get_logger().error(f"No .ply file found in directory {input_path}.")
            return None
        return ply_file[0]
    
    def load_ply_as_np(self, ply_file):
        """
        Load .ply file and convert to numpy array
        """
        self.get_logger().info(f"Loading point cloud file: {ply_file}")
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points)
        return points
        
    def process_point_cloud(self, points):
        """
        Process input numpy point cloud data, perform sampling and package into model input format
        """
        # Sample point cloud
        self.get_logger().info("Processing point cloud data...")
        if len(points) >= self.num_point:
            idxs = np.random.choice(len(points), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(points))
            idxs2 = np.random.choice(len(points), self.num_point-len(points), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = points[idxs]

        # Convert data type to model required format (end_points dictionary)
        end_points = dict()
        cloud_sampled_torch = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        end_points['point_clouds'] = cloud_sampled_torch.to(device)
        end_points['cloud_colors'] = None  # If color information is available, can be added here

        # Create open3d point cloud object for collision detection
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(points.astype(np.float32))
        self.get_logger().info("Point cloud data processing completed.")
        return end_points, cloud_o3d
    
    def get_grasps(self,net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg
    
    def collision_detection(self,gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def vis_grasps(self,gg, cloud):
        #print(gg)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.001, origin=[0, 0, 0])
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers, coordinate_frame])

    def handle_graspnet_request(self, request, response):
        mode = request.mode
        # === gsam+vggt+graspnet mode ===
        if mode == 1:
            self.get_logger().info('Request received, processing (vggt point cloud input)...')
            input_path = request.input_path
            ply_file = self.find_ply_file(input_path)
            input_cloud = self.load_ply_as_np(ply_file)
            end_points, cloud = self.process_point_cloud(input_cloud)
        # === gsam+graspnet mode ===
        elif mode == 0:
            self.get_logger().info('Request received, processing (gsam image input)...')
            input_path = request.input_path
            end_points, cloud = self.get_and_process_data(input_path)
        gg = self.get_grasps(self.graspnet_net, end_points)
        o3d.visualization.draw_geometries([cloud])
        # Log original grasp pose count
        self.get_logger().info(f"Original grasp pose count: {len(gg)}")
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
            self.get_logger().info(f"Grasp pose count after collision detection: {len(gg)}")
        if len(gg) == 0:
            response.grasp_poses = []
            response.success = False
            response.message = "No valid grasp pose detected, please adjust input point cloud or parameters and retry."
            return response
        
        gg.nms()
        gg.sort_by_score()
        best_grasp = gg[0]
        print(best_grasp)

        R_align = np.array([
        [0, 0, 1],  # newX = oldZ
        [0, 1, 0],  # newY = oldY
        [-1, 0, 0],  # newZ = -oldX  (或者相当于 oldX = -newZ)
        ], dtype=float)

        best_grasp.rotation_matrix = best_grasp.rotation_matrix @ R_align
        #best_grasp.rotation_matrix = best_grasp.rotation_matrix

        pose_msg = Pose()
         
        # Set position
        pose_msg.position = Point(
            x=float(best_grasp.translation[0]),
            y=float(best_grasp.translation[1]),
            z=float(best_grasp.translation[2])
        )
           
        # Set orientation
        quat = R.from_matrix(best_grasp.rotation_matrix).as_quat()  # xyzw
        pose_msg.orientation = Quaternion(
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
            w=float(quat[3])  
        )
        
        self.vis_grasps(gg[:1], cloud)

        response.grasp_poses = [pose_msg]

        response.success = True
        response.message = f"Successfully detected 1 best grasp pose."
        # Prepare response data       
        return response
    

def main(args=None):
    # Initialize ros2 client library
    rclpy.init(args=args)

    # Create server node
    graspnet_server = GraspnetServer()

    try:
        rclpy.spin(graspnet_server)
    except KeyboardInterrupt:
        graspnet_server.get_logger().info('Server node manually closed by user...')
    finally:
        graspnet_server.get_logger().info('Server node closed.') 
        graspnet_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()