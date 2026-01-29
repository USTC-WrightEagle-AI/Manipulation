import socket
import struct
import json
import cv2
import os
import numpy as np
import sys
import argparse
from robots import R1Robot, R1Robot_left
from scipy.spatial.transform import Rotation as R
import pandas as pd

# --- Configuration parameters ---
parser = argparse.ArgumentParser(description="Grasp Socket Client")
parser.add_argument('--server_host', default = '192.168.31.44', required=False, help='Server host address')
parser.add_argument('--server_port', default = 9090, type=int, required=False, help='Server port')
parser.add_argument('--rgb_path', default = './images/color.png', required=False, help='Path to RGB image')
parser.add_argument('--depth_path', default = './images/depth.png',required=False, help='Path to Depth image')
parser.add_argument('--text_prompt', required=False, help='Text prompt for grasping')
parser.add_argument('--mode', default=0, required=False, help='Mode of operation -- 0: graspnet+groundesam, 1:graspnet+groundedsam+vggt')
cfgs = parser.parse_args()

# --- Message helper functions ---
def send_msg(sock, msg_bytes):
    try:
        len_header = struct.pack('>Q', len(msg_bytes))
        sock.sendall(len_header)
        sock.sendall(msg_bytes)
    except Exception as e:
        print(f"Error sending message: {e}")
        raise

def recvall(sock, n):
    """
    Ensure receiving n bytes of data from socket
    """
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def recv_msg(sock):
    try:
        # Receive message length
        len_header_bytes = sock.recv(8)

        # Check if connection is closed
        # If no data is received, the connection is closed
        if not len_header_bytes:
            print("Socket connection closed")
            return None
        msg_len = struct.unpack('>Q', len_header_bytes)[0]

        # Loop receive until receiving complete message
        msg_bytes = b''
        while len(msg_bytes) < msg_len:
            remaining_bytes = msg_len - len(msg_bytes)
            bytes_to_recv = min(4096, remaining_bytes)
            chunk = sock.recv(bytes_to_recv)
            if not chunk:
                raise OSError("Socket connection closed, message reception interrupted")
            msg_bytes += chunk
        return msg_bytes
    
    except (ConnectionResetError, BrokenPipeError, OSError) as e:
        print(f"Socket message reception failed: {e}")
        raise
        

def run_grasp_client(server_host, server_port, rgb_path, depth_path, text_prompt, mode):
    print(f"Connecting to server {server_host}:{server_port}...")

    # --- Data preparation ---
    print("Preparing data...")
    # 1. Prepare JSON
    request_data = {
        'text_prompt': text_prompt,
        'mode': mode
    }
    json_bytes = json.dumps(request_data).encode('utf-8')
    print(f'JSON data: {request_data}')

    # 2. Prepare color image
    try:
        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_image is None:
            print(f"Cannot read RGB image: {rgb_path}")
            return
        success, rgb_bytes_encoded = cv2.imencode('.png', rgb_image)
        rgb_bytes = rgb_bytes_encoded.tobytes()
        print(f'RGB image {rgb_path} size: {len(rgb_bytes)} bytes')
    except Exception as e:
        print(f"Error reading RGB image: {e}")
        return
    
    # 3. Prepare depth image
    try:
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            print(f"Cannot read depth image: {depth_path}")
            return
        success, depth_bytes_encoded = cv2.imencode('.png', depth_image)
        depth_bytes = depth_bytes_encoded.tobytes()
        print(f'Depth image {depth_path} size: {len(depth_bytes)} bytes')
    except Exception as e:
        print(f"Error reading depth image: {e}")
        return
    
    # --- Establish connection and send data ---
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"Connecting to server {server_host}:{server_port}...")
            s.connect((server_host, server_port))
            print("Connection successful!")

            # --- Send data ---
            # Send JSON
            print("Sending JSON data...")
            send_msg(s, json_bytes)
            print("JSON data sent complete.")
            # Send RGB image
            print("Sending RGB image data...")
            send_msg(s, rgb_bytes)
            print("RGB image data sent complete.")
            # Send depth image
            print("Sending depth image data...")
            send_msg(s, depth_bytes)
            print("Depth image data sent complete.")
            print("All data sent complete, waiting for server response...")

            # --- Receive response ---
            response_bytes = recv_msg(s)
            if response_bytes is None:
                print("Did not receive server response.")
                return
            print(f"Received server response")
            response_data = json.loads(response_bytes.decode('utf-8'))
            print(json.dumps(response_data, indent=2))
            if response_data.get('success'):
                print("Grasp point calculation successful!")
            else:
                print("Grasp point calculation failed.")
            
            pose_response = response_data.get('grasp_poses', {})

            grasp_pose = pose_response

            return grasp_pose
    except ConnectionAbortedError:
        print("Connection aborted by server.")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

def xyzrpy_to_matrix(x, y, z, roll, pitch, yaw):

    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = r.as_matrix()
    transformation_matrix[:3, 3] = [x, y, z]

    return transformation_matrix

def transform_pose(pose, cam2base):

    x, y, z, qx, qy, qz, qw = pose

    # Create rotation quaternion object
    rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = rotation.as_matrix()  # 3x3 rotation matrix
    
    # Construct 4x4 homogeneous transformation matrix
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = [x, y, z]
    
    # 2. Use cam2base transformation
    new_pose_matrix = cam2base @ pose_matrix
    
    # 3. Extract results from new 4x4 matrix
    new_rotation_matrix = new_pose_matrix[:3, :3]
    new_translation = new_pose_matrix[:3, 3]
    
    # Rotate 90 degrees to adapt to robotic arm issue
    r = np.array([[1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]])
    new_rotation_matrix = new_rotation_matrix @ r

    # Get new quaternion from rotation matrix
    new_rotation = R.from_matrix(new_rotation_matrix)
    new_quat = new_rotation.as_quat()  # Get quaternion (qx, qy, qz, qw)
    
    # print(new_translation)
    # exit()
    # Return new pose
    return np.hstack([new_translation, new_quat])

def get_pose(gg, retreat_distance=0.03):
    position = gg.translation
    rotation_matrix = gg.rotation_matrix

    # Normal direction is the third column (z-axis) of rotation_matrix
    approach_vector = rotation_matrix[:, 0]  # shape: (3,)
    
    # Retreat retreat_distance along normal direction
    adjusted_position = position - approach_vector * retreat_distance

    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # [x, y, z, w]

    pose = np.concatenate((adjusted_position, quaternion))
    return pose

class gg_data:
    def __init__(self):
        self.translation = None
        self.rotation_matrix = None

def grasp(cam2base_path, grasp_pose):
    print("--------------------------",grasp_pose)

    qx, qy, qz, qw = grasp_pose['orientation']
    rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = rotation.as_matrix()



    gg = gg_data()
    gg.translation = grasp_pose['position']
    gg.rotation_matrix = rotation_matrix

    print("--------------------------",gg.rotation_matrix)

    robot = R1Robot('r1')
    pose = robot.read_current_pose()
    x, y, z, roll, pitch, yaw = pose
    end2base = xyzrpy_to_matrix(x, y, z, roll, pitch, yaw)
    # exit()
    data = pd.read_csv(cam2base_path, header=None)
    cam2end = data.to_numpy()    
    cam2base = end2base @ cam2end
    # print(end2base)
    print(cam2base)
    print(end2base)
    # exit()
    # print(get_pose(gg[0]))
    # ipdb.set_trace()

    
    # pose_base_0 = transform_pose(get_pose(gg,retreat_distance=0.50), cam2base)
    pose_base_1 = transform_pose(get_pose(gg,retreat_distance=0.08), cam2base)
    pose_base_2 = transform_pose(get_pose(gg,retreat_distance=0.04), cam2base)
    print(pose)    
    print(pose_base_2)
    # pose_base[2] = pose_base[2] + 0.04      # Collision prevention
    # exit()+
    print("*********************")
    print(robot.read_current_pose())
    # exit()
    # robot.set_endpose(pose_base_0)
    robot.set_endpose(pose_base_1)
    robot.set_endpose(pose_base_2)
    print("after grasp ee pose: \n", robot.read_current_pose())
    exit()
    # print(pose_base)`


if __name__ == "__main__":
    print("Parameters parsed")
    if not os.path.isfile(cfgs.rgb_path):
        print(f"RGB image file does not exist: {cfgs.rgb_path}")
        sys.exit(1)
    if not os.path.isfile(cfgs.depth_path):
        print(f"Depth image file does not exist: {cfgs.depth_path}")
        sys.exit(1)
    grasp_pose = run_grasp_client(cfgs.server_host, 
                     cfgs.server_port, 
                     cfgs.rgb_path, 
                     cfgs.depth_path, 
                     cfgs.text_prompt, 
                     cfgs.mode)
    cam2base_path = './cam2end_H.csv'
    grasp(cam2base_path, grasp_pose)