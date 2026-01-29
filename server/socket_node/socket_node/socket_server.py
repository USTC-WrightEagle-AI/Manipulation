import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
import time
from scipy.spatial.transform import Rotation as R
# --- SocketBridgeNode imports ---
import socket 
import threading
import struct
import json
import os
import uuid

# --- Image processing imports ---
import cv2
import numpy as np

# --- Service interface imports ---
from grasp_srv_interface.srv import TriggerGrasp

# --- Auxiliary functions ---
def send_msg(sock, msg_bytes):
    try:
        # Send message length
        msg_length = struct.pack('>Q', len(msg_bytes))
        sock.sendall(msg_length)
        # Send message content
        sock.sendall(msg_bytes)
    except (ConnectionResetError, BrokenPipeError, OSError) as e:
        print(f"Socket message sending failed: {e}")
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
        # len_header_bytes = recvall(sock,8)
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

class SocketBridgeNode(Node):
    def __init__(self):
        super().__init__('socket_bridge_node')
        self.get_logger().info('Socket Bridge node starting')

        self.reentrant_group = ReentrantCallbackGroup()

        # Declare parameters
        self.declare_parameter('image_save_path', '/root/host_home/ros2_ws/data/images')
        self.declare_parameter('socket_host', '0.0.0.0')
        self.declare_parameter('socket_port', 9090)

        # Get parameters
        self.image_save_path = self.get_parameter('image_save_path').get_parameter_value().string_value
        self.socket_host = self.get_parameter('socket_host').get_parameter_value().string_value
        self.socket_port = self.get_parameter('socket_port').get_parameter_value().integer_value

        try:
            if not os.path.exists(self.image_save_path):
                os.makedirs(self.image_save_path)
                self.get_logger().info(f'Created image save path: {self.image_save_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to create image save path: {e}')
            raise

        # Initialize Ros2 service client
        self.trigger_cli = self.create_client(TriggerGrasp, 
                                            'trigger_grasp_pipeline',
                                            callback_group=self.reentrant_group)
        while not self.trigger_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().info('TriggerGrasp service unavailable, waiting...')
        self.get_logger().info('TriggerGrasp client started.')

        # Initialize Socket server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((self.socket_host, self.socket_port))
            self.server_socket.listen(5)
            self.get_logger().info(f'Socket server started, listening on {self.socket_host}:{self.socket_port}')
        except Exception as e:
            self.get_logger().error(f'Socket server failed to bind to {self.socket_host}:{self.socket_port}: {e}')
            raise
        self.socket_thread = threading.Thread(target=self._socket_server_loop)
        self.socket_thread.daemon = True
        self.socket_thread.start()
    
    def _socket_server_loop(self):
        """
        Socket server main loop, accept client connections and handle image reception and service calls
        """
        # Infinite loop to accept client connections
        # As long as this ROS2 node is running (rclpy.ok()), continue accepting connections
        while rclpy.ok():
            try:
                conn, addr = self.server_socket.accept()
                self.get_logger().info(f'Accepted connection from {addr}')
                # Start a new thread to handle this connection, so original thread can continue accepting other connections
                client_thread = threading.Thread(target=self._handle_client_connection, args=(conn, addr))
                client_thread.daemon = True
                client_thread.start()
            except OSError as e:
                if rclpy.ok():
                    self.get_logger().error(f'Socket accept connection failed: {e}')
                else:
                    pass
        self.get_logger().info('Socket listening thread is closing...')
        self.server_socket.close()

    def _handle_client_connection(self, conn, addr):
        """
        Handle single client connection
        1. (Socket) Receive request (JSON: text_prompt, mode; Image data: RGB+Depth)
        2. Create unique directory to save images
        3. Call TriggerGrasp service
        4. Send response to client
        5. Close connection
        """
        self.get_logger().info(f'[Client{addr}] Handling thread started')
        try:
            # --- Create unique request directory ---
            # request_dir_name = f'request_{uuid.uuid4().hex}'
            # request_dir_path = os.path.join(self.image_save_path, request_dir_name)
            request_dir_path = self.image_save_path
            try:
                os.makedirs(request_dir_path, exist_ok=True)
                self.get_logger().info(f'[Client{addr}] Created request directory: {request_dir_path}')
            except Exception as e:
                self.get_logger().error(f'[Client{addr}] Failed to create request directory: {e}')
                return
            
            # --- Receive JSON and image data ---
            # 1. Receive JSON request
            self.get_logger().info(f'[Client{addr}] Waiting to receive JSON data...')
            request_json_bytes = recv_msg(conn)
            if request_json_bytes is None:
                self.get_logger().error(f'[Client{addr}] Did not receive request data, connection may be closed')
                return
            # 2. Receive RGB image
            self.get_logger().info(f'[Client{addr}] Waiting to receive RGB image data...')
            rgb_image_bytes = recv_msg(conn)
            if rgb_image_bytes is None:
                self.get_logger().error(f'[Client{addr}] Did not receive RGB image data, connection may be closed')
                return
            # 3. Receive Depth image
            self.get_logger().info(f'[Client{addr}] Waiting to receive Depth image data...')
            depth_image_bytes = recv_msg(conn)
            if depth_image_bytes is None:
                self.get_logger().error(f'[Client{addr}] Did not receive Depth image data, connection may be closed')
                return
            
            # --- Process and save data ---
            # 1. Parse JSON
            try:
                request_data = json.loads(request_json_bytes.decode('utf-8'))
                text_prompt = request_data.get('text_prompt', '')
                mode = request_data.get('mode', 0)
                self.get_logger().info(f'[Client{addr}] JSON data parsed successfully: text_prompt="{text_prompt}", mode={mode}')
            except Exception as e:
                self.get_logger().error(f'[Client{addr}] Failed to parse JSON data: {e}')
                return
            # 2. Save RGB image
            try:
                rgb_array = np.frombuffer(rgb_image_bytes, dtype=np.uint8)
                rgb_image = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
                if rgb_image is None:
                    raise ValueError("Failed to decode RGB image")
                rgb_image_path = os.path.join(request_dir_path, 'color.png')
                cv2.imwrite(rgb_image_path, rgb_image)
                self.get_logger().info(f'[Client{addr}] RGB image saved successfully: {rgb_image_path}')
            except Exception as e:
                self.get_logger().error(f'[Client{addr}] Failed to save RGB image: {e}')
                return
            # 3. Save Depth image
            try:
                depth_array = np.frombuffer(depth_image_bytes, dtype=np.uint8)
                depth_image = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
                if depth_image is None:
                    raise ValueError("Failed to decode Depth image")
                depth_image_path = os.path.join(request_dir_path, 'depth.png')
                cv2.imwrite(depth_image_path, depth_image)
                self.get_logger().info(f'[Client{addr}] Depth image saved successfully: {depth_image_path}')
            except Exception as e:
                self.get_logger().error(f'[Client{addr}] Failed to save Depth image: {e}')
                return
            
            # --- Call TriggerGrasp service ---
            self.get_logger().info(f'[Client{addr}] Preparing to call TriggerGrasp service...')
            trigger_request = TriggerGrasp.Request()
            trigger_request.input_path = request_dir_path
            trigger_request.text_prompt = text_prompt
            trigger_request.mode = mode

            try:
                ros_response = self.trigger_cli.call(trigger_request)
                self.get_logger().info(f'[Client{addr}] Service response: success={ros_response.success}, message="{ros_response.message}"')
            except Exception as e:
                self.get_logger().error(f'[Client{addr}] Failed to call TriggerGrasp service: {e}')
                response_json_dict = {
                    'success': False,
                    'message': f'Service call failed: {str(e)}'
                }
                response_bytes = json.dumps(response_json_dict).encode('utf-8')
                send_msg(conn, response_bytes)
                return
            # ros_future = self.trigger_cli.call_async(trigger_request)
            # while rclpy.ok() and not ros_future.done():
            #     time.sleep(0.1)
            #     print(ros_future.done())
            # if not ros_future.done():
            #     self.get_logger().error(f'[Client{addr}] ROS2 was closed, TriggerGrasp service call timeout or failed')
            #     return
            # ros_response = ros_future.result()
            # self.get_logger().info(f'[Client{addr}] Service response: success={ros_response.success}, message="{ros_response.message}"')

            # --- Build and send Socket response ---
            response_json_dict = {}
            if ros_response.success:
                p = ros_response.grasp_poses[0].position
                o = ros_response.grasp_poses[0].orientation
                
                pos = [p.x, p.y, p.z]
                quat = [o.x, o.y, o.z, o.w]
                # Build successful JSON dict
                response_json_dict = {
                    'success': True,
                    'message': ros_response.message,
                    'grasp_poses': {
                        'position': pos,
                        'orientation': quat
                    }
                }
            else:
                # Build failed JSON dict
                response_json_dict = {
                    'success': False,
                    'message': ros_response.message
                }
            response_bytes = json.dumps(response_json_dict).encode('utf-8')
            self.get_logger().info(f'[Client{addr}] Sending response data...')
            send_msg(conn, response_bytes)
            self.get_logger().info(f'[Client{addr}] Response data sent complete')

        except (ConnectionAbortedError, BrokenPipeError, OSError) as e:
            self.get_logger().warn(f'[Client{addr}] Exception occurred while handling connection: {e}')
        
        except Exception as e:
            self.get_logger().error(f'[Client{addr}] Unexpected error occurred while handling connection: {e}')
        
        finally:
            self.get_logger().info(f'[Client{addr}] Closing connection')
            conn.close()
            self.get_logger().info(f'[Client{addr}] Handling thread ended')
    
def main(args=None):
    rclpy.init(args=args)
    socket_bridge_node = None
    executor = None
    try:
        socket_bridge_node = SocketBridgeNode()
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
        executor.add_node(socket_bridge_node)
        
        try:
            socket_bridge_node.get_logger().info('SocketBridgeNode started and beginning to spin (multi-threaded)...')
            # Use multi-threaded executor to "spin"
            executor.spin()
        finally:
            # (This finally ensures executor is closed before node shutdown)
            socket_bridge_node.get_logger().info('Shutting down Executor...')
            executor.shutdown()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error occurred during node runtime: {e}')
    finally:
        socket_bridge_node.get_logger().info('Shutting down Socket Bridge node...')
        socket_bridge_node.destroy_node()
        rclpy.shutdown()
    
if __name__ == '__main__':
    main()
        
            
