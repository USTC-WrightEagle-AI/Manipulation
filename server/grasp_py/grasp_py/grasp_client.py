import sys
from grasp_srv_interface.srv import Graspnet, GroundedSam, Vggt, Move, TriggerGrasp                                            
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose
import time 

class GraspServerNode(Node):
    def __init__(self):
        super().__init__('grasp_client')
        self.get_logger().info('Grasp client node starting')

        self.reentrant_group = ReentrantCallbackGroup()

        ############################################################
        # --- Internal service clients, responsible for calling graspnet, groundedsam, vggt services ---
        ############################################################

        # Create groundedsam_cli service
        self.gsam_cli = self.create_client(GroundedSam, 'grounded_sam_service', 
                                           callback_group=self.reentrant_group)
        while not self.gsam_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service unavailable, waiting...')
        self.get_logger().info('GroundedSam client started.')

        # Create graspnet_cli service
        self.gspnet_cli = self.create_client(Graspnet, 'graspnet_service', 
                                             callback_group=self.reentrant_group)
        while not self.gspnet_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service unavailable, waiting...')
        self.get_logger().info('Graspnet client started.')arted.')

        # #Create vggt_cli service
        # self.vggt_cli = self.create_client(Vggt, 'vggt_service')
        # while not self.vggt_cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Service unavailable, waiting...')
        # self.get_logger().info('Vggt client started.')

        # #Create move_cli service
        # self.move_cli = self.create_client(Move, 'move')
        # while not self.move_cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('Move service unavailable, waiting...')
        # self.get_logger().info('Move client started.')

        #############################################
        # --- External service server, responsible for receiving commands to start grasp process ---
        #############################################
        self.trigger_service = self.create_service(TriggerGrasp, 
                                                   'trigger_grasp_pipeline', 
                                                   self.trigger_grasp_callback,
                                                   callback_group=self.reentrant_group)
        self.get_logger().info('TriggerGrasp service started, waiting for requests...')

    def call_grounded_sam(self,input_path=None, text_prompt=None, mode=0):
        self.get_logger().info('Preparing to call GroundedSam service...')
        # Create request
        gsam_req = GroundedSam.Request()
        # Set request parameters
        self.get_logger().info('Starting to set request parameters')
        gsam_req.input_path = input_path
        gsam_req.text_prompt = text_prompt
        gsam_req.mode = mode
        self.get_logger().info(f'Request parameters set: input_path={gsam_req.input_path}, text_prompt={gsam_req.text_prompt}')
        self.get_logger().info('Sending request...')
        try:
            response = self.gsam_cli.call(gsam_req)
            self.get_logger().info('Received GroundedSam response')
            return response
        except Exception as e:
            self.get_logger().error(f'Error calling GroundedSam service: {e}')
            return None
    
    def call_vggt(self,input_path=None):
        self.get_logger().info("Preparing to call Vggt service...")
        # Create request
        vggt_req = Vggt.Request()
        # Set request parameters
        self.get_logger().info("Starting to set request parameters")
        vggt_req.input_path = input_path
        self.get_logger().info(f"Request parameters set: input_path={vggt_req.input_path}")
        self.get_logger().info("Sending request...")
        try:
            response = self.vggt_cli.call(vggt_req)
            self.get_logger().info("Received Vggt response")
            return response
        except Exception as e:
            self.get_logger().error(f"Error calling Vggt service: {e}")
            return None
    
    def call_graspnet(self, input_path=None, mode=0):
        self.get_logger().info('Preparing to call Graspnet service...')
        # Create request
        gspnet_req = Graspnet.Request()
        # Set request parameters
        self.get_logger().info('Starting to set request parameters')
        gspnet_req.input_path = input_path
        gspnet_req.mode = mode     
        self.get_logger().info(f'Request parameters set: input_path={gspnet_req.input_path}')
        self.get_logger().info('Sending request...')
        try:
            response = self.gspnet_cli.call(gspnet_req)
            self.get_logger().info('Received Graspnet response')
            return response
        except Exception as e:
            self.get_logger().error(f'Error calling Graspnet service: {e}')
            return None
    
    def call_move(self, target_pose=None):
        self.get_logger().info('Preparing to call Move service...')
        # Create request
        move_req = Move.Request()
        # Set request parameters
        move_req.mode = 0  # Set to GSAM image input mode
        self.get_logger().info('Sending async request...')
        move_req_future = self.move_cli.call_async(move_req)
        self.get_logger().info('Waiting for service response...')
        rclpy.spin_until_future_complete(self, move_req_future)
        self.get_logger().info('Received move response')
        return move_req_future.result()
        
    def send_request(self,input_path=None, text_prompt=None, mode=0):
        # mode = self.mode
        print("------")
        # Call different service combinations based on mode
        # --- mode 0: GroundedSam + Graspnet ---
        # --- mode 1: GroundedSam + Vggt + Graspnet ---

        #########################################
        # --- mode 0: GroundedSam + Graspnet ---
        #########################################
        if mode == 0:
             # Call groundedsam service
            self.get_logger().info(f'Preparing to call GroundedSam service with input_path={input_path}, text_prompt={text_prompt}, mode={mode}')
            gsam_response = self.call_grounded_sam(input_path=input_path,
                                                    text_prompt=text_prompt,
                                                    mode=mode)
            if not gsam_response or not gsam_response.success:
                self.get_logger().error('GroundedSam service call failed or did not execute successfully.')
                return None
            self.get_logger().info('GroundedSam service call successful')

            # Call graspnet service
            self.get_logger().info(f'Preparing to call Graspnet service with input_path={gsam_response.output_path}, mode={mode}')
            gspnet_response = self.call_graspnet(gsam_response.output_path, mode=mode)
            if not gspnet_response:
                self.get_logger().error('Graspnet service call failed or did not execute successfully.')
                return None
            self.get_logger().info('Graspnet service call successful.') 
            return gspnet_response
        
        ###############################################
        # --- mode 1: GroundedSam + Vggt + Graspnet ---
        ###############################################
        elif mode ==1:
            # Call groundedsam service
            self.get_logger().info(f'Preparing to call GroundedSam service with input_path={input_path}, text_prompt={text_prompt}, mode={mode}')
            gsam_response = self.call_grounded_sam(input_path=input_path,
                                                    text_prompt=text_prompt,
                                                    mode=mode)
            if not gsam_response or not gsam_response.success:
                self.get_logger().error('GroundedSam service call failed or did not execute successfully.')
                return None
            self.get_logger().info('GroundedSam service call successful')
            
            # Call vggt service
            self.get_logger().info(f"Preparing to call Vggt service with input_path={gsam_response.output_path}")
            vggt_response = self.call_vggt(gsam_response.output_path)
            if not vggt_response or not vggt_response.success:
                self.get_logger().error('Vggt service call failed or did not execute successfully.')
                return None
            self.get_logger().info('Vggt service call successful.')

            # Call graspnet service
            self.get_logger().info(f'Preparing to call Graspnet service with input_path={vggt_response.output_path}, mode={mode}')
            gspnet_response = self.call_graspnet(input_path=gsam_response.output_path, 
            mode=mode)
            if not gspnet_response:
                self.get_logger().error('Graspnet service call failed or did not execute successfully.')
                return None
            self.get_logger().info('Graspnet service call successful.') 
            return gspnet_response
        
    def trigger_grasp_callback(self, request, response):
        """
        Triggered when 'trigger_grasp_pipeline' service is called
        
        Args:
            request (grasp_srv_interface.srv.TriggerGrasp.Request): 
                Service request, containing parameters to trigger grasp process
                - input_path (str): Input image path
                - text_prompt (str): Text prompt
                - mode (uint8): Grasp mode selection
            

            response (grasp_srv_interface.srv.TriggerGrasp.Response): 
                Service response, containing grasp result
                - success (bool): Whether grasp was successful
                - message (str): Grasp result information
                - grasp_poses (list of Pose): List of grasp poses

        Returns:
            grasp_srv_interface.srv.TriggerGrasp.Response:
            - success (bool): Whether grasp was successful
            - message (str): Grasp result information
            - grasp_poses (list of Pose): List of grasp poses
        """
        self.get_logger().info(f'Received TriggerGrasp service request: path={request.input_path}, '
                               f'prompt={request.text_prompt}, '
                               f'mode={request.mode}, '
                               '\nStarting grasp process...')
        # To be improved, currently using external request address, should be adjusted to use captured image path after integrating move capture
        response_req = self.send_request(input_path=request.input_path,
                                     text_prompt=request.text_prompt,
                                     mode=request.mode)
        # Fill and return response
        # print(response)
        if response_req and response_req.success:
            self.get_logger().info('Vision processing executed successfully, returning pose...')
            response.success = True
            response.message = "Vision Pipeline executed successfully"
            response.grasp_poses = response_req.grasp_poses
            # print(response)
        else:
            self.get_logger().error('Vision processing execution failed, cannot return pose.')
            response.success = False
            response.message = "Vision Pipeline execution failed, no grasp pose available"
        print("Returning response")
        return response
        
def main(args=None):
    rclpy.init(args=args)
    grasp_server_node = GraspServerNode()
    # Create multi-threaded executor
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(grasp_server_node)

    try:
        grasp_server_node.get_logger().info('GraspServerNode is running, waiting for service requests...')
        executor.spin()
    except KeyboardInterrupt:
        grasp_server_node.get_logger().info('GraspServerNode manually closed by user...')
    except Exception as e:
        grasp_server_node.get_logger().error(f'GraspServerNode encountered error during runtime: {e}')
    finally:
        grasp_server_node.get_logger().info('GraspServerNode is closing...')
        executor.shutdown()
        grasp_server_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()