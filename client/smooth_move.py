import rospy
import subprocess
import os
from geometry_msgs.msg import PointStamped

def single_message_handler():
    """Receive one message only then exit"""
    try:
        rospy.init_node('smooth_place_single')
        rospy.loginfo("Waiting for single message on /placement/base_link_3d_position...")
        
        # 等待并接收一条消息
        msg = rospy.wait_for_message('/placement/base_link_3d_position', PointStamped, timeout=30.0)
        
        rospy.loginfo("Received placement position message")
        
        # Extract target position from message
        target_position = [msg.point.x, msg.point.y, msg.point.z]
        
        # Call external Python script to execute robot movement
        script_path = os.path.join(os.path.dirname(__file__), 'robot_mover.py')
        cmd = [
            'python', script_path, 
            str(target_position[0]), 
            str(target_position[1]), 
            str(target_position[2])
        ]
        
        # Execute robot movement script (wait for completion)
        subprocess.call(cmd)
        rospy.loginfo("Robot movement completed")
        
    except rospy.ROSException:
        rospy.logerr("Timeout: No message received within 30 seconds")
    except Exception as e:
        rospy.logerr(f"Error: {str(e)}")

if __name__ == "__main__":
    single_message_handler()
'''
rostopic pub -r 10 /motion_target/target_pose_arm_right geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:                    
    secs: 0
    nsecs: 0
  frame_id: ''
pose:
  position:                                                                          
    x: 0
    y: 0
    z: 0.35
  orientation:
    x: 0.7
    y: 0
    z: 0
    w: 0.7" 
'''