import tf
import tf.transformations as tft
from .base_robot import Robot
import argparse
import numpy as np
import rospy
from sensor_msgs.msg import JointState  # Import JointState message type
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

parser = argparse.ArgumentParser()

class R1Robot(Robot):    
    def __init__(self,name):
        super().__init__(name)
        rospy.init_node('r1_robot', anonymous=True)  # Initialize ROS node
        self.base = "right_arm_base_link"   # Reference coordinate frame for calibration

    def get_tf_transform(self,target_frame, source_frame):
        """
        Get ROS TF coordinate transformation information, return only (x, y, z) and (Roll, Pitch, Yaw) in degrees.
        
        :param target_frame: Target coordinate frame (child frame)
        :param source_frame: Source coordinate frame (parent frame)
        :return: (xyz, rpy_degrees) or None
        """
        listener = tf.TransformListener()

        try:
            # Wait for TF transformation information
            listener.waitForTransform(source_frame, target_frame, rospy.Time(0), rospy.Duration(2.0))

            # Get TF transformation
            (trans, rot) = listener.lookupTransform(source_frame, target_frame, rospy.Time(0))

            # Extract x, y, z
            xyz = trans  # (x, y, z)

            # Convert quaternion to Euler angles (radians)
            rpy = tft.euler_from_quaternion(rot)
            rpy = [r for r in rpy]

            return xyz + rpy  # Return only translation (xyz) and rotation angles (rpy)
    
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF lookup failed: %s -> %s" % (source_frame, target_frame))
            return None

    def check_ready(self):
        return True

    def read_current_pose(self):
        '''
        return: [x,y,z,roll,pitch,yaw]
        Radians
        '''
        target = "right_gripper_link"
        source = self.base
        pose = self.get_tf_transform(target, source)
        # print("**************************************************")
        # print(pose)
        
        return pose
        

    def get_control_angle(self):
        pass

    def set_pose(self,pose):
        '''
        pose: [x1, x2, x3, x4, x5, x6]
        '''
        pub = rospy.Publisher('/motion_target/target_joint_state_arm_right', JointState, queue_size=10)
        rate = rospy.Rate(10)  # 10Hz transmission
        for i in range(10):
            msg = JointState()
            msg.header.seq = 0
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = ''
            msg.name = ['']
            msg.position = pose
            msg.velocity = [0]
            msg.effort = [0]

            # rospy.loginfo("Publishing JointState: %s", msg)
            pub.publish(msg)
            rate.sleep()
    
    def set_endpose(self,pose):
        '''
        endpose: [x,y,z,x,y,z,w]
        '''
        target_pose = PoseStamped()
        # Set Header information
        target_pose.header = Header()
        target_pose.header.seq = 0
        target_pose.header.stamp = rospy.Time.now()  # Current timestamp
        # Set target position
        target_pose.pose.position.x = pose[0]
        target_pose.pose.position.y = pose[1]
        target_pose.pose.position.z = pose[2]

        # Set target attitude (quaternion)
        target_pose.pose.orientation.x = pose[3]
        target_pose.pose.orientation.y = pose[4]
        target_pose.pose.orientation.z = pose[5]
        target_pose.pose.orientation.w = pose[6]

        # Create a ROS publisher, publish to specified topic
        rate = rospy.Rate(10)  # 10Hz transmission
        pub = rospy.Publisher('/motion_target/target_pose_arm_right', PoseStamped, queue_size=10)

        # Publish message
        for i in range(50):
            pub.publish(target_pose)
            rate.sleep()

    def set_endpose_quick(self,pose):
        '''
        endpose: [x,y,z,x,y,z,w]
        '''
        target_pose = PoseStamped()
        # Set Header information
        target_pose.header = Header()
        target_pose.header.seq = 0
        target_pose.header.stamp = rospy.Time.now()  # Current timestamp
        # Set target position
        target_pose.pose.position.x = pose[0]
        target_pose.pose.position.y = pose[1]
        target_pose.pose.position.z = pose[2]

        # Set target attitude (quaternion)
        target_pose.pose.orientation.x = pose[3]
        target_pose.pose.orientation.y = pose[4]
        target_pose.pose.orientation.z = pose[5]
        target_pose.pose.orientation.w = pose[6]

        # Create a ROS publisher, publish to specified topic
        rate = rospy.Rate(10)  # 10Hz transmission
        pub = rospy.Publisher('/motion_target/target_pose_arm_right', PoseStamped, queue_size=10)
        pub.publish(target_pose)
        # Publish message
        # for i in range(3):
        #     pub.publish(target_pose)
        #     rate.sleep()

    def set_gripper(self):
        pass

    def set_angle(self,angle):
        pass
        
    def close(self):
        pass


class R1Robot_left(Robot):    
    def __init__(self,name):
        super().__init__(name)
        rospy.init_node('r1_robot_left', anonymous=True)  # Initialize ROS node
        self.base = "left_arm_base_link"

    def get_tf_transform(self,target_frame, source_frame):
        """
        Get ROS TF coordinate transformation information, return only (x, y, z) and (Roll, Pitch, Yaw) in degrees.
        
        :param target_frame: Target coordinate frame (child frame)
        :param source_frame: Source coordinate frame (parent frame)
        :return: (xyz, rpy_degrees) or None
        """
        listener = tf.TransformListener()

        try:
            # Wait for TF transformation information
            listener.waitForTransform(source_frame, target_frame, rospy.Time(0), rospy.Duration(2.0))

            # Get TF transformation
            (trans, rot) = listener.lookupTransform(source_frame, target_frame, rospy.Time(0))

            # Extract x, y, z
            xyz = trans  # (x, y, z)

            # Convert quaternion to Euler angles (radians)
            rpy = tft.euler_from_quaternion(rot)
            rpy = [r for r in rpy]

            return xyz + rpy  # Return only translation (xyz) and rotation angles (rpy)
    
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF lookup failed: %s -> %s" % (source_frame, target_frame))
            return None

    def check_ready(self):
        return True

    def read_current_pose(self):
        '''
        return: [x,y,z,roll,pitch,yaw]
        Radians
        '''
        target = "left_gripper_link"
        source = self.base
        pose = self.get_tf_transform(target, source)
        print("**************************************************")
        print(pose)
        
        return pose
        

    def get_control_angle(self):
        pass

    def set_pose(self,pose):
        '''
        pose: [x1, x2, x3, x4, x5, x6]
        '''
        pub = rospy.Publisher('/motion_target/target_joint_state_arm_left', JointState, queue_size=10)
        rate = rospy.Rate(10)  # 10Hz transmission
        for i in range(10):
            msg = JointState()
            msg.header.seq = 0
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = ''
            msg.name = ['']
            msg.position = pose
            msg.velocity = [0]
            msg.effort = [0]

            # rospy.loginfo("Publishing JointState: %s", msg)
            pub.publish(msg)
            rate.sleep()

    def set_endpose(self,pose):
        '''
        endpose: [x,y,z,x,y,z,w]
        '''
        target_pose = PoseStamped()
        # Set Header information
        target_pose.header = Header()
        target_pose.header.seq = 0
        target_pose.header.stamp = rospy.Time.now()  # Current timestamp
        # Set target position
        target_pose.pose.position.x = pose[0]
        target_pose.pose.position.y = pose[1]
        target_pose.pose.position.z = pose[2]

        # Set target attitude (quaternion)
        target_pose.pose.orientation.x = pose[3]
        target_pose.pose.orientation.y = pose[4]
        target_pose.pose.orientation.z = pose[5]
        target_pose.pose.orientation.w = pose[6]

        # Create a ROS publisher, publish to specified topic
        rate = rospy.Rate(10)  # 10Hz transmission
        pub = rospy.Publisher('/motion_target/target_pose_arm_left', PoseStamped, queue_size=10)

        # Publish message
        for i in range(10):
            pub.publish(target_pose)
            rate.sleep()
        
    def set_angle(self,angle):
        pass
        
    def close(self):
        pass