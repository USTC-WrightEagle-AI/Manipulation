#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import threading
import argparse
import time

import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState

class GripperSmoothController(object):
    def __init__(
        self,
        target_pos,
        step_size=2.0,
        control_rate=1,
        tolerance=30,
        state_topic="/hdas/feedback_gripper_right",
        command_topic="/motion_control/position_control_gripper_right",
        joint_name="right_gripper",
    ):
        """
        :param target_pos: Target position
        :param step_size: Distance to move per step
        :param control_rate: Control frequency Hz
        :param tolerance: Error threshold for reaching target
        :param state_topic: JointState topic for current gripper position
        :param command_topic: Float32 topic for controlling gripper
        :param joint_name: Joint name corresponding to gripper in JointState
        """
        rospy.init_node("gripper_slow_move_once", anonymous=True)

        self.target_pos = float(target_pos)
        self.step_size = abs(float(step_size))
        self.tolerance = abs(float(tolerance))
        self.rate = rospy.Rate(control_rate)

        self.state_topic = state_topic
        self.command_topic = command_topic
        self.joint_name = joint_name

        # Current gripper position, updated by JointState callback
        self.current_pos = None
        self.state_lock = threading.Lock()

        # Parameters for detecting "stop" in feedback
        self.last_pos = None
        self.stable_count = 0
        self.stable_threshold = 0.5  # Minimum change considered as "no movement"
        self.stable_max_count = int(control_rate * 2)  # Considered stopped if no movement for about 2 secondsped if no movement for about 2 seconds

        # Publisher: control gripper position
        self.cmd_pub = rospy.Publisher(self.command_topic, Float32, queue_size=10)

        # Subscriber: read actual gripper position
        self.state_sub = rospy.Subscriber(self.state_topic, JointState, self.state_callback)

        rospy.loginfo("Starting gripper smooth control node")
        rospy.loginfo(
            f"target_pos: {self.target_pos:.3f}, step_size: {self.step_size:.3f}, "
            f"rate: {control_rate:.1f} Hz, tolerance: {self.tolerance:.3f}"
        )
        rospy.loginfo(
            f"state_topic: {self.state_topic}, command_topic: {self.command_topic}, joint_name: {self.joint_name}"
        )

    def state_callback(self, msg):
        """
        JointState callback, extract position corresponding to joint_name.
        """
        try:
            if self.joint_name in msg.name:
                idx = msg.name.index(self.joint_name)
                pos = msg.position[idx]
                with self.state_lock:
                    self.current_pos = pos
        except Exception as e:
            rospy.logwarn(f"Failed to parse JointState: {e}")

    def wait_for_first_state(self, timeout=5.0):
        """
        Wait for the first gripper state message.
        """
        start_time = rospy.Time.now().to_sec()
        rospy.loginfo("Waiting for gripper current state message...")
        while not rospy.is_shutdown():
            with self.state_lock:
                if self.current_pos is not None:
                    rospy.loginfo(f"Received initial gripper position: {self.current_pos:.3f}")
                    # Initialize last_pos to avoid triggering "stop" detection at the start
                    self.last_pos = self.current_pos
                    return True

            if rospy.Time.now().to_sec() - start_time > timeout:
                rospy.logerr("Did not receive gripper state message within specified time, exiting.")
                return False

            rospy.sleep(0.1)

    def control_loop(self):
        """
        Control loop: smoothly approach target position based on current actual position.
        Ends when current value and target value error is less than tolerance.
        If feedback value remains almost unchanged for a long time, send final target value and end.
        """
        if not self.wait_for_first_state():
            return

        while not rospy.is_shutdown():
            with self.state_lock:
                current = self.current_pos

            if current is None:
                rospy.logwarn("Current gripper position unknown, waiting for next frame state...")
                self.rate.sleep()
                continue

            error = self.target_pos - current

            # Arrival judgment: error within tolerance range, directly end
            if abs(error) <= self.tolerance:
                rospy.loginfo(
                    f"Reached near target: current={current:.3f}, "
                    f"target={self.target_pos:.3f}, error={error:.3f}, tolerance={self.tolerance:.3f}"
                )
                break

            # Check if feedback has "stopped"
            if self.last_pos is not None:
                delta = abs(current - self.last_pos)
                if delta < self.stable_threshold:
                    self.stable_count += 1
                else:
                    self.stable_count = 0

                if self.stable_count >= self.stable_max_count:
                    rospy.logwarn(
                        f"Detected feedback basically stopped (consecutive {self.stable_count} steps Δ={delta:.5f}), "
                        f"directly sending final target value {self.target_pos:.3f} and exiting control loop."
                    )
                    self.cmd_pub.publish(Float32(self.target_pos))
                    time.sleep(1)
                    break

            self.last_pos = current

            # Approach target by step size
            direction = 1.0 if error > 0.0 else -1.0
            next_pos = current + direction * self.step_size

            # Prevent overshoot in wrong direction: if next step already passes target, directly send target value
            if (direction > 0.0 and next_pos > self.target_pos) or (
                direction < 0.0 and next_pos < self.target_pos
            ):
                next_pos = int(self.target_pos)

            # Publish control command
            self.cmd_pub.publish(Float32(next_pos))
            rospy.loginfo(
                f"current: {current:.3f} -> sending target: {next_pos:.3f} (overall target: {self.target_pos:.3f})"
            )

            self.rate.sleep()

        rospy.loginfo("Control loop ended, script will exit.")

def parse_args():
    """
    Command line arguments:
      python slow.py 0
      python slow.py 50 --step 1.0 --rate 20 --tol 5.0
    """
    parser = argparse.ArgumentParser(
        description="Control robotic arm gripper to smoothly move to target position (unit consistent with JointState)."
    )
    parser.add_argument(
        "target", type=float, help="Target position (unit consistent with JointState.position)"
    )
    parser.add_argument(
        "--step", type=float, default=6, help="Distance to move per step, default 5.0"
    )
    parser.add_argument(
        "--rate", type=float, default=5, help="Control loop frequency Hz, default 10.0"
    )
    parser.add_argument(
        "--tol", type=float, default=5, help="Error threshold for reaching target position, default 15.0"
    )
    parser.add_argument(
        "--state_topic",
        type=str,
        default="/hdas/feedback_gripper_right",
        help="Gripper JointState topic name, default /hdas/feedback_gripper_right",
    )
    parser.add_argument(
        "--command_topic",
        type=str,
        default="/motion_control/position_control_gripper_right",
        help="Float32 topic name for controlling gripper, default /motion_control/position_control_gripper_right",
    )
    parser.add_argument(
        "--joint_name",
        type=str,
        default="right_gripper",
        help="Joint name corresponding to gripper in JointState, default right_gripper",
    )

    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

if __name__ == "__main__":
    args = parse_args()

    try:
        controller = GripperSmoothController(
            target_pos=args.target,
            step_size=args.step,
            control_rate=args.rate,
            tolerance=args.tol,
            state_topic=args.state_topic,
            command_topic=args.command_topic,
            joint_name=args.joint_name,
        )

        controller.control_loop()
    except rospy.ROSInterruptException:
        pass