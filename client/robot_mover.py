#!/usr/bin/env python
import sys
import time
import numpy as np
from robots import R1Robot

def cubic_polynomial(t, T):
    """Cubic polynomial trajectory planning"""
    tau = t / T
    s = 3 * (tau ** 2) - 2 * (tau ** 3)
    return s

def main():
    if len(sys.argv) != 4:
        print("Usage: python robot_mover.py x y z")
        return
    
    try:
        # Parse target position
        target_position = [
            float(sys.argv[1]),
            float(sys.argv[2]), 
            float(sys.argv[3])
        ]
        
        # Prevent collision with robotic arm
        target_position[0] -= 0.1
        target_position[2] += 0.13
        
        
        # Initialize robot (in new ROS node)
        robot = R1Robot('r1')
        print("Connected to robot successfully")
        
        # Adjust target position
        first_position = target_position[:]

        first_position[0] = first_position[0] - 0.4
        print(f"First position: {first_position}")
        # print(target_position)
        target_orientation = [0.7, 0, 0, 0.7]
        first_pose = np.concatenate((first_position, target_orientation))
        robot.set_endpose(first_pose)

        target_pose = np.concatenate((target_position, target_orientation))
        print("Target Pose: ", target_pose)
        
        # Get start pose
        start_pose = robot.read_current_pose()
        print(f"Start pose: {start_pose}")
        
        # Trajectory execution code (same as before)
        duration = 2.0
        steps = 10
        interval = duration / steps
        start_time = time.time()

        print("Starting smooth trajectory...")
        
        for i in range(steps + 1):
            current_time = i * interval
            fraction = cubic_polynomial(current_time, duration)

            interpolated_position = [
                start_pose[0] + fraction * (target_position[0] - start_pose[0]),
                start_pose[1] + fraction * (target_position[1] - start_pose[1]),
                start_pose[2] + fraction * (target_position[2] - start_pose[2])
            ]
            interpolated_orientation = [0.7, 0, 0, 0.7]

            interpolated_pose = np.concatenate((interpolated_position, interpolated_orientation))
            robot.set_endpose_quick(interpolated_pose)

            expected_time = start_time + (i + 1) * interval
            sleep_time = expected_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        robot.set_endpose_quick(target_pose)
        print("Trajectory completed successfully")
        
    except Exception as e:
        print(f"Error in robot movement: {str(e)}")

if __name__ == "__main__":
    main()