import time
import numpy as np
from robots import R1Robot

def smooth_return_to_start(start_pose, duration=5.0, steps=10):
    """
    Smoothly return from current position to initial position, keeping orientation unchanged
    
    Args:
        start_pose: Initial position coordinates [x, y, z, qx, qy, qz, qw]
        duration: Total time for smooth motion
        steps: Number of steps for smooth motion
    """
    robot = R1Robot('r1')
    
    # Get current position
    current_pose = robot.read_current_pose()
    
    print(f"Current position: {current_pose[:3]}")
    print(f"Target position (initial position): {start_pose[:3]}")
    print(f"Keep orientation: {start_pose[3:]}")
    
    # Use smoother_move function for smooth return
    smoother_move(start_pose, duration, steps)

def smoother_move(target_pose, duration=2.0, steps=10):
    """
    Perform smooth movement using trajectory planning
    Args:
        target_pose: Target pose [x, y, z, qx, qy, qz, qw]
        duration: Total movement time (seconds)
        steps: Number of decomposed steps
    """
    robot = R1Robot('r1')
    cur_pose = robot.read_current_pose()
    start_position = cur_pose[:3]
    start_position[2] += 0.05
    start_pose = np.concatenate((start_position, target_pose[3:]))
    
    # First lift the arm up a little bit
    robot.set_endpose_quick(start_pose)

    start_time = time.time()
    interval = duration / steps

    for i in range(steps + 1):
        current_time = i * interval
        fraction = cubic_polynomial(current_time, duration)

        # Calculate interpolated pose
        interpolated_position = [
            start_pose[0] + fraction * (target_pose[0] - start_pose[0]),
            start_pose[1] + fraction * (target_pose[1] - start_pose[1]),
            start_pose[2] + fraction * (target_pose[2] - start_pose[2])
        ]
        
        # Keep target orientation unchanged
        interpolated_orientation = target_pose[3:]
        
        print(f"Step {i}/{steps}, Position: {[f'{p:.3f}' for p in interpolated_position]}")
        
        interpolated_pose = np.concatenate((interpolated_position, interpolated_orientation))
        robot.set_endpose_quick(interpolated_pose)

        # Precise time control
        expected_time = start_time + (i + 1) * interval
        sleep_time = expected_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Ensure final arrival at target point
    robot.set_endpose(target_pose)
    print("Smooth return completed")

def cubic_polynomial(t, T):
    """Cubic polynomial trajectory planning, t is current time, T is total time. Returns interpolation coefficient s (0->1)"""
    if T == 0:
        return 1.0
    tau = t / T
    s = 3 * (tau ** 2) - 2 * (tau ** 3)
    return s

# Usage example
if __name__ == "__main__":
    # Assuming you have already saved the initial position
    initial_pose = [0.0, 0.0, 0.3, 0.7, 0, 0, 0.7]  # Please replace with your actual initial position
    
    # Smoothly return to initial position
    smooth_return_to_start(
        start_pose=initial_pose,
        duration=2,
        steps=10
    )