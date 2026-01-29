# RoboCup@Home Manipulation System

A ROS 2-based robotic manipulation system integrating vision-based object detection, grasp pose estimation, and multi-robot control for RoboCup@Home competitions.

## 🌟 Features

- **Multi-Modal Object Detection**: Integration of Grounded-SAM-2 and GraspNet for robust object segmentation and grasp pose prediction
- **Multi-Robot Support**: Unified interface supporting R1 humanoid robot, Kinova arm, Nvidia platforms, and more
- **Distributed Architecture**: Client-server design with socket-based communication for flexible deployment
- **Vision-Guided Grasping**: Complete pipeline from RGB-D perception to grasp execution
- **Trajectory Planning**: Smooth motion control with cubic polynomial trajectory planning

## 📋 Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## 🏗️ Architecture

The system follows a distributed client-server architecture:

```
┌─────────────┐         Socket          ┌──────────────────┐
│   Client    │◄─────────────────────►  │  Server (ROS2)   │
│             │                          │                  │
│ - Robot     │                          │ - Socket Bridge  │
│   Control   │                          │ - Grasp Pipeline │
│ - Image     │                          │ - GraspNet       │
│   Capture   │                          │ - Grounded-SAM   │
└─────────────┘                          └──────────────────┘
```

### Server Components (ROS 2)

- **Socket Bridge Node**: Handles socket communication and image reception
- **Grasp Pipeline Node**: Orchestrates the complete grasping workflow
- **GraspNet Service**: Predicts 6-DOF grasp poses from point clouds
- **Grounded-SAM Service**: Performs text-prompted object segmentation

### Client Components

- **Robot Controllers**: Hardware-specific interfaces for different robotic platforms
- **Socket Client**: Communicates with server for grasp pose requests
- **Image Acquisition**: Captures RGB-D images from cameras
- **Trajectory Planner**: Generates smooth motion trajectories

## 🔧 Prerequisites

### Hardware Requirements
- Robotic manipulator (R1, Kinova, or compatible)
- RGB-D camera (RealSense or compatible)
- CUDA-capable GPU (recommended for inference)

### Software Requirements
- Ubuntu 20.04 / 22.04
- ROS 2 (Foxy/Humble) - **Server only**
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration) - **Server only**

### Python Dependencies

**Client:**
```
numpy
opencv-python
scipy
pandas
```

**Server:**
```
rclpy
torch
torchvision
open3d
opencv-python
numpy
scipy
graspnetAPI
supervision
hydra-core
omegaconf
pycocotools
Pillow
```

## 📦 Installation

### 1. Clone the Repository
```bash
cd ~/ros2_ws/src
git clone https://github.com/USTC-WrightEagle-AI/Manipulation.git
cd manipulation
```

### 2. Install Dependencies

#### ROS 2 Dependencies (Server Only)
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

#### Python Dependencies

**For Client:**
```bash
cd ~/ros2_ws/src/manipulation/client
pip install -r requirements.txt
```

**For Server:**
```bash
cd ~/ros2_ws/src/manipulation/server
pip install -r requirements.txt
```

### 3. Install GraspNet-Baseline
```bash
cd ~/ros2_ws
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
```
Then, follow the official guidance of [graspnet-baseline](https://github.com/graspnet/graspnet-baseline)

### 4. Install Grounded-SAM-2
```bash
cd ~/ros2_ws
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
```
Then, follow the official guidance of [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)

### 5. Build the Workspace
```bash
cd ~/ros2_ws
colcon build --packages-select grasp_srv_interface grasp_py graspnet_py grounded_sam_py socket_node
source install/setup.bash
```

## 🚀 Usage

### Starting the Server

#### Terminal 1: Launch Socket Bridge
```bash
source ~/ros2_ws/install/setup.bash
ros2 run socket_node socket_server
```

#### Terminal 2: Launch Grasp Pipeline
```bash
source ~/ros2_ws/install/setup.bash
ros2 run grasp_py grasp_client
```

#### Terminal 3: Launch GraspNet Service
```bash
source ~/ros2_ws/install/setup.bash
ros2 run graspnet_py graspnet_server
```

#### Terminal 4: Launch Grounded-SAM Service
```bash
source ~/ros2_ws/install/setup.bash
ros2 run grounded_sam_py grounded_sam_server
```

### Running the Client

#### Basic Grasping Example
```bash
cd manipulation/client
python socket_client.py \
    --server_host 192.168.31.44 \
    --server_port 9090 \
    --rgb_path ./images/color.png \
    --depth_path ./images/depth.png \
    --text_prompt "bottle" \
    --mode 0
```

#### Capture Images from Camera
```bash
python get_image.py
```

#### Robot Motion Control
```bash
# Smooth trajectory execution
python smooth_move.py

# Return to initial position
python smooth_move_back.py
```

## ⚙️ Configuration

### Camera Calibration

Edit `client/images/camera.json`:
```json
{
    "camera_matrix": [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ],
    "width": 1280,
    "height": 720,
    "factor_depth": 1000.0
}
```

### Hand-Eye Calibration

Update `client/cam2end_H.csv` with your camera-to-end-effector transformation matrix.

### Server Parameters

Edit ROS 2 launch parameters or use command-line arguments:

**GraspNet Server**
- `checkpoint_path`: Path to GraspNet model checkpoint
- `num_point`: Number of points to sample (default: 20000)
- `collision_thresh`: Collision detection threshold (default: 0.01)
- `voxel_size`: Voxel size for collision detection (default: 0.01)

**Socket Server**
- `socket_host`: Server IP address (default: 0.0.0.0)
- `socket_port`: Server port (default: 9090)
- `image_save_path`: Directory for saving images

## 📝 License

This project is developed for RoboCup@Home competitions. Please check with the original repository for licensing information.

## 🙏 Acknowledgments

- [GraspNet](https://github.com/graspnet/graspnet-baseline) for grasp pose detection
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) for object segmentation
- [ROS 2](https://docs.ros.org/) for robotic middleware
- [WrightEagle AI Team](https://wrighteagleai.homes/) for system integration

## 📧 Contact

For questions and support, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ for RoboCup@Home**
