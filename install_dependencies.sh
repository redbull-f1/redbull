#!/bin/bash
# RedBull Package Dependencies Installation Script

echo "🚗 Installing dependencies for RedBull package..."

# System packages
echo "📦 Updating system packages..."
sudo apt update

echo "🐍 Installing Python development tools..."
sudo apt install -y python3-pip python3-dev python3-venv

# Check if ROS2 is installed
if ! command -v ros2 &> /dev/null; then
    echo "⚠️  ROS2 not found. Please install ROS2 Humble first:"
    echo "   https://docs.ros.org/en/humble/Installation.html"
    exit 1
fi

echo "✅ ROS2 found!"

# Python packages
echo "📚 Installing Python packages..."

# Core packages
echo "  - Installing NumPy, SciPy, Matplotlib..."
pip3 install numpy>=1.21.0
pip3 install scipy>=1.7.0
pip3 install matplotlib>=3.5.0
pip3 install pandas>=1.3.0
pip3 install scikit-learn>=1.0.0

# PyTorch installation
echo "  - Installing PyTorch..."
echo "    Choose PyTorch version:"
echo "    1) CPU only (recommended for most users)"
echo "    2) CUDA 11.7 (if you have NVIDIA GPU)"
echo "    3) CUDA 11.8 (if you have NVIDIA GPU)"
read -p "    Enter your choice (1-3): " pytorch_choice

case $pytorch_choice in
    1)
        echo "    Installing PyTorch CPU version..."
        pip3 install torch>=1.12.0 torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cpu
        ;;
    2)
        echo "    Installing PyTorch CUDA 11.7 version..."
        pip3 install torch>=1.12.0 torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cu117
        ;;
    3)
        echo "    Installing PyTorch CUDA 11.8 version..."
        pip3 install torch>=1.12.0 torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cu118
        ;;
    *)
        echo "    Invalid choice. Installing CPU version..."
        pip3 install torch>=1.12.0 torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

# Optional packages
echo "  - Installing optional packages..."
pip3 install wandb>=0.12.0

# ROS2 packages (check if they exist)
echo "🤖 Checking ROS2 packages..."
required_packages=(
    "ros-humble-sensor-msgs"
    "ros-humble-geometry-msgs"
    "ros-humble-visualization-msgs"
    "ros-humble-std-msgs"
    "ros-humble-nav-msgs"
    "ros-humble-tf2-ros"
    "ros-humble-launch"
    "ros-humble-launch-ros"
    "python3-colcon-common-extensions"
)for package in "${required_packages[@]}"; do
    if ! dpkg -l | grep -q "^ii  $package "; then
        echo "  - Installing $package..."
        sudo apt install -y $package
    else
        echo "  ✅ $package already installed"
    fi
done

# Create launch directory if it doesn't exist
if [ ! -d "launch" ]; then
    echo "📁 Creating launch directory..."
    mkdir -p launch
fi

# Create config directory if it doesn't exist
if [ ! -d "config" ]; then
    echo "📁 Creating config directory..."
    mkdir -p config
fi

echo ""
echo "✅ Dependencies installed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Build the package:"
echo "   cd ~/ros2_ws"
echo "   colcon build --packages-select redbull"
echo "   source install/setup.bash"
echo ""
echo "2. Train a model (optional):"
echo "   cd ~/ros2_ws/src/redbull/train"
echo "   jupyter notebook train_CenterSpeed_dense.ipynb"
echo ""
echo "3. Run the detector:"
echo "   ros2 run redbull dynamic_vehicle_detector"
echo ""
echo "🎯 Ready to go!"
