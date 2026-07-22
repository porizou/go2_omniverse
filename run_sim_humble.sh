#!/usr/bin/env bash
set -euo pipefail

# Isaac Sim launcher aligned with run_sim.sh, but using bundled ROS 2 Humble.
#
# This keeps the same local layout and activation flow as run_sim.sh while
# switching the bundled ROS payload from Jazzy to Humble so the simulator can
# interoperate with Humble ROS 2 nodes without cross-distro serialization issues.

export ISAAC_VENV="${ISAAC_VENV:-$HOME/isaacsim/env_isaaclab}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/isaacsim/IsaacLab}"
export OMNI_KIT_ACCEPT_EULA=YES
export LIVESTREAM=2
export ENABLE_CAMERAS=1
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$ISAAC_VENV/bin/activate"

# Isaac Sim 5.x / 6.x keep bundled ROS 2 payloads under one of these ext dirs.
ISAAC_ROS2_EXT="$(python -c "import isaacsim, os; b=os.path.dirname(isaacsim.__file__); d=os.environ['ROS_DISTRO']; print(next(os.path.join(b, 'exts', e) for e in ('isaacsim.ros2.core', 'isaacsim.ros2.bridge') if os.path.isdir(os.path.join(b, 'exts', e, d, 'lib'))))")"
BUNDLED_LIB="$ISAAC_ROS2_EXT/$ROS_DISTRO/lib"
BUNDLED_RCLPY="$ISAAC_ROS2_EXT/$ROS_DISTRO/rclpy"

if [[ ! -d "$BUNDLED_LIB" ]]; then
    echo "[run_sim_humble] bundled ROS 2 $ROS_DISTRO libs not found at $BUNDLED_LIB" >&2
    exit 1
fi

export PYTHONPATH="$BUNDLED_RCLPY${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$BUNDLED_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$SCRIPT_DIR"
exec python -u main.py --robot_amount 1 --robot go2 --terrain flat "$@"
