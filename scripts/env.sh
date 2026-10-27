# Shared settings for every launcher in scripts/. Sourced, not executed.
#
# Every path has a default that works with the layout described in README.md.
# To change one, export it before running a script, e.g.
#   export ISAACLAB_DIR=/opt/IsaacLab
#   ./scripts/play.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Isaac Lab checkout (CGV-HGU/isaac-go2-ros2, branch simul-trio) with this
# repo's overrides installed by ./install.sh
: "${ISAACLAB_DIR:=$HOME/IsaacLab}"
# Isaac Sim 5.1 install. Isaac Lab's standard setup links it at _isaac_sim.
: "${ISAAC_SIM_ROOT:=$ISAACLAB_DIR/_isaac_sim}"

# Lab assets (not in git, see assets/README.md)
: "${ASSETS_DIR:=$REPO_ROOT/assets}"
: "${GO2_SCENE_USD:=$ASSETS_DIR/scene/05_08_real(original).usd}"
: "${MAP_DIR:=$ASSETS_DIR/map}"
: "${MAP_BACKUP_DIR:=$ASSETS_DIR/map_backups}"
: "${SKRL_CHECKPOINT:=$ASSETS_DIR/checkpoints/skrl_unitree_go2_flat_2026-04-13_12-51-57_ppo_torch/best_agent.pt}"
: "${RSL_RL_CHECKPOINT:=$ASSETS_DIR/checkpoints/rsl_rl_unitree_go2_flat_2026-04-02_17-36-56/model_1999.pt}"

: "${NAV2_PARAMS:=$REPO_ROOT/config/nav2_params.yaml}"
# Only used by rtabmap_localization_urdf.sh
: "${GO2_URDF:=$ISAACLAB_DIR/../go2-ros2/robots/go2_description/urdf/go2_description.urdf}"
# RTAB-Map publishes its occupancy grid under its own /rtabmap namespace
: "${MAP_TOPIC:=/rtabmap/map}"

# flat_env_cfg.py reads the scene path from the environment
export GO2_SCENE_USD

require_file() {
    if [ ! -f "$1" ]; then
        echo "❌ Not found: $1"
        echo "   $2"
        exit 1
    fi
}

# For scripts that launch Isaac Sim.
# Isaac Sim's bundled Python is 3.11, but system ROS 2 Jazzy (/opt/ros/jazzy) is
# built for 3.12. If the system ROS env leaks in, rclpy fails to import
# (_rclpy_pybind11) or aborts on a typesupport mismatch. Drop the system ROS
# vars and use only Isaac Sim's bundled Jazzy bridge.
use_isaac_sim_ros2_env() {
    require_file "$ISAACLAB_DIR/isaaclab.sh" "Set ISAACLAB_DIR to your Isaac Lab checkout (see README.md)."
    local bridge="$ISAAC_SIM_ROOT/exts/isaacsim.ros2.bridge/jazzy"
    if [ ! -d "$bridge" ]; then
        echo "❌ Isaac Sim ROS 2 bridge not found: $bridge"
        echo "   Set ISAAC_SIM_ROOT to your Isaac Sim 5.1 install."
        exit 1
    fi
    unset AMENT_PREFIX_PATH ROS_DISTRO COLCON_PREFIX_PATH ROS_VERSION ROS_PYTHON_VERSION
    export PYTHONPATH="$bridge/rclpy"
    # Remove only the /opt/ros/jazzy entries (keep CUDA etc.) and put the bridge libs first
    local clean_ld
    clean_ld=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/opt/ros/jazzy' | paste -sd ':' - || true)
    export LD_LIBRARY_PATH="$bridge/lib${clean_ld:+:$clean_ld}"
    export RMW_IMPLEMENTATION="rmw_fastrtps_cpp"
}

# For scripts that run system ROS 2 tools (RTAB-Map, Nav2, RViz).
use_system_ros2_env() {
    if [ -z "${ROS_DISTRO:-}" ] && [ -f /opt/ros/jazzy/setup.bash ]; then
        source /opt/ros/jazzy/setup.bash
    fi
    if ! command -v ros2 > /dev/null; then
        echo "❌ ros2 not found. Install ROS 2 Jazzy and source /opt/ros/jazzy/setup.bash."
        exit 1
    fi
}

# Background ROS processes ignore Ctrl+C in a script, so stop them on exit
# (otherwise a stray static TF / map_server keeps running into the next session)
stop_background_jobs_on_exit() {
    trap 'kill $(jobs -p) 2> /dev/null' EXIT
}
