#!/bin/bash
# Same as rtabmap_localization.sh, plus robot_state_publisher with the Go2 URDF
# (so RViz shows the robot model) and a taller depth band for the laser scan.
# Set GO2_URDF if your go2_description URDF is somewhere else.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
use_system_ros2_env
require_file "$MAP_DIR/rtabmap.yaml" "Place the map as described in assets/README.md, or set MAP_DIR."
require_file "$MAP_DIR/rtabmap.db" "Place the map as described in assets/README.md, or set MAP_DIR."
stop_background_jobs_on_exit

echo "📍 1. Starting map_server with $MAP_DIR/rtabmap.yaml..."
ros2 run nav2_map_server map_server \
    --ros-args -p yaml_filename:="$MAP_DIR/rtabmap.yaml" -p use_sim_time:=true &
sleep 2

ros2 run nav2_lifecycle_manager lifecycle_manager \
    --ros-args -p autostart:=true -p node_names:='["map_server"]' \
    -p bond_timeout:=0.0 -p attempt_respawn_reconnection:=true -p use_sim_time:=true &
sleep 2

echo "🤖 1.5 Starting robot_state_publisher (URDF)..."
if [ -f "$GO2_URDF" ]; then
    ros2 run robot_state_publisher robot_state_publisher "$GO2_URDF" --ros-args -p use_sim_time:=true &
    sleep 2
else
    echo "⚠️ URDF not found, skipping robot model: $GO2_URDF"
fi

echo "🔗 Publishing static TF odom -> World..."
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom World &

echo "🔦 2. Converting depth images to a 2D laser scan..."
ros2 run depthimage_to_laserscan depthimage_to_laserscan_node \
    --ros-args \
    -r depth:=/go2_camera/depth/image_raw \
    -r depth_camera_info:=/go2_camera/depth/camera_info \
    -r scan:=/scan \
    -p output_frame:=base \
    -p use_sim_time:=true \
    -p scan_height:=50 \
    -p range_min:=0.2 \
    -p range_max:=5.0 &

echo "📍 3. Starting RTAB-Map in localization mode..."
ros2 launch rtabmap_launch rtabmap.launch.py \
    localization:=true \
    database_path:="$MAP_DIR/rtabmap.db" \
    rgb_topic:=/go2_camera/rgb/image_raw \
    depth_topic:=/go2_camera/depth/image_raw \
    camera_info_topic:=/go2_camera/rgb/camera_info \
    odom_topic:=/odom \
    odom_frame_id:=odom \
    visual_odometry:=false \
    frame_id:=base \
    grid_map_topic:=/rtabmap/map \
    approx_sync:=true \
    use_sim_time:=true &

sleep 5

echo "🚗 4. Starting the Nav2 stack ($NAV2_PARAMS)..."
ros2 launch nav2_bringup navigation_launch.py \
    use_sim_time:=True \
    autostart:=True \
    params_file:="$NAV2_PARAMS" &

sleep 3

echo "🖥️ 5. Starting RViz2..."
ros2 launch nav2_bringup rviz_launch.py \
    use_sim_time:=True

wait
