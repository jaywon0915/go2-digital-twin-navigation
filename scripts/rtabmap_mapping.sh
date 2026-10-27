#!/bin/bash
# RTAB-Map in mapping mode: builds a new 3D map from the simulated RGB-D camera.
# Needs ./scripts/play.sh running. Drive loops (not just straight lines) so
# RTAB-Map can close loops. Before stopping, run ./scripts/save_2d_map.sh in
# another terminal: the 2D map can only be saved while this is still running.
#
# ⚠️ Wipes $MAP_DIR/rtabmap.db on start. Run ./scripts/backup_maps.sh first to keep it.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
use_system_ros2_env
stop_background_jobs_on_exit
mkdir -p "$MAP_DIR"

echo "🗺️ Starting RTAB-Map mapping (database: $MAP_DIR/rtabmap.db, previous map is deleted)..."

echo "🔗 Publishing static TF odom -> World..."
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom World &

# Topic names match ros2_sensor_setup.py
ros2 launch rtabmap_launch rtabmap.launch.py \
    rtabmap_args:="--delete_db_on_start" \
    database_path:="$MAP_DIR/rtabmap.db" \
    rgb_topic:=/go2_camera/rgb/image_raw \
    depth_topic:=/go2_camera/depth/image_raw \
    camera_info_topic:=/go2_camera/rgb/camera_info \
    odom_topic:=/odom \
    odom_frame_id:=odom \
    visual_odometry:=false \
    frame_id:=base \
    approx_sync:=true \
    use_sim_time:=true
