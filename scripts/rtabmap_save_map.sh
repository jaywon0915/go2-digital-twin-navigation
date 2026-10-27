#!/bin/bash
# Re-publish an existing $MAP_DIR/rtabmap.db so its 2D map can be saved again,
# without re-driving. Use it when mapping was stopped before the 2D map was saved.
#
# Usage:
#   1) ./scripts/play.sh in another terminal (RTAB-Map needs Isaac Sim's /clock,
#      TF and camera topics; the robot doesn't need to move)
#   2) ./scripts/rtabmap_save_map.sh
#   3) after a few seconds, in a third terminal: ./scripts/save_2d_map.sh
#   4) Ctrl+C this script and play.sh
#
# Same as rtabmap_mapping.sh without --delete_db_on_start. RTAB-Map runs in
# mapping mode here, which appends a new session to whatever database it opens,
# so it works on a temporary copy and $MAP_DIR/rtabmap.db stays unchanged.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
use_system_ros2_env
require_file "$MAP_DIR/rtabmap.db" "Place the map as described in assets/README.md, or set MAP_DIR."

WORK_DIR=$(mktemp -d)
trap 'kill $(jobs -p) 2> /dev/null; rm -rf "$WORK_DIR"' EXIT
cp "$MAP_DIR/rtabmap.db" "$WORK_DIR/rtabmap.db"

echo "🗺️ RTAB-Map: re-publishing the map from a copy of $MAP_DIR/rtabmap.db..."

echo "🔗 Publishing static TF odom -> World..."
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom World &

ros2 launch rtabmap_launch rtabmap.launch.py \
    database_path:="$WORK_DIR/rtabmap.db" \
    rgb_topic:=/go2_camera/rgb/image_raw \
    depth_topic:=/go2_camera/depth/image_raw \
    camera_info_topic:=/go2_camera/rgb/camera_info \
    odom_topic:=/odom \
    odom_frame_id:=odom \
    visual_odometry:=false \
    frame_id:=base \
    approx_sync:=true \
    use_sim_time:=true
