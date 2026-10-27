#!/bin/bash
# Save RTAB-Map's live 2D occupancy grid to $MAP_DIR/rtabmap.pgm + rtabmap.yaml
# (the map Nav2 loads). Run while rtabmap_mapping.sh or rtabmap_save_map.sh
# is still running: map_saver needs a live map topic.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
use_system_ros2_env
mkdir -p "$MAP_DIR"

echo "💾 Saving $MAP_TOPIC to $MAP_DIR/rtabmap.{pgm,yaml}..."
ros2 run nav2_map_server map_saver_cli -f "$MAP_DIR/rtabmap" \
    --ros-args -p use_sim_time:=true -r map:="$MAP_TOPIC"
