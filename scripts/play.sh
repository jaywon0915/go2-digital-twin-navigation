#!/bin/bash
# Launch Isaac Sim with the Go2 standing in the Oseok corridor digital twin.
# Drive with WASD (Q/E yaw, R respawn) after clicking the viewport, or let Nav2
# drive it through /cmd_vel. Use this for mapping and for manual Nav2 goals.
#
# Extra arguments are passed to play.py, e.g. ./scripts/play.sh --real-time

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
use_isaac_sim_ros2_env
require_file "$GO2_SCENE_USD" "Place the scene files as described in assets/README.md, or set GO2_SCENE_USD."
require_file "$SKRL_CHECKPOINT" "Place the skrl checkpoint as described in assets/README.md, or set SKRL_CHECKPOINT."

echo "🚀 Starting Unitree Go2 simulation (play)..."
cd "$ISAACLAB_DIR" || exit 1
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 1 \
    --checkpoint="$SKRL_CHECKPOINT" \
    "$@"
