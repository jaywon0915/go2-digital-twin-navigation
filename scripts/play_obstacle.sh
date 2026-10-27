#!/bin/bash
# Run the RSL-RL Go2 walking policy with Isaac Lab's stock RSL-RL play script
# (no keyboard/Nav2 control). Useful for checking the policy on its own.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
use_isaac_sim_ros2_env
require_file "$GO2_SCENE_USD" "Place the scene files as described in assets/README.md, or set GO2_SCENE_USD."
require_file "$RSL_RL_CHECKPOINT" "Place the RSL-RL checkpoint as described in assets/README.md, or set RSL_RL_CHECKPOINT."

cd "$ISAACLAB_DIR" || exit 1
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 1 \
    --checkpoint "$RSL_RL_CHECKPOINT" \
    "$@"
