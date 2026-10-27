#!/bin/bash
# Automated Nav2 navigation benchmark: 3 start/goal pairs x 30 valid trials.
# Starts its own Isaac Sim instance, so don't run play.sh at the same time
# (both publish the same ROS 2 topics). Run ./scripts/rtabmap_localization.sh
# in a second terminal. Results are appended live to eval_results.csv in
# $ISAACLAB_DIR.
#
# Runs with the GUI by default. --headless --enable_cameras can be passed
# through, but sensor topics were unreliable in headless mode.

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
use_isaac_sim_ros2_env
require_file "$GO2_SCENE_USD" "Place the scene files as described in assets/README.md, or set GO2_SCENE_USD."
# Must be the RSL-RL checkpoint: OnPolicyRunner expects actor_state_dict /
# critic_state_dict. The skrl best_agent.pt (policy/value/optimizer) crashes
# with KeyError here.
require_file "$RSL_RL_CHECKPOINT" "Place the RSL-RL checkpoint as described in assets/README.md, or set RSL_RL_CHECKPOINT."

echo "🚀 Starting Unitree Go2 Nav2 evaluation..."
cd "$ISAACLAB_DIR" || exit 1
./isaaclab.sh -p scripts/reinforcement_learning/skrl/evaluate_nav.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 1 \
    --checkpoint "$RSL_RL_CHECKPOINT" \
    "$@"
