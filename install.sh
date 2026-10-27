#!/bin/bash
# Install this repo's Python overrides into an Isaac Lab checkout
# (CGV-HGU/isaac-go2-ros2, branch simul-trio). Originals are kept as *.orig.
#
#   ./install.sh                      # Isaac Lab at ~/IsaacLab
#   ISAACLAB_DIR=/path/to/IsaacLab ./install.sh

set -e
source "$(dirname "${BASH_SOURCE[0]}")/scripts/env.sh"

TASK_DIR="$ISAACLAB_DIR/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2"
SKRL_DIR="$ISAACLAB_DIR/scripts/reinforcement_learning/skrl"

for d in "$TASK_DIR" "$SKRL_DIR"; do
    if [ ! -d "$d" ]; then
        echo "❌ Not found: $d"
        echo "   ISAACLAB_DIR ($ISAACLAB_DIR) should be a checkout of"
        echo "   https://github.com/CGV-HGU/isaac-go2-ros2 (branch simul-trio)."
        exit 1
    fi
done

install_file() {
    local src="$REPO_ROOT/isaaclab_overrides/$1" dest="$2/$1"
    if [ -f "$dest" ] && [ ! -f "$dest.orig" ]; then
        cp "$dest" "$dest.orig"
    fi
    cp "$src" "$dest"
    echo "✅ $1 -> $dest"
}

install_file flat_env_cfg.py "$TASK_DIR"
install_file play.py "$SKRL_DIR"
install_file evaluate_nav.py "$SKRL_DIR"
install_file ros2_sensor_setup.py "$SKRL_DIR"
chmod +x "$REPO_ROOT"/scripts/*.sh

echo ""
echo "Done. Next: put the lab assets in place (assets/README.md), then run ./scripts/play.sh"
