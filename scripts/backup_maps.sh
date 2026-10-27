#!/bin/bash
# Snapshot the current RTAB-Map database + 2D map before a new mapping session
# overwrites them (rtabmap_mapping.sh uses --delete_db_on_start).
# Run this BEFORE ./scripts/rtabmap_mapping.sh whenever you're about to remap.

set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEST="$MAP_BACKUP_DIR/$TIMESTAMP"

mkdir -p "$DEST"

for f in rtabmap.db rtabmap.yaml rtabmap.pgm; do
    if [ -f "$MAP_DIR/$f" ]; then
        cp "$MAP_DIR/$f" "$DEST/"
        echo "✅ Backed up $f -> $DEST/$f"
    else
        echo "⚠️  $MAP_DIR/$f not found, skipping"
    fi
done

# convenience symlink to the most recent backup
ln -sfn "$DEST" "$MAP_BACKUP_DIR/latest"

echo ""
echo "📦 Snapshot saved to: $DEST"
echo "   (also linked as: $MAP_BACKUP_DIR/latest)"
echo ""
echo "To restore this snapshot as the active map:"
echo "  cp \"$DEST\"/rtabmap.* \"$MAP_DIR/\""
