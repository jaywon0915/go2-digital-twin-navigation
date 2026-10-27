# Assets (not in git)

The digital-twin scene, SLAM map and trained checkpoints are not in this
repository (about 530 MB of lab data). Place them here with this layout, or
point the environment variables in `scripts/env.sh` at another location.

```
assets/
├── scene/
│   └── 05_08_real(original).usd          # main scene: GO2_SCENE_USD
├── Downloads/                            # name is fixed, see note below
│   ├── oseok111.usd                      # corridor mesh (collision)
│   └── hello/
│       ├── cloud_final.usd               # corridor mesh (collision)
│       └── realGaussian.usdz             # 3D Gaussian splat (visuals)
├── map/                                  # MAP_DIR
│   ├── rtabmap.db                        # RTAB-Map database (3D map)
│   ├── rtabmap.pgm                       # 2D occupancy grid for Nav2
│   └── rtabmap.yaml
└── checkpoints/
    ├── skrl_unitree_go2_flat_2026-04-13_12-51-57_ppo_torch/
    │   └── best_agent.pt                 # SKRL_CHECKPOINT (play.sh)
    └── rsl_rl_unitree_go2_flat_2026-04-02_17-36-56/
        └── model_1999.pt                 # RSL_RL_CHECKPOINT (evaluate_nav.sh)
```

**Why `Downloads/`:** the main scene loads its parts through relative paths
baked into the USD (`../Downloads/oseok111.usd`, `../Downloads/hello/...`).
Whatever folder holds the main scene needs a sibling folder named `Downloads`
with those files in it.

`map/` and `map_backups/` are written by the mapping scripts.
