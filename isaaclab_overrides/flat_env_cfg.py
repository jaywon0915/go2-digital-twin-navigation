# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import UsdFileCfg
import isaaclab.sim as sim_utils

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # 1. 부모 클래스(RoughEnv)의 설정 먼저 불러오기
        super().__post_init__()

        # 기본값(velocity_env_cfg.py) episode_length_s=20.0이면 WASD로 수동 주행/RTAB-Map
        # 매핑 중에도 20초마다 강제로 에피소드가 리셋되어 로봇이 시작 지점으로 순간이동함
        # (SLAM 그래프를 깨뜨리는 가짜 포즈 점프). 수동 주행 세션 동안은 사실상 끄기 위해
        # 아주 크게 설정.
        self.episode_length_s = 3600.0

        # --- [보상 설정] 키보드 주행 및 회전(QE) 최적화 ---
        if hasattr(self.rewards, "track_ang_vel_yaw_exp"):
            self.rewards.track_ang_vel_yaw_exp.weight = 1.0
        
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # --- [명령 설정] 회전 범위 확장 ---
        if hasattr(self.commands, "base_velocity"):
            self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
            self.commands.base_velocity.heading_command = False

        # --- [환경 설정: 기본 바닥 삭제 및 메쉬 전용 환경] ---
        
        # 2. 기존 검은색 격자 바닥(Plane) 생성을 완전히 제거
        self.scene.terrain = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=None,  # 이 설정으로 인해 기본 바닥이 깔리지 않습니다.
        )
        
        # 3. 가우시안 복도 메쉬를 유일한 지형으로 등록
        # 씬 경로는 하드코딩하지 않고 scripts/env.sh가 export하는 GO2_SCENE_USD에서 읽음
        scene_usd = os.environ.get("GO2_SCENE_USD", "")
        if not os.path.isfile(scene_usd):
            raise FileNotFoundError(
                f"GO2_SCENE_USD must point to the corridor scene USD (got {scene_usd!r}). "
                "Launch through scripts/*.sh or export GO2_SCENE_USD; see assets/README.md."
            )
        self.scene.custom_environment = AssetBaseCfg(
            prim_path="/World/fused_scene",
            spawn=UsdFileCfg(
                usd_path=scene_usd,
                # 0.54배로도 여전히 15% 작아 보인다는 확인에 따라 추가 보정
                # 0.54 * 0.85 = 0.459
                scale=(0.459, 0.459, 0.459),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
        )

        # --- [로봇 시작 위치 설정] ---
        # 환경 scale을 0.54 -> 0.459로 추가 축소하면서 바닥 z도 같은 비율(0.85배)로
        # 따라 올라옴: -0.162 * 0.85 = -0.1377 (XY는 원점이라 스케일에 영향 없음).
        self.scene.robot.init_state.pos = (0.0, 0.0, -0.1377)

        # 지형 스캔 관련 불필요한 기능 끄기
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # 테스트용 환경 설정 (1개 로봇)
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        
        # 테스트 시 불필요한 노이즈 및 랜덤 충격 제거
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
