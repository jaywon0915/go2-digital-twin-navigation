# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import carb
import carb.input
import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        policy_nn = None
    else:
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # =========================================================================
    # 🎯 100회 자동 평가 시나리오 설정 (최상단 정의)
    # =========================================================================
    import csv
    import math
    import random
    from pxr import Gf, UsdGeom
    import omni.physx
    import omni.kit.app
    # --headless 프로파일(apps/isaaclab.python.headless.kit)은 omni.graph 확장을
    # 기본으로 로드하지 않음 (GUI 모드의 play.sh에서는 기본 포함되어 있어 문제 없었음).
    # import 전에 명시적으로 켜줘야 ModuleNotFoundError('omni.graph')가 안 남.
    _ext_manager = omni.kit.app.get_app().get_extension_manager()
    _ext_manager.set_extension_enabled_immediate("omni.graph.core", True)
    _ext_manager.set_extension_enabled_immediate("omni.graph.action", True)
    import omni.graph.core as og

    # 2026-09-16 (2차 수정): 37.8% 성공률의 실제 원인을 다시 조사한 결과, 목적지
    # 좌표 문제가 아니라 config/nav2_params.yaml의 inflation_radius(0.6m)가 이 복도의
    # 실제 폭(scipy distance-transform으로 측정한 가장 넓은 곳도 반폭 0.81m, 실제
    # 주행 궤적 클리어런스 중앙값 0.55m)에 비해 너무 커서 복도 전체가 항상 벽
    # inflation cost 안에 있었던 게 근본 원인이었음 (inflation_radius를 0.35로
    # 낮춤, 이 파일의 costmap 설정 참고). 그에 맞춰 목적지 쌍도 다시 선정:
    # 이번엔 직선 경로 전체가 아니라 각 지점 자체의 실측 클리어런스만 요구함
    # (Nav2가 알아서 굽은 경로를 계획하므로) -- 복도 3등분 구간에서 각각
    # 클리어런스가 가장 넓은 지점들 중 3~7m 떨어진 쌍을 선택.
    # spawn_z는 실측 궤적의 순간 오도메트리 높이를 그대로 썼더니 로봇이 바닥에
    # 파묻혀 스폰되는 현상이 있었음 (걸음걸이 중 순간적인 높이 변동/드리프트 때문에
    # 실제 정지 시 바닥 높이보다 낮게 기록될 수 있음) -- +0.15m 여유를 둬서 살짝
    # 위에서 떨어지도록 보정 (100틱 안정화 대기가 있으니 이 정도 낙하는 안전함).
    GOAL_SETS = [
        {"start": (-13.02, 4.68), "goal": (-9.97, 3.57), "spawn_z": -0.126 + 0.15},
        {"start": (7.47, -3.04), "goal": (3.42, -1.48), "spawn_z": -0.694 + 0.15},
        {"start": (19.02, -7.47), "goal": (14.48, -5.61), "spawn_z": -1.005 + 0.15},
    ]
    EPISODES_PER_GOAL = 30  # 목적지당 유효(재시도 제외) 실험 횟수 -- 3개 목적지 = 총 90회
    INSTANT_FAIL_STEPS = 5  # 이 스텝 이내에 dones가 뜨면 "불안정한 스폰"으로 보고 집계에서 제외, 재시도
    MAX_RETRIES_PER_TRIAL = 15  # 안전장치: 같은 시도가 계속 불안정하면 무한루프 방지용으로 그냥 기록

    # 목적지와 출발지 '사이'에 물체를 놓기 위한 범위 정의
    X_MIN, X_MAX = -1.0, 3.0
    Y_BOUND = 0.76

    result_file = "./eval_results.csv"

    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Goal_Set", "Start_X", "Start_Y", "Goal_X", "Goal_Y",
                          "Episode", "Success", "Time_Steps", "Final_Dist", "Retries"])

    # [중요] stage 및 경로 설정
    stage = env.unwrapped.scene.stage
    drawer_path = "/World/Moveable_Objects/Drawer"
    physx_iface = omni.physx.get_physx_interface()

    # [중요] ROS2 Bridge 확장 기능 활성화 및 로딩 대기 (headless 모드에서는 바로
    # 그래프를 만들면 "Failed to wrap graph in node" 에러가 남 -- 확장이
    # 활성화된 프레임과 실제 노드 타입이 레지스트리에 등록 완료되는 시점 사이에
    # 지연이 있음)
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    ext_manager.set_extension_enabled_immediate("isaacsim.core.nodes", True)

    print("[INFO] ROS2 노드 타입 로딩 대기 중...")
    for _ in range(100):
        simulation_app.update()

    # [필수] RGB/Depth/Odom/TF ROS2 브리지 구성 (RTAB-Map/Nav2가 필요로 하는
    # 센서 데이터). 이게 없으면 이 스크립트는 로봇 상태를 RTAB-Map 쪽으로
    # 전혀 내보내지 않아 로봇이 절대 움직이지 않음.
    # play.py와 동일한 방식으로 robot_base_path/camera_path를 구해 설정.
    robot_asset = env.unwrapped.scene["robot"]
    robot_prim_path = robot_asset.root_physx_view.prim_paths[0]
    robot_base_path = robot_prim_path if robot_prim_path.endswith("/base") else f"{robot_prim_path}/base"
    camera_path = f"{robot_base_path}/front_camera"

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import ros2_sensor_setup
    ros2_sensor_setup.setup_ros2_sensors(stage, robot_base_path=robot_base_path, camera_path=camera_path)

    # ros2_sensor_setup이 만드는 /World/ROS2_Camera_Graph에 이미 /cmd_vel
    # 구독 노드(ROS2CmdVel)가 있으므로 별도 그래프를 또 만들 필요가 없음
    # (예전 코드는 /World/ROS2_Nav_Graph라는 중복 그래프를 따로 만들고 있었음).
    # play.py가 이미 쓰고 있는 검증된 방식(node + 상대 attribute 이름) 그대로 사용.
    _cmd_vel_node = og.Controller.node("/World/ROS2_Camera_Graph/ROS2CmdVel")
    linear_attr = og.Controller.attribute("outputs:linearVelocity", _cmd_vel_node)
    angular_attr = og.Controller.attribute("outputs:angularVelocity", _cmd_vel_node)

    # 목적지는 매 틱 갱신될 필요 없는 1회성 값이라 OmniGraph 대신 plain rclpy로
    # 발행 (이 Isaac Sim 5.1 ROS2 bridge 컴파일된 플러그인을 직접 확인한 결과
    # ROS2PublishPoseStamped라는 노드 타입 자체가 존재하지 않음 -- 예전 코드가
    # 쓰던 노드 이름이 애초에 틀렸던 것). play.py의 Nav2AutoGoal과 동일한 패턴.
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped

    if not rclpy.ok():
        rclpy.init()

    class _OneShotGoalPublisher(Node):
        def __init__(self):
            super().__init__("evaluate_nav_goal_publisher")
            self.pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        def send(self, x, y):
            msg = PoseStamped()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            msg.pose.orientation.w = 1.0
            self.pub.publish(msg)

    _goal_node = _OneShotGoalPublisher()
    time.sleep(1.0)  # 발행 전 디스커버리 대기 (Nav2AutoGoal.send_nav2_goal과 동일)
    # [중요] 여기서 딱 한 번만 보내면 안 됨 -- 이 시점에 bt_navigator가 아직
    # /goal_pose 구독을 끝내지 못했으면 이 메시지는 (volatile QoS라) 그냥
    # 유실되고 다시는 안 옴. 매 에피소드 시작마다 아래 루프에서 다시 보냄.

    TOTAL_TRIALS = len(GOAL_SETS) * EPISODES_PER_GOAL
    print(f"\n================ {len(GOAL_SETS)}개 목적지 x {EPISODES_PER_GOAL}회 = 총 {TOTAL_TRIALS}회 자동화 Nav2 평가 시작 ================")
    for gi, gs in enumerate(GOAL_SETS, 1):
        print(f"  목적지 {gi}: 출발 {gs['start']} -> 목적지 {gs['goal']}")

    for goal_idx, goal_set in enumerate(GOAL_SETS, 1):
      START_X, START_Y = goal_set["start"]
      GOAL_X, GOAL_Y = goal_set["goal"]
      SPAWN_Z = goal_set["spawn_z"]
      print(f"\n---------------- 목적지 세트 {goal_idx}/{len(GOAL_SETS)}: ({START_X},{START_Y}) -> ({GOAL_X},{GOAL_Y}) ----------------")

      valid_count = 0
      retries_this_trial = 0
      while valid_count < EPISODES_PER_GOAL and simulation_app.is_running():
        retry_note = f" (재시도 {retries_this_trial}회째)" if retries_this_trial else ""
        print(f"\n🎬 [목적지 {goal_idx}/{len(GOAL_SETS)} | 유효 시도 {valid_count+1}/{EPISODES_PER_GOAL}{retry_note}] 에피소드 초기화 중...")

        # [1] 로봇을 지정된 출발지로 강제 이동 (Teleport) 및 낙하 높이 안정화
        with torch.inference_mode():
            root_state = env.unwrapped.scene["robot"].data.default_root_state.clone()
            root_state[:, 0] = START_X
            root_state[:, 1] = START_Y
            # 바닥 높이는 위치마다 달라 실제 매핑 주행 중 이 지점에서 기록된 실측 로봇 높이를 사용
            root_state[:, 2] = SPAWN_Z

            # 시작 시 목적지를 똑바로 바라보도록 Yaw 각도 계산 및 정렬
            target_yaw = math.atan2(GOAL_Y - START_Y, GOAL_X - START_X)
            root_state[:, 3] = math.cos(target_yaw / 2.0)
            root_state[:, 6] = math.sin(target_yaw / 2.0)
            
            env.unwrapped.scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            env.unwrapped.scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            
            env.unwrapped.scene["robot"].reset()
        
        # [2] 장애물 무작위 배치 및 물리 엔진 동기화 리셋
        drawer_prim = stage.GetPrimAtPath(drawer_path)
        if drawer_prim.IsValid():
            if not drawer_prim.IsActive(): 
                drawer_prim.SetActive(True)

            x_rand = random.uniform(X_MIN, X_MAX)
            y_rand = random.uniform(-Y_BOUND, Y_BOUND)
            yaw_rand = random.uniform(0, 360)
            
            mat = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_rand))
            mat.SetTranslate(Gf.Vec3d(x_rand, y_rand, 0.0))
            
            UsdGeom.Xformable(drawer_prim).MakeMatrixXform().Set(mat)
            
            physx_iface.update_transform(drawer_path)
            try:
                physx_iface.set_rigid_body_linear_velocity(drawer_path, [0.0, 0.0, 0.0])
                physx_iface.set_rigid_body_angular_velocity(drawer_path, [0.0, 0.0, 0.0])
            except Exception:
                pass

            print(f"    🎲 [장애물 랜덤 배치 완료] X: {x_rand:.2f}, Y: {y_rand:.2f}")

        # 초기화 상태 안정화를 위한 대기 (기존 5틱=0.1초는 너무 짧음 -- 사용자가 실제
        # 매핑 주행 중 스폰 직후 로봇이 안정화되기까지 1~2초 정도 걸리는 것을 직접
        # 관찰함. play.py도 리셋 후 60틱을 기다림; 렌더링 스텝 0.02초 기준 100틱=2초)
        for _ in range(100):
            simulation_app.update()

        obs = env.get_observations()
        # 매 에피소드마다 실제로 목적지를 (재)발행 -- 로봇이 매번 출발지로
        # 텔레포트되므로 Nav2도 매번 새로 경로를 계획해야 함
        _goal_node.send(GOAL_X, GOAL_Y)
        print(f"    🎯 [Nav2 통신] 목적지 토픽 발행 완료 -> ({GOAL_X}, {GOAL_Y})")

        step_count = 0
        max_steps = 1500  
        success = False
        final_dist = 999.0
        
        while step_count < max_steps and simulation_app.is_running():
            start_time = time.time()
            
            # [4] OmniGraph로부터 ROS 2 /cmd_vel 속도 명령 실시간 패치
            # (og.Controller에 get_attr_value라는 메서드는 없음 -- play.py가 이미
            # 쓰고 있는 og.Controller.get(attribute)가 맞는 API)
            lin_vel = og.Controller.get(linear_attr)
            ang_vel = og.Controller.get(angular_attr)
            
            v_forward = lin_vel[0] if lin_vel is not None else 0.0
            v_lateral = lin_vel[1] if lin_vel is not None else 0.0
            v_yaw = ang_vel[2] if ang_vel is not None else 0.0
            
            with torch.inference_mode():
                current_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
                curr_x = current_pos[0].item()
                curr_y = current_pos[1].item()
                
                final_dist = math.dist((curr_x, curr_y), (GOAL_X, GOAL_Y))
                
                # 목적지 통과 판정 (50cm 이내 접근 시 성공)
                if final_dist < 0.5:
                    success = True
                    print(f"    🎉 [성공] Nav2 제어로 목적지 도달! (소요 스텝: {step_count}, 오차: {final_dist:.2f}m)")
                    break

                # [5] Nav2 명령을 로봇 제어기에 전달
                cmd = torch.tensor([v_forward, v_lateral, v_yaw], device=env.unwrapped.device, dtype=torch.float32)
                cmd = cmd.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
                env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd

                # RL Policy 구동 및 시뮬레이션 전진
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    policy_nn.reset(dones)
                    
                if dones[0]:
                    print("    💥 [실패] 로봇이 넘어지거나 외부 충돌로 인하여 에피소드가 파괴됨.")
                    break
                    
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
                
            step_count += 1
            
        if not success and step_count >= max_steps:
            print(f"    ⏰ [실패] 타임아웃 제한 시간 초과 (남은 거리: {final_dist:.2f}m)")
            
        # [불안정 스폰 감지] 스폰 직후 몇 스텝 만에 실패한 경우 -- 사용자가 관찰한
        # "가끔 스폰이 이상하게 되는" 현상. 실제 주행/충돌 실패가 아니므로 집계하지
        # 않고 같은 유효 시도 슬롯을 재시도함 (MAX_RETRIES_PER_TRIAL 넘으면 안전장치로
        # 그냥 기록하고 진행 -- 무한루프 방지).
        if (not success) and step_count <= INSTANT_FAIL_STEPS and retries_this_trial < MAX_RETRIES_PER_TRIAL:
            retries_this_trial += 1
            print(f"    ⚠️  [무효 시도] 스폰 직후 {step_count}스텝 만에 종료됨 -- 불안정한 스폰으로 판단, "
                  f"집계하지 않고 재시도합니다. (누적 재시도 {retries_this_trial}/{MAX_RETRIES_PER_TRIAL})")
            continue

        valid_count += 1
        # 결과를 CSV 데이터베이스 파일에 기록
        with open(result_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([goal_idx, START_X, START_Y, GOAL_X, GOAL_Y,
                              valid_count, success, step_count, round(final_dist, 2), retries_this_trial])
        retries_this_trial = 0

    print(f"\n================ 🎉 다중 목적지({len(GOAL_SETS)}개 x {EPISODES_PER_GOAL}회) 자동 평가가 최종 완료되었습니다! ================")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()