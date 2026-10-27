# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import os
import time
import csv
import math
import random
import torch
import torch.nn as nn
import gymnasium as gym
from isaaclab.app import AppLauncher

# local imports
import cli_args

# argparse 설정
parser = argparse.ArgumentParser(description="Play with Obstacle Randomization and CSV logging.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0", help="Task name.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 앱 실행
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 앱 부팅 후 임포트
import omni
import carb
import omni.appwindow
from pxr import Gf, UsdGeom
import omni.graph.core as og
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

# 센서 세팅 모듈
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import ros2_sensor_setup

# 공용 정책 네트워크 (ELU 기반)
# best_agent.pt (skrl PPO)의 실제 net_container 구조가 128-128-128 은닉층인 것을
# 확인 후 맞춤 (기존 512-256-128은 체크포인트와 shape이 맞지 않아 로드가 항상 실패했음)
class UniversalPolicy(nn.Module):
    def __init__(self, input_size=48, hidden_size=128, output_size=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, obs):
        if isinstance(obs, dict): obs = obs.get("states", obs)
        return self.net(obs)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

# ROS 2 자동 목적지 전송 노드
class Nav2AutoGoal(Node):
    def __init__(self):
        super().__init__('nav2_auto_goal_node')
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

    def send_nav2_goal(self, start_x, start_y, goal_x, goal_y):
        # 1. 초기 위치 전송 (Nav2 정렬)
        init_msg = PoseWithCovarianceStamped()
        init_msg.header.frame_id = "map"
        init_msg.header.stamp = self.get_clock().now().to_msg()
        init_msg.pose.pose.position.x = float(start_x)
        init_msg.pose.pose.position.y = float(start_y)
        yaw = math.atan2(goal_y - start_y, goal_x - start_x)
        init_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        init_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        init_msg.pose.covariance = [0.0]*36
        init_msg.pose.covariance[0], init_msg.pose.covariance[7], init_msg.pose.covariance[35] = 0.25, 0.25, 0.06
        self.init_pub.publish(init_msg)
        
        time.sleep(1.0) # Nav2 처리 대기
        
        # 2. 목적지 전송
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = float(goal_x)
        goal_msg.pose.position.y = float(goal_y)
        goal_msg.pose.orientation.w = 1.0
        self.goal_pub.publish(goal_msg)
        print(f"\n[Auto Goal] Nav2 목적지 전송 완료: ({goal_x}, {goal_y})")

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg, *args, **kwargs):
    # ROS 2 초기화
    if not rclpy.ok(): rclpy.init()
    nav_node = Nav2AutoGoal()

    # 환경 생성 (이전 커밋에서 실수로 삭제되었던 부분 복원)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 1. 로봇 경로 획득 및 ROS 2 센서 세팅
    robot_asset = env.unwrapped.scene["robot"]
    # robot_prim_path는 보통 "/World/envs/env_0/Robot" 형태입니다.
    robot_prim_path = robot_asset.root_physx_view.prim_paths[0] 

    # [수정] 경로 중복 방지를 위해 조건부로 base 추가
    if robot_prim_path.endswith("/base"):
        robot_base_path = robot_prim_path
    else:
        robot_base_path = f"{robot_prim_path}/base"

    camera_path = f"{robot_base_path}/front_camera"

    
    stage = omni.usd.get_context().get_stage()
    ros2_sensor_setup.setup_ros2_sensors(stage, 
                                       robot_base_path=robot_base_path, 
                                       camera_path=camera_path)

    # 2. 모델 로드 (순수 PyTorch 방식)
    resume_path = args_cli.checkpoint
    if not resume_path:
        from isaaclab_tasks.utils import get_checkpoint_path
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    policy = UniversalPolicy().to(env.unwrapped.device)
    checkpoint = torch.load(resume_path, map_location=env.unwrapped.device)
    state_dict = checkpoint.get("policy", checkpoint.get("agent", checkpoint.get("state_dict", checkpoint)))
    # skrl PPO 체크포인트는 공유 trunk를 "net_container.{0,2,4}", 행동 평균 출력 헤드를
    # "policy_layer"로 저장함 (value_layer/log_std_parameter는 play에 불필요).
    # 기존의 .replace("policy.","").replace("model.","")는 이 이름들에 대해 아무 것도
    # 바꾸지 못해 cleaned_dict가 항상 비어 있었고, load_state_dict(strict=False)가
    # 조용히 0개 텐서만 로드한 채 랜덤 초기화 가중치로 계속 실행되던 것이 원인이었음.
    key_map = {
        "net_container.0.weight": "net.0.weight", "net_container.0.bias": "net.0.bias",
        "net_container.2.weight": "net.2.weight", "net_container.2.bias": "net.2.bias",
        "net_container.4.weight": "net.4.weight", "net_container.4.bias": "net.4.bias",
        "policy_layer.weight": "net.6.weight", "policy_layer.bias": "net.6.bias",
    }
    cleaned_dict = {key_map[k]: v for k, v in state_dict.items() if k in key_map}
    # strict=True: 매핑이 어긋나면 조용히 넘어가지 말고 바로 에러로 드러나게 함
    policy.load_state_dict(cleaned_dict, strict=True)
    print(f"[INFO] Loaded {len(cleaned_dict)}/8 policy weight tensors from checkpoint")
    policy.eval()

    # 3. 평가 및 장애물 설정
    GOAL_X, GOAL_Y = 5.0, 0.0
    drawer_path = "/World/fused_scene/Moveable_Objects/Drawer"
    result_file = "./play_eval_results.csv"
    
    if not os.path.exists(result_file):
        with open(result_file, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Success", "Distance_to_Goal"])

    print("\n[INFO] Play 모드 시작: Nav2로 목표를 주면 자동으로 성공 여부를 기록합니다.")

    # keyboard subscription
    def make_keyboard_state():
        return {"forward": 0.0, "side": 0.0, "yaw": 0.0, "reset": False}

    def update_keyboard_state(state, event):
        pressed = event.type == carb.input.KeyboardEventType.KEY_PRESS
        released = event.type == carb.input.KeyboardEventType.KEY_RELEASE
        if not (pressed or released): return
        
        # [디버그] 키보드 입력 확인
        # print(f"[Key Event] {event.input} {'Pressed' if pressed else 'Released'}")
        
        if event.input == carb.input.KeyboardInput.W: state["forward"] = 1.0 if pressed else 0.0
        elif event.input == carb.input.KeyboardInput.S: state["forward"] = -1.0 if pressed else 0.0
        elif event.input == carb.input.KeyboardInput.A: state["side"] = 0.5 if pressed else 0.0
        elif event.input == carb.input.KeyboardInput.D: state["side"] = -0.5 if pressed else 0.0
        elif event.input == carb.input.KeyboardInput.Q: state["yaw"] = 0.8 if pressed else 0.0
        elif event.input == carb.input.KeyboardInput.E: state["yaw"] = -0.8 if pressed else 0.0
        elif event.input == carb.input.KeyboardInput.R: state["reset"] = True if pressed else False

    keyboard_state = make_keyboard_state()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()
    def on_keyboard_event(event, *args, **kwargs):
        update_keyboard_state(keyboard_state, event)
        return True
    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("\n================ Keyboard Control ================")
    print("Click the 3D viewport once, then use:")
    print("W / S : forward / backward")
    print("A / D : left / right")
    print("Q / E : yaw left / yaw right")
    print("R : Manual Reset (Respawn)")
    print("=================================================\n")

    ep_count = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            # 1️⃣ 에피소드 시작 시 환경 리셋
            env.reset()
            
            # [수정] 환경 scale을 0.54 -> 0.459로 추가 축소하면서 바닥 z도 0.85배로
            # 따라 올라옴: -0.162 * 0.85 = -0.1377
            START_X, START_Y = 0.0, 0.0
            root_state = robot_asset.data.default_root_state.clone()
            root_state[:, 0] = START_X
            root_state[:, 1] = START_Y
            root_state[:, 2] = -0.1377 # 안정적인 스폰 높이 (환경 0.459배 축소 반영)
            robot_asset.write_root_pose_to_sim(root_state[:, :7])
            robot_asset.reset()
            
            # [수정] 리셋 직후 로봇이 안정적으로 지면에 내려올 때까지 충분히 대기 (60스텝)
            for _ in range(60):
                simulation_app.update()
            
            # [제거] WASD 수동 주행/RTAB-Map 매핑 중 리셋될 때마다 Nav2 목적지가
            # 자동으로 재전송되던 부분을 삭제 (사용자 요청). 필요하면 nav2_drive.sh나
            # auto_goal_sender.py로 별도 실행할 것.

            # 2️⃣ 장애물 랜덤 배치
            drawer_prim = stage.GetPrimAtPath(drawer_path)
            if drawer_prim.IsValid():
                x_rand = random.uniform(-1.0, 3.0)
                y_rand = random.uniform(-0.76, 0.76)
                mat = Gf.Matrix4d().SetTranslate(Gf.Vec3d(x_rand, y_rand, 0.01))
                UsdGeom.Xformable(drawer_prim).MakeMatrixXform().Set(mat)
            
            # 3️⃣ 물리 업데이트 및 깨끗한 관측치 추출
            simulation_app.update()
            obs_dict = env.unwrapped.observation_manager.compute()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

            step_count = 0
            while simulation_app.is_running():
                start_time = time.time()
                step_count += 1

                # [수동 리셋 (R키)] 
                if keyboard_state["reset"]:
                    print("🔄 [수동 리셋] R키가 입력되어 에피소드를 재시작합니다.")
                    keyboard_state["reset"] = False
                    break 
                
                # 1. Nav2 명령 수신
                nav_x, nav_y, nav_yaw = 0.0, 0.0, 0.0
                try:
                    cmd_node = og.Controller.node("/World/ROS2_Camera_Graph/ROS2CmdVel")
                    if cmd_node.is_valid():
                        lin_vel = og.Controller.get(og.Controller.attribute("outputs:linearVelocity", cmd_node))
                        ang_vel = og.Controller.get(og.Controller.attribute("outputs:angularVelocity", cmd_node))
                        nav_x, nav_y, nav_yaw = float(lin_vel[0]), float(lin_vel[1]), float(ang_vel[2])
                except: pass

                # 2. 키보드 입력 처리 (Keyboard override Nav2)
                kb_x = keyboard_state["forward"]
                kb_y = keyboard_state["side"]
                kb_yaw = keyboard_state["yaw"]
                
                # 키보드 입력이 있으면 키보드 값을, 없으면 Nav2 값을 사용
                final_x = kb_x if abs(kb_x) > 0.01 else nav_x
                final_y = kb_y if abs(kb_y) > 0.01 else nav_y
                final_yaw = kb_yaw if abs(kb_yaw) > 0.01 else nav_yaw

                # 3. 로봇에게 명령 전달
                cmd = torch.tensor([final_x, final_y, final_yaw], device=env.unwrapped.device).unsqueeze(0)
                env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd

                # 4. 정책 실행 및 시뮬레이션 스텝
                actions = policy(obs)
                obs_dict, _, terminated, truncated, _ = env.step(actions)
                obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

                # ROS 2 콜백 처리
                rclpy.spin_once(nav_node, timeout_sec=0)

                # [디버그] 명령 상태 출력 (0.5초 간격)
                if int(time.time() * 2) % 2 == 0:
                    print(f"\r[Control] Vel X: {final_x:.2f}, Y: {final_y:.2f}, Yaw: {final_yaw:.2f} | Dist: {math.dist(robot_asset.data.root_pos_w[0][:2].tolist(), [GOAL_X, GOAL_Y]):.2f}m", end="")

                # 성공 여부 체크
                curr_pos = robot_asset.data.root_pos_w[0]
                dist = math.dist((curr_pos[0].item(), curr_pos[1].item()), (GOAL_X, GOAL_Y))
                
                if dist < 0.5:
                    print(f"\n🚩 [목적지 도달] 성공 기록 저장!")
                    with open(result_file, 'a', newline='') as f:
                        csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), True, round(dist, 2)])
                    time.sleep(1.0); break

                # 넘어짐 또는 시간 초과 시 리셋 (초기 50스텝 동안은 무시하여 안정화 유도)
                if (terminated | truncated)[0] and step_count > 50:
                    print(f"\n💥 [리셋] 로봇이 넘어졌거나 제한 시간을 초과했습니다.")
                    break

                if args_cli.real_time:
                    time.sleep(max(0, 0.02 - (time.time() - start_time)))
            
            ep_count += 1
            print(f"🔄 에피소드 {ep_count} 종료.")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
