# /home/minjun/wr_ws/src/welding_robot/dm_ros/dm_ros/welding_free2.py
import rclpy
import os
import numpy as np
import mujoco
import threading
import time
from prc import Fr3Controller
from .utils.multi_thread import MujocoROSBridge
# from .utils.collision_avoidance import MuJoCoCollisionAvoidance, integrate_collision_avoidance_to_bridge

class WeldingEnvironmentRandomizer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # 테이블과 관련 body들의 ID 저장
        self.table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
        self.lap_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lap_base")
        self.fillet_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fillet_joint_base")
        self.curved_pipe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "curved_pipe")
        self.moving_obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_obstacle")
        self.worker_torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "worker_torso")
        
        # 원래 위치 저장 (상대 위치 계산용)
        if self.table_id != -1:
            self.original_table_pos = model.body_pos[self.table_id].copy()
            self.relative_positions = {}
            
            if self.lap_base_id != -1:
                self.relative_positions['lap_base'] = model.body_pos[self.lap_base_id] - self.original_table_pos
            if self.fillet_joint_id != -1:
                self.relative_positions['fillet_joint'] = model.body_pos[self.fillet_joint_id] - self.original_table_pos
            if self.curved_pipe_id != -1:
                self.relative_positions['curved_pipe'] = model.body_pos[self.curved_pipe_id] - self.original_table_pos
            if self.moving_obstacle_id != -1:
                self.relative_positions['moving_obstacle'] = model.body_pos[self.moving_obstacle_id] - self.original_table_pos
            if self.worker_torso_id != -1:
                self.relative_positions['worker_torso'] = model.body_pos[self.worker_torso_id] - self.original_table_pos
        
    def randomize_table_position(self, x_range=(0.45, 0.65), y_range=(-0.15, 0.15), z_range=(0.42, 0.48)):
        """테이블 위치를 무작위화하고 관련된 모든 오브젝트도 함께 이동 (로봇 작업공간 내)"""
        if self.table_id == -1:
            return np.array([0.5, 0, 0.45])  # 기본값 반환
        
        max_attempts = 10
        for attempt in range(max_attempts):
            # 새로운 테이블 위치 생성 (더 보수적인 범위)
            new_table_pos = np.array([
                np.random.uniform(*x_range),
                np.random.uniform(*y_range),
                np.random.uniform(*z_range)
            ])
            
            # 테이블 위치 업데이트
            self.model.body_pos[self.table_id] = new_table_pos
            
            # 테이블 위의 모든 오브젝트들도 상대 위치 유지하며 이동
            if self.lap_base_id != -1 and 'lap_base' in self.relative_positions:
                self.model.body_pos[self.lap_base_id] = new_table_pos + self.relative_positions['lap_base']
            if self.fillet_joint_id != -1 and 'fillet_joint' in self.relative_positions:
                self.model.body_pos[self.fillet_joint_id] = new_table_pos + self.relative_positions['fillet_joint']
            if self.curved_pipe_id != -1 and 'curved_pipe' in self.relative_positions:
                self.model.body_pos[self.curved_pipe_id] = new_table_pos + self.relative_positions['curved_pipe']
            if self.moving_obstacle_id != -1 and 'moving_obstacle' in self.relative_positions:
                self.model.body_pos[self.moving_obstacle_id] = new_table_pos + self.relative_positions['moving_obstacle']
            if self.worker_torso_id != -1 and 'worker_torso' in self.relative_positions:
                self.model.body_pos[self.worker_torso_id] = new_table_pos + self.relative_positions['worker_torso']
            
            # Forward kinematics 업데이트
            mujoco.mj_forward(self.model, self.data)
            
            # 모든 웨이포인트가 도달 가능한지 확인
            all_reachable = all([
                self.check_reachability("lap_start"),
                self.check_reachability("lap_waypoint1"),
                self.check_reachability("lap_waypoint2"),
                self.check_reachability("lap_waypoint3"),
                self.check_reachability("lap_end")
            ])
            
            if all_reachable:
                print(f"✅ Valid environment found on attempt {attempt + 1}")
                return new_table_pos
            else:
                print(f"⚠️  Attempt {attempt + 1}: Some waypoints unreachable, retrying...")
        
        # 모든 시도 실패시 안전한 기본 위치로
        print("🔄 Using safe default position")
        safe_pos = np.array([0.55, 0.0, 0.45])
        self.model.body_pos[self.table_id] = safe_pos
        return safe_pos
    
    def randomize_obstacles(self):
        """장애물 위치를 무작위화"""
        # 움직이는 장애물의 초기 위치 설정
        obstacle_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_z")
        ]
        
        for joint_id in obstacle_joint_ids:
            if joint_id != -1:
                range_min = self.model.jnt_range[joint_id][0]
                range_max = self.model.jnt_range[joint_id][1]
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = np.random.uniform(range_min, range_max)
    
    def randomize_lap_joint_angle(self, angle_range=(-0.3, 0.3)):
        """Lap joint의 초기 각도를 무작위화"""
        lap_base_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lap_base")
        if lap_base_joint != -1 and self.lap_base_id != -1:
            angle = np.random.uniform(*angle_range)
            self.model.body_quat[self.lap_base_id] = self._euler_to_quat(0, 0, angle)
            return angle
        return 0
    
    def check_reachability(self, site_name, robot_reach=0.8):
        """특정 사이트가 로봇의 도달 범위 내에 있는지 확인 (더 보수적)"""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id != -1:
            site_pos = self.data.site_xpos[site_id]
            # 로봇 베이스를 원점으로 가정하여 거리 계산
            distance = np.linalg.norm(site_pos[:2])  # XY 평면에서의 거리만 고려
            height = site_pos[2]  # Z 좌표
            
            # 높이 제한 (0.2m ~ 0.7m)
            height_ok = 0.2 <= height <= 0.7
            # 수평 거리 제한
            distance_ok = distance <= robot_reach
            
            reachable = height_ok and distance_ok
            if not reachable:
                print(f"❌ {site_name}: dist={distance:.3f}m, height={height:.3f}m (reach={robot_reach}m)")
            
            return reachable
        return False
    
    def update_obstacles(self):
        """장애물 애니메이션 업데이트"""
        t = self.data.time
        
        # 움직이는 장애물 애니메이션
        obstacle_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x")
        obstacle_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y")
        
        if obstacle_x_id != -1:
            addr = self.model.jnt_qposadr[obstacle_x_id]
            self.data.qpos[addr] = 0.05 * np.sin(t * 0.5)  # 천천히 좌우 이동
            
        if obstacle_y_id != -1:
            addr = self.model.jnt_qposadr[obstacle_y_id]
            self.data.qpos[addr] = 0.03 * np.cos(t * 0.7)  # 천천히 앞뒤 이동
    
    def randomize_all(self):
        """전체 환경을 무작위화"""
        # 테이블 위치 무작위화
        table_pos = self.randomize_table_position()
        
        # Lap joint 각도 무작위화
        lap_angle = self.randomize_lap_joint_angle()
        
        # 장애물 무작위화
        self.randomize_obstacles()
        
        # Forward kinematics 업데이트
        mujoco.mj_forward(self.model, self.data)
        
        # 도달 가능성 확인
        reachable = all([
            self.check_reachability("lap_start"),
            self.check_reachability("lap_waypoint1"),
            self.check_reachability("lap_waypoint2"),
            self.check_reachability("lap_waypoint3"),
            self.check_reachability("lap_end")
        ])
        
        return {
            'table_position': table_pos,
            'lap_angle': lap_angle,
            'all_waypoints_reachable': reachable
        }
    
    def _euler_to_quat(self, roll, pitch, yaw):
        """Euler angles to quaternion conversion"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])

class WeldingMujocoROSBridge(MujocoROSBridge):
    """용접 환경을 위한 확장된 MujocoROSBridge (안전 모드)"""
    
    def __init__(self, robot_info, camera_info, robot_controller):
        super().__init__(robot_info, camera_info, robot_controller)
        
        # 환경 무작위화 초기화
        self.randomizer = WeldingEnvironmentRandomizer(self.model, self.data)
        
        # 환경 업데이트 관련 변수
        self.environment_initialized = False  # 환경 초기화 여부
        self.obstacle_animation_enabled = True  # 장애물 애니메이션만 유지
        
        # 충돌 모니터링 비활성화 (안전을 위해)
        self.collision_monitoring_enabled = False
        
        # 충돌 통계
        self.collision_stats = {
            'total_checks': 0,
            'collisions_detected': 0,
            'last_collision_time': 0
        }
        
        # 초기 환경 설정 (시뮬레이션 시작 시 한 번만)
        result = self.randomizer.randomize_all()
        self.environment_initialized = True
        
        self.get_logger().info(f"🎯 Welding environment initialized once:")
        self.get_logger().info(f"   Table position: {result['table_position']}")
        self.get_logger().info(f"   Lap angle: {result['lap_angle']:.3f} rad")
        self.get_logger().info(f"   All waypoints reachable: {result['all_waypoints_reachable']}")
        self.get_logger().info(f"🔒 Environment positions are now FIXED for this simulation session")
        self.get_logger().info(f"⚠️  Collision monitoring DISABLED for stability")
    
    def robot_control(self):
        """원래 robot_control 메서드를 오버라이드하여 용접 환경 업데이트 추가"""
        self.ctrl_step = 0
        sync_step = 30  # every 30 ctrl_steps

        try:
            while rclpy.ok() and self.running:            
                with self.lock:
                    start_time = time.perf_counter()                        

                    # 원래 시뮬레이션 스텝
                    mujoco.mj_step(self.model, self.data)

                    # 용접 환경 업데이트 (장애물 애니메이션)
                    self.update_welding_environment()

                    self.rc.updateModel(self.data, self.ctrl_step)
                    
                    # -------------------- ADD Controller ---------------------------- #
                    rclpy.spin_once(self.rc, timeout_sec=0.0001) # for scene monitor
                    rclpy.spin_once(self, timeout_sec=0.0001) # for robot controller
                    self.data.ctrl[:self.ctrl_dof] = self.rc.compute()   

                    # --- publish joint positions ---
                    js = self.get_joint_state_message()
                    self.joint_state_pub.publish(js)

                    # ---------------------------------------------------------------- #
                    self.ctrl_step += 1

                self.time_sync(self.dt, start_time, False)
            
        except KeyboardInterrupt:
            self.get_logger().info("\nSimulation interrupted. Closing robot controller ...")
            self.rc.destroy_node()

    def get_joint_state_message(self):
        """JointState 메시지 생성"""
        from sensor_msgs.msg import JointState
        
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        js.position = [
            float(self.data.qpos[self.model.joint(j).qposadr])
            for j in self.joint_names
        ]
        return js
    
    def update_welding_environment(self):
        """용접 환경 업데이트 - 장애물 애니메이션만 (안전 모드)"""
        # 장애물 애니메이션만 유지 (매 스텝)
        if self.obstacle_animation_enabled:
            self.randomizer.update_obstacles()
        
        # 충돌 모니터링 비활성화됨
    
    def monitor_collisions(self):
        """실시간 충돌 모니터링 (안전한 버전)"""
        try:
            self.collision_stats['total_checks'] += 1
            
            # 안전한 충돌 검사
            collision_detected, pairs = self.safe_collision_check()
            
            if collision_detected:
                self.collision_stats['collisions_detected'] += 1
                self.collision_stats['last_collision_time'] = self.data.time
                
                self.get_logger().warn(f"⚠️  COLLISION DETECTED at t={self.data.time:.2f}s")
                for i, (geom1, geom2, pos) in enumerate(pairs):
                    self.get_logger().warn(f"   {i+1}. {geom1} ↔ {geom2} at {pos}")
            
            # 주기적 통계 출력 (1000번마다)
            if self.collision_stats['total_checks'] % 1000 == 0:
                detection_rate = self.collision_stats['collisions_detected'] / self.collision_stats['total_checks'] * 100
                self.get_logger().info(f"📊 Collision Statistics:")
                self.get_logger().info(f"   Checks: {self.collision_stats['total_checks']}")
                self.get_logger().info(f"   Detections: {self.collision_stats['collisions_detected']}")
                self.get_logger().info(f"   Rate: {detection_rate:.2f}%")
                
        except Exception as e:
            self.get_logger().error(f"Collision monitoring error: {e}")
            # 충돌 모니터링 비활성화
            self.collision_monitoring_enabled = False
    
    def safe_collision_check(self):
        """안전한 충돌 검사"""
        try:
            # MuJoCo 충돌 계산 실행
            mujoco.mj_collision(self.model, self.data)
            
            collision_detected = self.data.ncon > 0
            collision_pairs = []
            
            if collision_detected and self.data.ncon < 100:  # 너무 많은 접촉은 무시
                for i in range(min(self.data.ncon, 10)):  # 최대 10개만 처리
                    try:
                        contact = self.data.contact[i]
                        
                        # geom ID 유효성 검사
                        if (0 <= contact.geom1 < self.model.ngeom and 
                            0 <= contact.geom2 < self.model.ngeom):
                            
                            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
                            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
                            
                            # 로봇-환경 충돌만 관심
                            if self.is_robot_environment_collision(geom1_name, geom2_name):
                                collision_pairs.append((geom1_name, geom2_name, contact.pos.copy()))
                                
                    except Exception as e:
                        # 개별 접촉 처리 오류는 무시하고 계속
                        continue
            
            return len(collision_pairs) > 0, collision_pairs
            
        except Exception as e:
            # 충돌 검사 자체에 오류가 있으면 안전하게 처리
            return False, []
    
    def get_collision_status(self):
        """현재 충돌 상태 반환 (비활성화됨)"""
        return {
            'collision_detected': False,
            'num_collision_pairs': 0,
            'collision_pairs': [],
            'monitoring_enabled': False
        }
    
    def disable_obstacle_animation(self):
        """장애물 애니메이션 비활성화 (완전 정적 환경)"""
        self.obstacle_animation_enabled = False
        self.get_logger().info("🔒 Obstacle animation disabled - fully static environment")
    
    def enable_obstacle_animation(self):
        """장애물 애니메이션 활성화"""
        self.obstacle_animation_enabled = True
        self.get_logger().info("🔓 Obstacle animation enabled")
    
    def manual_randomize_environment(self):
        """수동으로 환경 재무작위화 (필요시 호출)"""
        if self.environment_initialized:
            result = self.randomizer.randomize_all()
            self.get_logger().info(f"🔄 Environment manually randomized:")
            self.get_logger().info(f"   Table position: {result['table_position']}")
            self.get_logger().info(f"   All waypoints reachable: {result['all_waypoints_reachable']}")

def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    # ROS2 초기화
    rclpy.init()
    
    # 파일 경로 설정
    xml_path = os.path.join(current_dir, '../robots', "welding_scene2.xml")
    urdf_path = os.path.join(current_dir, '../robots', 'fr3/fr3_hand.urdf')
    
    # 파일 존재 확인
    if not os.path.exists(xml_path):
        print(f"❌ XML file not found: {xml_path}")
        return
    if not os.path.exists(urdf_path):
        print(f"❌ URDF file not found: {urdf_path}")
        return
    
    try:
        print("🤖 Initializing Welding Robot Controller...")
        
        # Fr3 컨트롤러 초기화
        rc = Fr3Controller(urdf_path)
        
        # 로봇 및 카메라 정보 설정
        robot_info = [xml_path, urdf_path, 1000]  # [xml_path, urdf_path, hz]
        camera_info = ['hand_eye', 320, 240, 30]  # [camera_name, width, height, fps]
        
        print("🌉 Setting up Welding MuJoCo-ROS Bridge...")
        
        # 용접 환경용 MuJoCo-ROS 브리지 초기화
        bridge = WeldingMujocoROSBridge(robot_info, camera_info, rc)
        
        print("🚀 Starting Welding Simulation with ROS Bridge...")
        print("💡 Available services:")
        print("   - /task_move_srv")
        print("   - /get_site_position")
        print("   - /get_site_orientation")
        print("🔧 You can now run the waypoint client!")
        print("   ros2 run dm_task_manager multi_pose_task_client_waypoints")
        print("🎮 Environment features:")
        print("   - Environment randomized ONCE at startup")
        print("   - Moving obstacle animation (realistic)")
        print("   - Fixed table and static obstacle positions")
        print("   - Real-time waypoint reachability checking")
        print("⚠️  Safety mode:")
        print("   - MuJoCo collision monitoring disabled")
        print("   - Client-side collision avoidance active")
        print("   - Stable operation prioritized")
        
        # 브리지 실행 (ROS 서비스와 시뮬레이션 동시 실행)
        bridge.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down welding simulation...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 정리
        if 'bridge' in locals():
            bridge.destroy_node()
        rclpy.shutdown()
        print("✅ Welding simulation terminated.")

if __name__ == "__main__":    
    main()


# # /home/minjun/wr_ws/src/welding_robot/dm_ros/dm_ros/welding_free2.py
# import rclpy
# import os
# import numpy as np
# import mujoco
# import threading
# import time
# from prc import Fr3Controller
# from .utils.multi_thread import MujocoROSBridge
# from .utils.collision_avoidance import MuJoCoCollisionAvoidance, integrate_collision_avoidance_to_bridge

# class WeldingEnvironmentRandomizer:
#     def __init__(self, model, data):
#         self.model = model
#         self.data = data
        
#         # 테이블과 관련 body들의 ID 저장
#         self.table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
#         self.lap_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lap_base")
#         self.fillet_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fillet_joint_base")
#         self.curved_pipe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "curved_pipe")
#         self.moving_obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_obstacle")
#         self.worker_torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "worker_torso")
        
#         # 원래 위치 저장 (상대 위치 계산용)
#         if self.table_id != -1:
#             self.original_table_pos = model.body_pos[self.table_id].copy()
#             self.relative_positions = {}
            
#             if self.lap_base_id != -1:
#                 self.relative_positions['lap_base'] = model.body_pos[self.lap_base_id] - self.original_table_pos
#             if self.fillet_joint_id != -1:
#                 self.relative_positions['fillet_joint'] = model.body_pos[self.fillet_joint_id] - self.original_table_pos
#             if self.curved_pipe_id != -1:
#                 self.relative_positions['curved_pipe'] = model.body_pos[self.curved_pipe_id] - self.original_table_pos
#             if self.moving_obstacle_id != -1:
#                 self.relative_positions['moving_obstacle'] = model.body_pos[self.moving_obstacle_id] - self.original_table_pos
#             if self.worker_torso_id != -1:
#                 self.relative_positions['worker_torso'] = model.body_pos[self.worker_torso_id] - self.original_table_pos
        
#     def randomize_table_position(self, x_range=(0.45, 0.65), y_range=(-0.15, 0.15), z_range=(0.42, 0.48)):
#         """테이블 위치를 무작위화하고 관련된 모든 오브젝트도 함께 이동 (로봇 작업공간 내)"""
#         if self.table_id == -1:
#             return np.array([0.5, 0, 0.45])  # 기본값 반환
        
#         max_attempts = 10
#         for attempt in range(max_attempts):
#             # 새로운 테이블 위치 생성 (더 보수적인 범위)
#             new_table_pos = np.array([
#                 np.random.uniform(*x_range),
#                 np.random.uniform(*y_range),
#                 np.random.uniform(*z_range)
#             ])
            
#             # 테이블 위치 업데이트
#             self.model.body_pos[self.table_id] = new_table_pos
            
#             # 테이블 위의 모든 오브젝트들도 상대 위치 유지하며 이동
#             if self.lap_base_id != -1 and 'lap_base' in self.relative_positions:
#                 self.model.body_pos[self.lap_base_id] = new_table_pos + self.relative_positions['lap_base']
#             if self.fillet_joint_id != -1 and 'fillet_joint' in self.relative_positions:
#                 self.model.body_pos[self.fillet_joint_id] = new_table_pos + self.relative_positions['fillet_joint']
#             if self.curved_pipe_id != -1 and 'curved_pipe' in self.relative_positions:
#                 self.model.body_pos[self.curved_pipe_id] = new_table_pos + self.relative_positions['curved_pipe']
#             if self.moving_obstacle_id != -1 and 'moving_obstacle' in self.relative_positions:
#                 self.model.body_pos[self.moving_obstacle_id] = new_table_pos + self.relative_positions['moving_obstacle']
#             if self.worker_torso_id != -1 and 'worker_torso' in self.relative_positions:
#                 self.model.body_pos[self.worker_torso_id] = new_table_pos + self.relative_positions['worker_torso']
            
#             # Forward kinematics 업데이트
#             mujoco.mj_forward(self.model, self.data)
            
#             # 모든 웨이포인트가 도달 가능한지 확인
#             all_reachable = all([
#                 self.check_reachability("lap_start"),
#                 self.check_reachability("lap_waypoint1"),
#                 self.check_reachability("lap_waypoint2"),
#                 self.check_reachability("lap_waypoint3"),
#                 self.check_reachability("lap_end")
#             ])
            
#             if all_reachable:
#                 print(f"✅ Valid environment found on attempt {attempt + 1}")
#                 return new_table_pos
#             else:
#                 print(f"⚠️  Attempt {attempt + 1}: Some waypoints unreachable, retrying...")
        
#         # 모든 시도 실패시 안전한 기본 위치로
#         print("🔄 Using safe default position")
#         safe_pos = np.array([0.55, 0.0, 0.45])
#         self.model.body_pos[self.table_id] = safe_pos
#         return safe_pos
    
#     def randomize_obstacles(self):
#         """장애물 위치를 무작위화"""
#         # 움직이는 장애물의 초기 위치 설정
#         obstacle_joint_ids = [
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y"),
#             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_z")
#         ]
        
#         for joint_id in obstacle_joint_ids:
#             if joint_id != -1:
#                 range_min = self.model.jnt_range[joint_id][0]
#                 range_max = self.model.jnt_range[joint_id][1]
#                 self.data.qpos[self.model.jnt_qposadr[joint_id]] = np.random.uniform(range_min, range_max)
    
#     def randomize_lap_joint_angle(self, angle_range=(-0.3, 0.3)):
#         """Lap joint의 초기 각도를 무작위화"""
#         lap_base_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lap_base")
#         if lap_base_joint != -1 and self.lap_base_id != -1:
#             angle = np.random.uniform(*angle_range)
#             self.model.body_quat[self.lap_base_id] = self._euler_to_quat(0, 0, angle)
#             return angle
#         return 0
    
#     def check_reachability(self, site_name, robot_reach=0.8):
#         """특정 사이트가 로봇의 도달 범위 내에 있는지 확인 (더 보수적)"""
#         site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
#         if site_id != -1:
#             site_pos = self.data.site_xpos[site_id]
#             # 로봇 베이스를 원점으로 가정하여 거리 계산
#             distance = np.linalg.norm(site_pos[:2])  # XY 평면에서의 거리만 고려
#             height = site_pos[2]  # Z 좌표
            
#             # 높이 제한 (0.2m ~ 0.7m)
#             height_ok = 0.2 <= height <= 0.7
#             # 수평 거리 제한
#             distance_ok = distance <= robot_reach
            
#             reachable = height_ok and distance_ok
#             if not reachable:
#                 print(f"❌ {site_name}: dist={distance:.3f}m, height={height:.3f}m (reach={robot_reach}m)")
            
#             return reachable
#         return False
    
#     def update_obstacles(self):
#         """장애물 애니메이션 업데이트"""
#         t = self.data.time
        
#         # 움직이는 장애물 애니메이션
#         obstacle_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x")
#         obstacle_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y")
        
#         if obstacle_x_id != -1:
#             addr = self.model.jnt_qposadr[obstacle_x_id]
#             self.data.qpos[addr] = 0.05 * np.sin(t * 0.5)  # 천천히 좌우 이동
            
#         if obstacle_y_id != -1:
#             addr = self.model.jnt_qposadr[obstacle_y_id]
#             self.data.qpos[addr] = 0.03 * np.cos(t * 0.7)  # 천천히 앞뒤 이동
    
#     def randomize_all(self):
#         """전체 환경을 무작위화"""
#         # 테이블 위치 무작위화
#         table_pos = self.randomize_table_position()
        
#         # Lap joint 각도 무작위화
#         lap_angle = self.randomize_lap_joint_angle()
        
#         # 장애물 무작위화
#         self.randomize_obstacles()
        
#         # Forward kinematics 업데이트
#         mujoco.mj_forward(self.model, self.data)
        
#         # 도달 가능성 확인
#         reachable = all([
#             self.check_reachability("lap_start"),
#             self.check_reachability("lap_waypoint1"),
#             self.check_reachability("lap_waypoint2"),
#             self.check_reachability("lap_waypoint3"),
#             self.check_reachability("lap_end")
#         ])
        
#         return {
#             'table_position': table_pos,
#             'lap_angle': lap_angle,
#             'all_waypoints_reachable': reachable
#         }
    
#     def _euler_to_quat(self, roll, pitch, yaw):
#         """Euler angles to quaternion conversion"""
#         cy = np.cos(yaw * 0.5)
#         sy = np.sin(yaw * 0.5)
#         cp = np.cos(pitch * 0.5)
#         sp = np.sin(pitch * 0.5)
#         cr = np.cos(roll * 0.5)
#         sr = np.sin(roll * 0.5)
        
#         w = cr * cp * cy + sr * sp * sy
#         x = sr * cp * cy - cr * sp * sy
#         y = cr * sp * cy + sr * cp * sy
#         z = cr * cp * sy - sr * sp * cy
        
#         return np.array([w, x, y, z])

# class WeldingMujocoROSBridge(MujocoROSBridge):
#     """용접 환경을 위한 확장된 MujocoROSBridge (충돌 회피 포함)"""
    
#     def __init__(self, robot_info, camera_info, robot_controller):
#         super().__init__(robot_info, camera_info, robot_controller)
        
#         # 환경 무작위화 초기화
#         self.randomizer = WeldingEnvironmentRandomizer(self.model, self.data)
        
#         # 충돌 회피 시스템 초기화
#         self.collision_avoidance = MuJoCoCollisionAvoidance(
#             self.model, self.data, robot_controller
#         )
        
#         # 환경 업데이트 관련 변수
#         self.environment_initialized = False  # 환경 초기화 여부
#         self.obstacle_animation_enabled = True  # 장애물 애니메이션만 유지
#         self.collision_monitoring_enabled = True  # 충돌 모니터링
        
#         # 충돌 통계
#         self.collision_stats = {
#             'total_checks': 0,
#             'collisions_detected': 0,
#             'last_collision_time': 0
#         }
        
#         # 초기 환경 설정 (시뮬레이션 시작 시 한 번만)
#         result = self.randomizer.randomize_all()
#         self.environment_initialized = True
        
#         self.get_logger().info(f"🎯 Welding environment initialized once:")
#         self.get_logger().info(f"   Table position: {result['table_position']}")
#         self.get_logger().info(f"   Lap angle: {result['lap_angle']:.3f} rad")
#         self.get_logger().info(f"   All waypoints reachable: {result['all_waypoints_reachable']}")
#         self.get_logger().info(f"🔒 Environment positions are now FIXED for this simulation session")
#         self.get_logger().info(f"🛡️ Collision avoidance system ready")
    
#     def robot_control(self):
#         """원래 robot_control 메서드를 오버라이드하여 용접 환경 업데이트 추가"""
#         self.ctrl_step = 0
#         sync_step = 30  # every 30 ctrl_steps

#         try:
#             while rclpy.ok() and self.running:            
#                 with self.lock:
#                     start_time = time.perf_counter()                        

#                     # 원래 시뮬레이션 스텝
#                     mujoco.mj_step(self.model, self.data)

#                     # 용접 환경 업데이트 (장애물 애니메이션)
#                     self.update_welding_environment()

#                     self.rc.updateModel(self.data, self.ctrl_step)
                    
#                     # -------------------- ADD Controller ---------------------------- #
#                     rclpy.spin_once(self.rc, timeout_sec=0.0001) # for scene monitor
#                     rclpy.spin_once(self, timeout_sec=0.0001) # for robot controller
#                     self.data.ctrl[:self.ctrl_dof] = self.rc.compute()   

#                     # --- publish joint positions ---
#                     js = self.get_joint_state_message()
#                     self.joint_state_pub.publish(js)

#                     # ---------------------------------------------------------------- #
#                     self.ctrl_step += 1

#                 self.time_sync(self.dt, start_time, False)
            
#         except KeyboardInterrupt:
#             self.get_logger().info("\nSimulation interrupted. Closing robot controller ...")
#             self.rc.destroy_node()

#     def get_joint_state_message(self):
#         """JointState 메시지 생성"""
#         from sensor_msgs.msg import JointState
        
#         js = JointState()
#         js.header.stamp = self.get_clock().now().to_msg()
#         js.name = self.joint_names
#         js.position = [
#             float(self.data.qpos[self.model.joint(j).qposadr])
#             for j in self.joint_names
#         ]
#         return js
    
#     def update_welding_environment(self):
#         """용접 환경 업데이트 - 장애물 애니메이션 및 안전한 충돌 모니터링"""
#         # 장애물 애니메이션만 유지 (매 스텝)
#         if self.obstacle_animation_enabled:
#             self.randomizer.update_obstacles()
        
#         # 충돌 모니터링 (덜 자주, 안전하게)
#         if self.collision_monitoring_enabled and self.ctrl_step % 500 == 0:  # 0.5초마다로 줄임
#             self.monitor_collisions()
    
#     def monitor_collisions(self):
#         """실시간 충돌 모니터링 (안전한 버전)"""
#         try:
#             self.collision_stats['total_checks'] += 1
            
#             # 안전한 충돌 검사
#             collision_detected, pairs = self.safe_collision_check()
            
#             if collision_detected:
#                 self.collision_stats['collisions_detected'] += 1
#                 self.collision_stats['last_collision_time'] = self.data.time
                
#                 self.get_logger().warn(f"⚠️  COLLISION DETECTED at t={self.data.time:.2f}s")
#                 for i, (geom1, geom2, pos) in enumerate(pairs):
#                     self.get_logger().warn(f"   {i+1}. {geom1} ↔ {geom2} at {pos}")
            
#             # 주기적 통계 출력 (1000번마다)
#             if self.collision_stats['total_checks'] % 1000 == 0:
#                 detection_rate = self.collision_stats['collisions_detected'] / self.collision_stats['total_checks'] * 100
#                 self.get_logger().info(f"📊 Collision Statistics:")
#                 self.get_logger().info(f"   Checks: {self.collision_stats['total_checks']}")
#                 self.get_logger().info(f"   Detections: {self.collision_stats['collisions_detected']}")
#                 self.get_logger().info(f"   Rate: {detection_rate:.2f}%")
                
#         except Exception as e:
#             self.get_logger().error(f"Collision monitoring error: {e}")
#             # 충돌 모니터링 비활성화
#             self.collision_monitoring_enabled = False
    
#     def safe_collision_check(self):
#         """안전한 충돌 검사"""
#         try:
#             # MuJoCo 충돌 계산 실행
#             mujoco.mj_collision(self.model, self.data)
            
#             collision_detected = self.data.ncon > 0
#             collision_pairs = []
            
#             if collision_detected and self.data.ncon < 100:  # 너무 많은 접촉은 무시
#                 for i in range(min(self.data.ncon, 10)):  # 최대 10개만 처리
#                     try:
#                         contact = self.data.contact[i]
                        
#                         # geom ID 유효성 검사
#                         if (0 <= contact.geom1 < self.model.ngeom and 
#                             0 <= contact.geom2 < self.model.ngeom):
                            
#                             geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
#                             geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
                            
#                             # 로봇-환경 충돌만 관심
#                             if self.is_robot_environment_collision(geom1_name, geom2_name):
#                                 collision_pairs.append((geom1_name, geom2_name, contact.pos.copy()))
                                
#                     except Exception as e:
#                         # 개별 접촉 처리 오류는 무시하고 계속
#                         continue
            
#             return len(collision_pairs) > 0, collision_pairs
            
#         except Exception as e:
#             # 충돌 검사 자체에 오류가 있으면 안전하게 처리
#             return False, []
    
#     def plan_safe_path(self, current_ee_pos, target_ee_pos):
#         """안전한 경로 계획"""
#         current_joints = self.data.qpos[:7].copy()
#         return self.collision_avoidance.plan_collision_free_path(
#             current_ee_pos, target_ee_pos, current_joints
#         )
    
#     def get_collision_status(self):
#         """현재 충돌 상태 반환"""
#         return self.collision_avoidance.get_collision_statistics()
    
#     def disable_obstacle_animation(self):
#         """장애물 애니메이션 비활성화 (완전 정적 환경)"""
#         self.obstacle_animation_enabled = False
#         self.get_logger().info("🔒 Obstacle animation disabled - fully static environment")
    
#     def enable_obstacle_animation(self):
#         """장애물 애니메이션 활성화"""
#         self.obstacle_animation_enabled = True
#         self.get_logger().info("🔓 Obstacle animation enabled")
    
#     def manual_randomize_environment(self):
#         """수동으로 환경 재무작위화 (필요시 호출)"""
#         if self.environment_initialized:
#             result = self.randomizer.randomize_all()
#             self.get_logger().info(f"🔄 Environment manually randomized:")
#             self.get_logger().info(f"   Table position: {result['table_position']}")
#             self.get_logger().info(f"   All waypoints reachable: {result['all_waypoints_reachable']}")

# def main():
#     current_dir = os.path.dirname(os.path.realpath(__file__))
    
#     # ROS2 초기화
#     rclpy.init()
    
#     # 파일 경로 설정
#     xml_path = os.path.join(current_dir, '../robots', "welding_scene2.xml")
#     urdf_path = os.path.join(current_dir, '../robots', 'fr3/fr3_hand.urdf')
    
#     # 파일 존재 확인
#     if not os.path.exists(xml_path):
#         print(f"❌ XML file not found: {xml_path}")
#         return
#     if not os.path.exists(urdf_path):
#         print(f"❌ URDF file not found: {urdf_path}")
#         return
    
#     try:
#         print("🤖 Initializing Welding Robot Controller...")
        
#         # Fr3 컨트롤러 초기화
#         rc = Fr3Controller(urdf_path)
        
#         # 로봇 및 카메라 정보 설정
#         robot_info = [xml_path, urdf_path, 1000]  # [xml_path, urdf_path, hz]
#         camera_info = ['hand_eye', 320, 240, 30]  # [camera_name, width, height, fps]
        
#         print("🌉 Setting up Welding MuJoCo-ROS Bridge...")
        
#         # 용접 환경용 MuJoCo-ROS 브리지 초기화
#         bridge = WeldingMujocoROSBridge(robot_info, camera_info, rc)
        
#         print("🚀 Starting Welding Simulation with ROS Bridge...")
#         print("💡 Available services:")
#         print("   - /task_move_srv")
#         print("   - /get_site_position")
#         print("   - /get_site_orientation")
#         print("🔧 You can now run the waypoint client!")
#         print("   ros2 run dm_task_manager multi_pose_task_client_waypoints")
#         print("🎮 Environment features:")
#         print("   - Environment randomized ONCE at startup")
#         print("   - Moving obstacle animation (realistic)")
#         print("   - Fixed table and static obstacle positions")
#         print("   - Real-time waypoint reachability checking")
#         print("🛡️ Collision avoidance features:")
#         print("   - Real-time collision monitoring")
#         print("   - Smart path planning with obstacle avoidance")
#         print("   - Multiple avoidance strategies (lift, side-step, curve)")
#         print("   - Visual collision feedback in terminal")
        
#         # 브리지 실행 (ROS 서비스와 시뮬레이션 동시 실행)
#         bridge.run()
        
#     except KeyboardInterrupt:
#         print("\n🛑 Shutting down welding simulation...")
#     except Exception as e:
#         print(f"❌ Error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         # 정리
#         if 'bridge' in locals():
#             bridge.destroy_node()
#         rclpy.shutdown()
#         print("✅ Welding simulation terminated.")

# if __name__ == "__main__":    
#     main()
    

# # # /home/minjun/wr_ws/src/welding_robot/dm_ros/dm_ros/welding_free2.py
# # import rclpy
# # import os
# # import numpy as np
# # import mujoco
# # import threading
# # import time
# # from prc import Fr3Controller
# # from .utils.multi_thread import MujocoROSBridge
# # from .utils.collision_avoidance import MuJoCoCollisionAvoidance, integrate_collision_avoidance_to_bridge

# # class WeldingEnvironmentRandomizer:
# #     def __init__(self, model, data):
# #         self.model = model
# #         self.data = data
        
# #         # 테이블과 관련 body들의 ID 저장
# #         self.table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
# #         self.lap_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lap_base")
# #         self.fillet_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fillet_joint_base")
# #         self.curved_pipe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "curved_pipe")
# #         self.moving_obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_obstacle")
# #         self.worker_torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "worker_torso")
        
# #         # 원래 위치 저장 (상대 위치 계산용)
# #         if self.table_id != -1:
# #             self.original_table_pos = model.body_pos[self.table_id].copy()
# #             self.relative_positions = {}
            
# #             if self.lap_base_id != -1:
# #                 self.relative_positions['lap_base'] = model.body_pos[self.lap_base_id] - self.original_table_pos
# #             if self.fillet_joint_id != -1:
# #                 self.relative_positions['fillet_joint'] = model.body_pos[self.fillet_joint_id] - self.original_table_pos
# #             if self.curved_pipe_id != -1:
# #                 self.relative_positions['curved_pipe'] = model.body_pos[self.curved_pipe_id] - self.original_table_pos
# #             if self.moving_obstacle_id != -1:
# #                 self.relative_positions['moving_obstacle'] = model.body_pos[self.moving_obstacle_id] - self.original_table_pos
# #             if self.worker_torso_id != -1:
# #                 self.relative_positions['worker_torso'] = model.body_pos[self.worker_torso_id] - self.original_table_pos
        
# #     def randomize_table_position(self, x_range=(0.45, 0.65), y_range=(-0.15, 0.15), z_range=(0.42, 0.48)):
# #         """테이블 위치를 무작위화하고 관련된 모든 오브젝트도 함께 이동 (로봇 작업공간 내)"""
# #         if self.table_id == -1:
# #             return np.array([0.5, 0, 0.45])  # 기본값 반환
        
# #         max_attempts = 10
# #         for attempt in range(max_attempts):
# #             # 새로운 테이블 위치 생성 (더 보수적인 범위)
# #             new_table_pos = np.array([
# #                 np.random.uniform(*x_range),
# #                 np.random.uniform(*y_range),
# #                 np.random.uniform(*z_range)
# #             ])
            
# #             # 테이블 위치 업데이트
# #             self.model.body_pos[self.table_id] = new_table_pos
            
# #             # 테이블 위의 모든 오브젝트들도 상대 위치 유지하며 이동
# #             if self.lap_base_id != -1 and 'lap_base' in self.relative_positions:
# #                 self.model.body_pos[self.lap_base_id] = new_table_pos + self.relative_positions['lap_base']
# #             if self.fillet_joint_id != -1 and 'fillet_joint' in self.relative_positions:
# #                 self.model.body_pos[self.fillet_joint_id] = new_table_pos + self.relative_positions['fillet_joint']
# #             if self.curved_pipe_id != -1 and 'curved_pipe' in self.relative_positions:
# #                 self.model.body_pos[self.curved_pipe_id] = new_table_pos + self.relative_positions['curved_pipe']
# #             if self.moving_obstacle_id != -1 and 'moving_obstacle' in self.relative_positions:
# #                 self.model.body_pos[self.moving_obstacle_id] = new_table_pos + self.relative_positions['moving_obstacle']
# #             if self.worker_torso_id != -1 and 'worker_torso' in self.relative_positions:
# #                 self.model.body_pos[self.worker_torso_id] = new_table_pos + self.relative_positions['worker_torso']
            
# #             # Forward kinematics 업데이트
# #             mujoco.mj_forward(self.model, self.data)
            
# #             # 모든 웨이포인트가 도달 가능한지 확인
# #             all_reachable = all([
# #                 self.check_reachability("lap_start"),
# #                 self.check_reachability("lap_waypoint1"),
# #                 self.check_reachability("lap_waypoint2"),
# #                 self.check_reachability("lap_waypoint3"),
# #                 self.check_reachability("lap_end")
# #             ])
            
# #             if all_reachable:
# #                 print(f"✅ Valid environment found on attempt {attempt + 1}")
# #                 return new_table_pos
# #             else:
# #                 print(f"⚠️  Attempt {attempt + 1}: Some waypoints unreachable, retrying...")
        
# #         # 모든 시도 실패시 안전한 기본 위치로
# #         print("🔄 Using safe default position")
# #         safe_pos = np.array([0.55, 0.0, 0.45])
# #         self.model.body_pos[self.table_id] = safe_pos
# #         return safe_pos
    
# #     def randomize_obstacles(self):
# #         """장애물 위치를 무작위화"""
# #         # 움직이는 장애물의 초기 위치 설정
# #         obstacle_joint_ids = [
# #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x"),
# #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y"),
# #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_z")
# #         ]
        
# #         for joint_id in obstacle_joint_ids:
# #             if joint_id != -1:
# #                 range_min = self.model.jnt_range[joint_id][0]
# #                 range_max = self.model.jnt_range[joint_id][1]
# #                 self.data.qpos[self.model.jnt_qposadr[joint_id]] = np.random.uniform(range_min, range_max)
    
# #     def randomize_lap_joint_angle(self, angle_range=(-0.3, 0.3)):
# #         """Lap joint의 초기 각도를 무작위화"""
# #         lap_base_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lap_base")
# #         if lap_base_joint != -1 and self.lap_base_id != -1:
# #             angle = np.random.uniform(*angle_range)
# #             self.model.body_quat[self.lap_base_id] = self._euler_to_quat(0, 0, angle)
# #             return angle
# #         return 0
    
# #     def check_reachability(self, site_name, robot_reach=0.8):
# #         """특정 사이트가 로봇의 도달 범위 내에 있는지 확인 (더 보수적)"""
# #         site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
# #         if site_id != -1:
# #             site_pos = self.data.site_xpos[site_id]
# #             # 로봇 베이스를 원점으로 가정하여 거리 계산
# #             distance = np.linalg.norm(site_pos[:2])  # XY 평면에서의 거리만 고려
# #             height = site_pos[2]  # Z 좌표
            
# #             # 높이 제한 (0.2m ~ 0.7m)
# #             height_ok = 0.2 <= height <= 0.7
# #             # 수평 거리 제한
# #             distance_ok = distance <= robot_reach
            
# #             reachable = height_ok and distance_ok
# #             if not reachable:
# #                 print(f"❌ {site_name}: dist={distance:.3f}m, height={height:.3f}m (reach={robot_reach}m)")
            
# #             return reachable
# #         return False
    
# #     def update_obstacles(self):
# #         """장애물 애니메이션 업데이트"""
# #         t = self.data.time
        
# #         # 움직이는 장애물 애니메이션
# #         obstacle_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x")
# #         obstacle_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y")
        
# #         if obstacle_x_id != -1:
# #             addr = self.model.jnt_qposadr[obstacle_x_id]
# #             self.data.qpos[addr] = 0.05 * np.sin(t * 0.5)  # 천천히 좌우 이동
            
# #         if obstacle_y_id != -1:
# #             addr = self.model.jnt_qposadr[obstacle_y_id]
# #             self.data.qpos[addr] = 0.03 * np.cos(t * 0.7)  # 천천히 앞뒤 이동
    
# #     def randomize_all(self):
# #         """전체 환경을 무작위화"""
# #         # 테이블 위치 무작위화
# #         table_pos = self.randomize_table_position()
        
# #         # Lap joint 각도 무작위화
# #         lap_angle = self.randomize_lap_joint_angle()
        
# #         # 장애물 무작위화
# #         self.randomize_obstacles()
        
# #         # Forward kinematics 업데이트
# #         mujoco.mj_forward(self.model, self.data)
        
# #         # 도달 가능성 확인
# #         reachable = all([
# #             self.check_reachability("lap_start"),
# #             self.check_reachability("lap_waypoint1"),
# #             self.check_reachability("lap_waypoint2"),
# #             self.check_reachability("lap_waypoint3"),
# #             self.check_reachability("lap_end")
# #         ])
        
# #         return {
# #             'table_position': table_pos,
# #             'lap_angle': lap_angle,
# #             'all_waypoints_reachable': reachable
# #         }
    
# #     def _euler_to_quat(self, roll, pitch, yaw):
# #         """Euler angles to quaternion conversion"""
# #         cy = np.cos(yaw * 0.5)
# #         sy = np.sin(yaw * 0.5)
# #         cp = np.cos(pitch * 0.5)
# #         sp = np.sin(pitch * 0.5)
# #         cr = np.cos(roll * 0.5)
# #         sr = np.sin(roll * 0.5)
        
# #         w = cr * cp * cy + sr * sp * sy
# #         x = sr * cp * cy - cr * sp * sy
# #         y = cr * sp * cy + sr * cp * sy
# #         z = cr * cp * sy - sr * sp * cy
        
# #         return np.array([w, x, y, z])

# # class WeldingMujocoROSBridge(MujocoROSBridge):
# #     """용접 환경을 위한 확장된 MujocoROSBridge (충돌 회피 포함)"""
    
# #     def __init__(self, robot_info, camera_info, robot_controller):
# #         super().__init__(robot_info, camera_info, robot_controller)
        
# #         # 환경 무작위화 초기화
# #         self.randomizer = WeldingEnvironmentRandomizer(self.model, self.data)
        
# #         # 충돌 회피 시스템 초기화
# #         self.collision_avoidance = MuJoCoCollisionAvoidance(
# #             self.model, self.data, robot_controller
# #         )
        
# #         # 환경 업데이트 관련 변수
# #         self.environment_initialized = False  # 환경 초기화 여부
# #         self.obstacle_animation_enabled = True  # 장애물 애니메이션만 유지
# #         self.collision_monitoring_enabled = True  # 충돌 모니터링
        
# #         # 충돌 통계
# #         self.collision_stats = {
# #             'total_checks': 0,
# #             'collisions_detected': 0,
# #             'last_collision_time': 0
# #         }
        
# #         # 초기 환경 설정 (시뮬레이션 시작 시 한 번만)
# #         result = self.randomizer.randomize_all()
# #         self.environment_initialized = True
        
# #         self.get_logger().info(f"🎯 Welding environment initialized once:")
# #         self.get_logger().info(f"   Table position: {result['table_position']}")
# #         self.get_logger().info(f"   Lap angle: {result['lap_angle']:.3f} rad")
# #         self.get_logger().info(f"   All waypoints reachable: {result['all_waypoints_reachable']}")
# #         self.get_logger().info(f"🔒 Environment positions are now FIXED for this simulation session")
# #         self.get_logger().info(f"🛡️ Collision avoidance system ready")
    
# #     def robot_control(self):
# #         """원래 robot_control 메서드를 오버라이드하여 용접 환경 업데이트 추가"""
# #         self.ctrl_step = 0
# #         sync_step = 30  # every 30 ctrl_steps

# #         try:
# #             while rclpy.ok() and self.running:            
# #                 with self.lock:
# #                     start_time = time.perf_counter()                        

# #                     # 원래 시뮬레이션 스텝
# #                     mujoco.mj_step(self.model, self.data)

# #                     # 용접 환경 업데이트 (장애물 애니메이션)
# #                     self.update_welding_environment()

# #                     self.rc.updateModel(self.data, self.ctrl_step)
                    
# #                     # -------------------- ADD Controller ---------------------------- #
# #                     rclpy.spin_once(self.rc, timeout_sec=0.0001) # for scene monitor
# #                     rclpy.spin_once(self, timeout_sec=0.0001) # for robot controller
# #                     self.data.ctrl[:self.ctrl_dof] = self.rc.compute()   

# #                     # --- publish joint positions ---
# #                     js = self.get_joint_state_message()
# #                     self.joint_state_pub.publish(js)

# #                     # ---------------------------------------------------------------- #
# #                     self.ctrl_step += 1

# #                 self.time_sync(self.dt, start_time, False)
            
# #         except KeyboardInterrupt:
# #             self.get_logger().info("\nSimulation interrupted. Closing robot controller ...")
# #             self.rc.destroy_node()

# #     def get_joint_state_message(self):
# #         """JointState 메시지 생성"""
# #         from sensor_msgs.msg import JointState
        
# #         js = JointState()
# #         js.header.stamp = self.get_clock().now().to_msg()
# #         js.name = self.joint_names
# #         js.position = [
# #             float(self.data.qpos[self.model.joint(j).qposadr])
# #             for j in self.joint_names
# #         ]
# #         return js
    
# #     def update_welding_environment(self):
# #         """용접 환경 업데이트 - 장애물 애니메이션 및 충돌 모니터링"""
# #         # 장애물 애니메이션만 유지 (매 스텝)
# #         if self.obstacle_animation_enabled:
# #             self.randomizer.update_obstacles()
        
# #         # 충돌 모니터링 (주기적으로)
# #         if self.collision_monitoring_enabled and self.ctrl_step % 100 == 0:  # 0.1초마다
# #             self.monitor_collisions()
    
# #     def monitor_collisions(self):
# #         """실시간 충돌 모니터링"""
# #         self.collision_stats['total_checks'] += 1
        
# #         collision_detected, pairs = self.collision_avoidance.check_robot_collision()
        
# #         if collision_detected:
# #             self.collision_stats['collisions_detected'] += 1
# #             self.collision_stats['last_collision_time'] = self.data.time
            
# #             self.get_logger().warn(f"⚠️  COLLISION DETECTED at t={self.data.time:.2f}s")
# #             for i, (geom1, geom2, pos) in enumerate(pairs):
# #                 self.get_logger().warn(f"   {i+1}. {geom1} ↔ {geom2} at {pos}")
        
# #         # 주기적 통계 출력 (10초마다)
# #         if self.collision_stats['total_checks'] % 1000 == 0:  # 100초마다
# #             detection_rate = self.collision_stats['collisions_detected'] / self.collision_stats['total_checks'] * 100
# #             self.get_logger().info(f"📊 Collision Statistics:")
# #             self.get_logger().info(f"   Checks: {self.collision_stats['total_checks']}")
# #             self.get_logger().info(f"   Detections: {self.collision_stats['collisions_detected']}")
# #             self.get_logger().info(f"   Rate: {detection_rate:.2f}%")
    
# #     def plan_safe_path(self, current_ee_pos, target_ee_pos):
# #         """안전한 경로 계획"""
# #         current_joints = self.data.qpos[:7].copy()
# #         return self.collision_avoidance.plan_collision_free_path(
# #             current_ee_pos, target_ee_pos, current_joints
# #         )
    
# #     def get_collision_status(self):
# #         """현재 충돌 상태 반환"""
# #         return self.collision_avoidance.get_collision_statistics()
    
# #     def disable_obstacle_animation(self):
# #         """장애물 애니메이션 비활성화 (완전 정적 환경)"""
# #         self.obstacle_animation_enabled = False
# #         self.get_logger().info("🔒 Obstacle animation disabled - fully static environment")
    
# #     def enable_obstacle_animation(self):
# #         """장애물 애니메이션 활성화"""
# #         self.obstacle_animation_enabled = True
# #         self.get_logger().info("🔓 Obstacle animation enabled")
    
# #     def manual_randomize_environment(self):
# #         """수동으로 환경 재무작위화 (필요시 호출)"""
# #         if self.environment_initialized:
# #             result = self.randomizer.randomize_all()
# #             self.get_logger().info(f"🔄 Environment manually randomized:")
# #             self.get_logger().info(f"   Table position: {result['table_position']}")
# #             self.get_logger().info(f"   All waypoints reachable: {result['all_waypoints_reachable']}")

# # def main():
# #     current_dir = os.path.dirname(os.path.realpath(__file__))
    
# #     # ROS2 초기화
# #     rclpy.init()
    
# #     # 파일 경로 설정
# #     xml_path = os.path.join(current_dir, '../robots', "welding_scene2.xml")
# #     urdf_path = os.path.join(current_dir, '../robots', 'fr3/fr3_hand.urdf')
    
# #     # 파일 존재 확인
# #     if not os.path.exists(xml_path):
# #         print(f"❌ XML file not found: {xml_path}")
# #         return
# #     if not os.path.exists(urdf_path):
# #         print(f"❌ URDF file not found: {urdf_path}")
# #         return
    
# #     try:
# #         print("🤖 Initializing Welding Robot Controller...")
        
# #         # Fr3 컨트롤러 초기화
# #         rc = Fr3Controller(urdf_path)
        
# #         # 로봇 및 카메라 정보 설정
# #         robot_info = [xml_path, urdf_path, 1000]  # [xml_path, urdf_path, hz]
# #         camera_info = ['hand_eye', 320, 240, 30]  # [camera_name, width, height, fps]
        
# #         print("🌉 Setting up Welding MuJoCo-ROS Bridge...")
        
# #         # 용접 환경용 MuJoCo-ROS 브리지 초기화
# #         bridge = WeldingMujocoROSBridge(robot_info, camera_info, rc)
        
# #         print("🚀 Starting Welding Simulation with ROS Bridge...")
# #         print("💡 Available services:")
# #         print("   - /task_move_srv")
# #         print("   - /get_site_position")
# #         print("   - /get_site_orientation")
# #         print("🔧 You can now run the waypoint client!")
# #         print("   ros2 run dm_task_manager multi_pose_task_client_waypoints")
# #         print("🎮 Environment features:")
# #         print("   - Environment randomized ONCE at startup")
# #         print("   - Moving obstacle animation (realistic)")
# #         print("   - Fixed table and static obstacle positions")
# #         print("   - Real-time waypoint reachability checking")
# #         print("🛡️ Collision avoidance features:")
# #         print("   - Real-time collision monitoring")
# #         print("   - Smart path planning with obstacle avoidance")
# #         print("   - Multiple avoidance strategies (lift, side-step, curve)")
# #         print("   - Visual collision feedback in terminal")
        
# #         # 브리지 실행 (ROS 서비스와 시뮬레이션 동시 실행)
# #         bridge.run()
        
# #     except KeyboardInterrupt:
# #         print("\n🛑 Shutting down welding simulation...")
# #     except Exception as e:
# #         print(f"❌ Error: {str(e)}")
# #         import traceback
# #         traceback.print_exc()
# #     finally:
# #         # 정리
# #         if 'bridge' in locals():
# #             bridge.destroy_node()
# #         rclpy.shutdown()
# #         print("✅ Welding simulation terminated.")

# # if __name__ == "__main__":    
# #     main()


# # # # /home/minjun/wr_ws/src/welding_robot/dm_ros/dm_ros/welding_free2.py
# # # import rclpy
# # # import os
# # # import numpy as np
# # # import mujoco
# # # import threading
# # # import time
# # # from prc import Fr3Controller
# # # from .utils.multi_thread import MujocoROSBridge

# # # class WeldingEnvironmentRandomizer:
# # #     def __init__(self, model, data):
# # #         self.model = model
# # #         self.data = data
        
# # #         # 테이블과 관련 body들의 ID 저장
# # #         self.table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
# # #         self.lap_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lap_base")
# # #         self.fillet_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fillet_joint_base")
# # #         self.curved_pipe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "curved_pipe")
# # #         self.moving_obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_obstacle")
# # #         self.worker_torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "worker_torso")
        
# # #         # 원래 위치 저장 (상대 위치 계산용)
# # #         if self.table_id != -1:
# # #             self.original_table_pos = model.body_pos[self.table_id].copy()
# # #             self.relative_positions = {}
            
# # #             if self.lap_base_id != -1:
# # #                 self.relative_positions['lap_base'] = model.body_pos[self.lap_base_id] - self.original_table_pos
# # #             if self.fillet_joint_id != -1:
# # #                 self.relative_positions['fillet_joint'] = model.body_pos[self.fillet_joint_id] - self.original_table_pos
# # #             if self.curved_pipe_id != -1:
# # #                 self.relative_positions['curved_pipe'] = model.body_pos[self.curved_pipe_id] - self.original_table_pos
# # #             if self.moving_obstacle_id != -1:
# # #                 self.relative_positions['moving_obstacle'] = model.body_pos[self.moving_obstacle_id] - self.original_table_pos
# # #             if self.worker_torso_id != -1:
# # #                 self.relative_positions['worker_torso'] = model.body_pos[self.worker_torso_id] - self.original_table_pos
        
# # #     def randomize_table_position(self, x_range=(0.45, 0.65), y_range=(-0.15, 0.15), z_range=(0.42, 0.48)):
# # #         """테이블 위치를 무작위화하고 관련된 모든 오브젝트도 함께 이동 (로봇 작업공간 내)"""
# # #         if self.table_id == -1:
# # #             return np.array([0.5, 0, 0.45])  # 기본값 반환
        
# # #         max_attempts = 10
# # #         for attempt in range(max_attempts):
# # #             # 새로운 테이블 위치 생성 (더 보수적인 범위)
# # #             new_table_pos = np.array([
# # #                 np.random.uniform(*x_range),
# # #                 np.random.uniform(*y_range),
# # #                 np.random.uniform(*z_range)
# # #             ])
            
# # #             # 테이블 위치 업데이트
# # #             self.model.body_pos[self.table_id] = new_table_pos
            
# # #             # 테이블 위의 모든 오브젝트들도 상대 위치 유지하며 이동
# # #             if self.lap_base_id != -1 and 'lap_base' in self.relative_positions:
# # #                 self.model.body_pos[self.lap_base_id] = new_table_pos + self.relative_positions['lap_base']
# # #             if self.fillet_joint_id != -1 and 'fillet_joint' in self.relative_positions:
# # #                 self.model.body_pos[self.fillet_joint_id] = new_table_pos + self.relative_positions['fillet_joint']
# # #             if self.curved_pipe_id != -1 and 'curved_pipe' in self.relative_positions:
# # #                 self.model.body_pos[self.curved_pipe_id] = new_table_pos + self.relative_positions['curved_pipe']
# # #             if self.moving_obstacle_id != -1 and 'moving_obstacle' in self.relative_positions:
# # #                 self.model.body_pos[self.moving_obstacle_id] = new_table_pos + self.relative_positions['moving_obstacle']
# # #             if self.worker_torso_id != -1 and 'worker_torso' in self.relative_positions:
# # #                 self.model.body_pos[self.worker_torso_id] = new_table_pos + self.relative_positions['worker_torso']
            
# # #             # Forward kinematics 업데이트
# # #             mujoco.mj_forward(self.model, self.data)
            
# # #             # 모든 웨이포인트가 도달 가능한지 확인
# # #             all_reachable = all([
# # #                 self.check_reachability("lap_start"),
# # #                 self.check_reachability("lap_waypoint1"),
# # #                 self.check_reachability("lap_waypoint2"),
# # #                 self.check_reachability("lap_waypoint3"),
# # #                 self.check_reachability("lap_end")
# # #             ])
            
# # #             if all_reachable:
# # #                 print(f"✅ Valid environment found on attempt {attempt + 1}")
# # #                 return new_table_pos
# # #             else:
# # #                 print(f"⚠️  Attempt {attempt + 1}: Some waypoints unreachable, retrying...")
        
# # #         # 모든 시도 실패시 안전한 기본 위치로
# # #         print("🔄 Using safe default position")
# # #         safe_pos = np.array([0.55, 0.0, 0.45])
# # #         self.model.body_pos[self.table_id] = safe_pos
# # #         return safe_pos
    
# # #     def randomize_obstacles(self):
# # #         """장애물 위치를 무작위화"""
# # #         # 움직이는 장애물의 초기 위치 설정
# # #         obstacle_joint_ids = [
# # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x"),
# # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y"),
# # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_z")
# # #         ]
        
# # #         for joint_id in obstacle_joint_ids:
# # #             if joint_id != -1:
# # #                 range_min = self.model.jnt_range[joint_id][0]
# # #                 range_max = self.model.jnt_range[joint_id][1]
# # #                 self.data.qpos[self.model.jnt_qposadr[joint_id]] = np.random.uniform(range_min, range_max)
    
# # #     def randomize_lap_joint_angle(self, angle_range=(-0.3, 0.3)):
# # #         """Lap joint의 초기 각도를 무작위화"""
# # #         lap_base_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lap_base")
# # #         if lap_base_joint != -1 and self.lap_base_id != -1:
# # #             angle = np.random.uniform(*angle_range)
# # #             self.model.body_quat[self.lap_base_id] = self._euler_to_quat(0, 0, angle)
# # #             return angle
# # #         return 0
    
# # #     def check_reachability(self, site_name, robot_reach=0.8):
# # #         """특정 사이트가 로봇의 도달 범위 내에 있는지 확인 (더 보수적)"""
# # #         site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
# # #         if site_id != -1:
# # #             site_pos = self.data.site_xpos[site_id]
# # #             # 로봇 베이스를 원점으로 가정하여 거리 계산
# # #             distance = np.linalg.norm(site_pos[:2])  # XY 평면에서의 거리만 고려
# # #             height = site_pos[2]  # Z 좌표
            
# # #             # 높이 제한 (0.2m ~ 0.7m)
# # #             height_ok = 0.2 <= height <= 0.7
# # #             # 수평 거리 제한
# # #             distance_ok = distance <= robot_reach
            
# # #             reachable = height_ok and distance_ok
# # #             if not reachable:
# # #                 print(f"❌ {site_name}: dist={distance:.3f}m, height={height:.3f}m (reach={robot_reach}m)")
            
# # #             return reachable
# # #         return False
    
# # #     def update_obstacles(self):
# # #         """장애물 애니메이션 업데이트"""
# # #         t = self.data.time
        
# # #         # 움직이는 장애물 애니메이션
# # #         obstacle_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x")
# # #         obstacle_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y")
        
# # #         if obstacle_x_id != -1:
# # #             addr = self.model.jnt_qposadr[obstacle_x_id]
# # #             self.data.qpos[addr] = 0.05 * np.sin(t * 0.5)  # 천천히 좌우 이동
            
# # #         if obstacle_y_id != -1:
# # #             addr = self.model.jnt_qposadr[obstacle_y_id]
# # #             self.data.qpos[addr] = 0.03 * np.cos(t * 0.7)  # 천천히 앞뒤 이동
    
# # #     def randomize_all(self):
# # #         """전체 환경을 무작위화"""
# # #         # 테이블 위치 무작위화
# # #         table_pos = self.randomize_table_position()
        
# # #         # Lap joint 각도 무작위화
# # #         lap_angle = self.randomize_lap_joint_angle()
        
# # #         # 장애물 무작위화
# # #         self.randomize_obstacles()
        
# # #         # Forward kinematics 업데이트
# # #         mujoco.mj_forward(self.model, self.data)
        
# # #         # 도달 가능성 확인
# # #         reachable = all([
# # #             self.check_reachability("lap_start"),
# # #             self.check_reachability("lap_waypoint1"),
# # #             self.check_reachability("lap_waypoint2"),
# # #             self.check_reachability("lap_waypoint3"),
# # #             self.check_reachability("lap_end")
# # #         ])
        
# # #         return {
# # #             'table_position': table_pos,
# # #             'lap_angle': lap_angle,
# # #             'all_waypoints_reachable': reachable
# # #         }
    
# # #     def _euler_to_quat(self, roll, pitch, yaw):
# # #         """Euler angles to quaternion conversion"""
# # #         cy = np.cos(yaw * 0.5)
# # #         sy = np.sin(yaw * 0.5)
# # #         cp = np.cos(pitch * 0.5)
# # #         sp = np.sin(pitch * 0.5)
# # #         cr = np.cos(roll * 0.5)
# # #         sr = np.sin(roll * 0.5)
        
# # #         w = cr * cp * cy + sr * sp * sy
# # #         x = sr * cp * cy - cr * sp * sy
# # #         y = cr * sp * cy + sr * cp * sy
# # #         z = cr * cp * sy - sr * sp * cy
        
# # #         return np.array([w, x, y, z])

# # # class WeldingMujocoROSBridge(MujocoROSBridge):
# # #     """용접 환경을 위한 확장된 MujocoROSBridge"""
    
# # #     def __init__(self, robot_info, camera_info, robot_controller):
# # #         super().__init__(robot_info, camera_info, robot_controller)
        
# # #         # 환경 무작위화 초기화
# # #         self.randomizer = WeldingEnvironmentRandomizer(self.model, self.data)
        
# # #         # 환경 업데이트 관련 변수
# # #         self.environment_initialized = False  # 환경 초기화 여부
# # #         self.obstacle_animation_enabled = True  # 장애물 애니메이션만 유지
        
# # #         # 초기 환경 설정 (시뮬레이션 시작 시 한 번만)
# # #         result = self.randomizer.randomize_all()
# # #         self.environment_initialized = True
        
# # #         self.get_logger().info(f"🎯 Welding environment initialized once:")
# # #         self.get_logger().info(f"   Table position: {result['table_position']}")
# # #         self.get_logger().info(f"   Lap angle: {result['lap_angle']:.3f} rad")
# # #         self.get_logger().info(f"   All waypoints reachable: {result['all_waypoints_reachable']}")
# # #         self.get_logger().info(f"🔒 Environment positions are now FIXED for this simulation session")
    
# # #     def robot_control(self):
# # #         """원래 robot_control 메서드를 오버라이드하여 용접 환경 업데이트 추가"""
# # #         self.ctrl_step = 0
# # #         sync_step = 30  # every 30 ctrl_steps

# # #         try:
# # #             while rclpy.ok() and self.running:            
# # #                 with self.lock:
# # #                     start_time = time.perf_counter()                        

# # #                     # 원래 시뮬레이션 스텝
# # #                     mujoco.mj_step(self.model, self.data)

# # #                     # 용접 환경 업데이트 (장애물 애니메이션)
# # #                     self.update_welding_environment()

# # #                     self.rc.updateModel(self.data, self.ctrl_step)
                    
# # #                     # -------------------- ADD Controller ---------------------------- #
# # #                     rclpy.spin_once(self.rc, timeout_sec=0.0001) # for scene monitor
# # #                     rclpy.spin_once(self, timeout_sec=0.0001) # for robot controller
# # #                     self.data.ctrl[:self.ctrl_dof] = self.rc.compute()   

# # #                     # --- publish joint positions ---
# # #                     js = self.get_joint_state_message()
# # #                     self.joint_state_pub.publish(js)

# # #                     # ---------------------------------------------------------------- #
# # #                     self.ctrl_step += 1

# # #                 self.time_sync(self.dt, start_time, False)
            
# # #         except KeyboardInterrupt:
# # #             self.get_logger().info("\nSimulation interrupted. Closing robot controller ...")
# # #             self.rc.destroy_node()

# # #     def get_joint_state_message(self):
# # #         """JointState 메시지 생성"""
# # #         from sensor_msgs.msg import JointState
        
# # #         js = JointState()
# # #         js.header.stamp = self.get_clock().now().to_msg()
# # #         js.name = self.joint_names
# # #         js.position = [
# # #             float(self.data.qpos[self.model.joint(j).qposadr])
# # #             for j in self.joint_names
# # #         ]
# # #         return js
    
# # #     def update_welding_environment(self):
# # #         """용접 환경 업데이트 - 장애물 애니메이션만 실행"""
# # #         # 장애물 애니메이션만 유지 (매 스텝)
# # #         if self.obstacle_animation_enabled:
# # #             self.randomizer.update_obstacles()
        
# # #         # 환경 재무작위화는 하지 않음 (한 번 설정된 환경 유지)
# # #         # 필요시 주석 해제: self.manual_randomize_environment()
    
# # #     def disable_obstacle_animation(self):
# # #         """장애물 애니메이션 비활성화 (완전 정적 환경)"""
# # #         self.obstacle_animation_enabled = False
# # #         self.get_logger().info("🔒 Obstacle animation disabled - fully static environment")
    
# # #     def enable_obstacle_animation(self):
# # #         """장애물 애니메이션 활성화"""
# # #         self.obstacle_animation_enabled = True
# # #         self.get_logger().info("🔓 Obstacle animation enabled")
    
# # #     def manual_randomize_environment(self):
# # #         """수동으로 환경 재무작위화 (필요시 호출)"""
# # #         if self.environment_initialized:
# # #             result = self.randomizer.randomize_all()
# # #             self.get_logger().info(f"🔄 Environment manually randomized:")
# # #             self.get_logger().info(f"   Table position: {result['table_position']}")
# # #             self.get_logger().info(f"   All waypoints reachable: {result['all_waypoints_reachable']}")

# # # def main():
# # #     current_dir = os.path.dirname(os.path.realpath(__file__))
    
# # #     # ROS2 초기화
# # #     rclpy.init()
    
# # #     # 파일 경로 설정
# # #     xml_path = os.path.join(current_dir, '../robots', "welding_scene2.xml")
# # #     urdf_path = os.path.join(current_dir, '../robots', 'fr3/fr3_hand.urdf')
    
# # #     # 파일 존재 확인
# # #     if not os.path.exists(xml_path):
# # #         print(f"❌ XML file not found: {xml_path}")
# # #         return
# # #     if not os.path.exists(urdf_path):
# # #         print(f"❌ URDF file not found: {urdf_path}")
# # #         return
    
# # #     try:
# # #         print("🤖 Initializing Welding Robot Controller...")
        
# # #         # Fr3 컨트롤러 초기화
# # #         rc = Fr3Controller(urdf_path)
        
# # #         # 로봇 및 카메라 정보 설정
# # #         robot_info = [xml_path, urdf_path, 1000]  # [xml_path, urdf_path, hz]
# # #         camera_info = ['hand_eye', 320, 240, 30]  # [camera_name, width, height, fps]
        
# # #         print("🌉 Setting up Welding MuJoCo-ROS Bridge...")
        
# # #         # 용접 환경용 MuJoCo-ROS 브리지 초기화
# # #         bridge = WeldingMujocoROSBridge(robot_info, camera_info, rc)
        
# # #         print("🚀 Starting Welding Simulation with ROS Bridge...")
# # #         print("💡 Available services:")
# # #         print("   - /task_move_srv")
# # #         print("   - /get_site_position")
# # #         print("   - /get_site_orientation")
# # #         print("🔧 You can now run the waypoint client!")
# # #         print("   ros2 run dm_task_manager multi_pose_task_client_waypoints")
# # #         print("🎮 Environment features:")
# # #         print("   - Environment randomized ONCE at startup")
# # #         print("   - Moving obstacle animation (realistic)")
# # #         print("   - Fixed table and static obstacle positions")
# # #         print("   - Real-time waypoint reachability checking")
        
# # #         # 브리지 실행 (ROS 서비스와 시뮬레이션 동시 실행)
# # #         bridge.run()
        
# # #     except KeyboardInterrupt:
# # #         print("\n🛑 Shutting down welding simulation...")
# # #     except Exception as e:
# # #         print(f"❌ Error: {str(e)}")
# # #         import traceback
# # #         traceback.print_exc()
# # #     finally:
# # #         # 정리
# # #         if 'bridge' in locals():
# # #             bridge.destroy_node()
# # #         rclpy.shutdown()
# # #         print("✅ Welding simulation terminated.")

# # # if __name__ == "__main__":    
# # #     main()

# # # # # /home/minjun/wr_ws/src/welding_robot/dm_ros/dm_ros/welding_free2.py
# # # # import rclpy
# # # # import os
# # # # import numpy as np
# # # # import mujoco
# # # # import threading
# # # # import time
# # # # from prc import Fr3Controller
# # # # from .utils.multi_thread import MujocoROSBridge

# # # # class WeldingEnvironmentRandomizer:
# # # #     def __init__(self, model, data):
# # # #         self.model = model
# # # #         self.data = data
        
# # # #         # 테이블과 관련 body들의 ID 저장
# # # #         self.table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
# # # #         self.lap_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lap_base")
# # # #         self.fillet_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fillet_joint_base")
# # # #         self.curved_pipe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "curved_pipe")
# # # #         self.moving_obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_obstacle")
# # # #         self.worker_torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "worker_torso")
        
# # # #         # 원래 위치 저장 (상대 위치 계산용)
# # # #         if self.table_id != -1:
# # # #             self.original_table_pos = model.body_pos[self.table_id].copy()
# # # #             self.relative_positions = {}
            
# # # #             if self.lap_base_id != -1:
# # # #                 self.relative_positions['lap_base'] = model.body_pos[self.lap_base_id] - self.original_table_pos
# # # #             if self.fillet_joint_id != -1:
# # # #                 self.relative_positions['fillet_joint'] = model.body_pos[self.fillet_joint_id] - self.original_table_pos
# # # #             if self.curved_pipe_id != -1:
# # # #                 self.relative_positions['curved_pipe'] = model.body_pos[self.curved_pipe_id] - self.original_table_pos
# # # #             if self.moving_obstacle_id != -1:
# # # #                 self.relative_positions['moving_obstacle'] = model.body_pos[self.moving_obstacle_id] - self.original_table_pos
# # # #             if self.worker_torso_id != -1:
# # # #                 self.relative_positions['worker_torso'] = model.body_pos[self.worker_torso_id] - self.original_table_pos
        
# # # #     def randomize_table_position(self, x_range=(0.4, 0.6), y_range=(-0.1, 0.1), z_range=(0.4, 0.5)):
# # # #         """테이블 위치를 무작위화하고 관련된 모든 오브젝트도 함께 이동"""
# # # #         if self.table_id == -1:
# # # #             return np.array([0.5, 0, 0.45])  # 기본값 반환
            
# # # #         # 새로운 테이블 위치 생성
# # # #         new_table_pos = np.array([
# # # #             np.random.uniform(*x_range),
# # # #             np.random.uniform(*y_range),
# # # #             np.random.uniform(*z_range)
# # # #         ])
        
# # # #         # 테이블 위치 업데이트
# # # #         self.model.body_pos[self.table_id] = new_table_pos
        
# # # #         # 테이블 위의 모든 오브젝트들도 상대 위치 유지하며 이동
# # # #         if self.lap_base_id != -1 and 'lap_base' in self.relative_positions:
# # # #             self.model.body_pos[self.lap_base_id] = new_table_pos + self.relative_positions['lap_base']
# # # #         if self.fillet_joint_id != -1 and 'fillet_joint' in self.relative_positions:
# # # #             self.model.body_pos[self.fillet_joint_id] = new_table_pos + self.relative_positions['fillet_joint']
# # # #         if self.curved_pipe_id != -1 and 'curved_pipe' in self.relative_positions:
# # # #             self.model.body_pos[self.curved_pipe_id] = new_table_pos + self.relative_positions['curved_pipe']
# # # #         if self.moving_obstacle_id != -1 and 'moving_obstacle' in self.relative_positions:
# # # #             self.model.body_pos[self.moving_obstacle_id] = new_table_pos + self.relative_positions['moving_obstacle']
# # # #         if self.worker_torso_id != -1 and 'worker_torso' in self.relative_positions:
# # # #             self.model.body_pos[self.worker_torso_id] = new_table_pos + self.relative_positions['worker_torso']
        
# # # #         return new_table_pos
    
# # # #     def randomize_obstacles(self):
# # # #         """장애물 위치를 무작위화"""
# # # #         # 움직이는 장애물의 초기 위치 설정
# # # #         obstacle_joint_ids = [
# # # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x"),
# # # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y"),
# # # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_z")
# # # #         ]
        
# # # #         for joint_id in obstacle_joint_ids:
# # # #             if joint_id != -1:
# # # #                 range_min = self.model.jnt_range[joint_id][0]
# # # #                 range_max = self.model.jnt_range[joint_id][1]
# # # #                 self.data.qpos[self.model.jnt_qposadr[joint_id]] = np.random.uniform(range_min, range_max)
    
# # # #     def randomize_lap_joint_angle(self, angle_range=(-0.3, 0.3)):
# # # #         """Lap joint의 초기 각도를 무작위화"""
# # # #         lap_base_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lap_base")
# # # #         if lap_base_joint != -1 and self.lap_base_id != -1:
# # # #             angle = np.random.uniform(*angle_range)
# # # #             self.model.body_quat[self.lap_base_id] = self._euler_to_quat(0, 0, angle)
# # # #             return angle
# # # #         return 0
    
# # # #     def check_reachability(self, site_name, robot_reach=0.855):
# # # #         """특정 사이트가 로봇의 도달 범위 내에 있는지 확인"""
# # # #         site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
# # # #         if site_id != -1:
# # # #             site_pos = self.data.site_xpos[site_id]
# # # #             # 로봇 베이스를 원점으로 가정
# # # #             distance = np.linalg.norm(site_pos)
# # # #             return distance <= robot_reach
# # # #         return False
    
# # # #     def update_obstacles(self):
# # # #         """장애물 애니메이션 업데이트"""
# # # #         t = self.data.time
        
# # # #         # 움직이는 장애물 애니메이션
# # # #         obstacle_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x")
# # # #         obstacle_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y")
        
# # # #         if obstacle_x_id != -1:
# # # #             addr = self.model.jnt_qposadr[obstacle_x_id]
# # # #             self.data.qpos[addr] = 0.05 * np.sin(t * 0.5)  # 천천히 좌우 이동
            
# # # #         if obstacle_y_id != -1:
# # # #             addr = self.model.jnt_qposadr[obstacle_y_id]
# # # #             self.data.qpos[addr] = 0.03 * np.cos(t * 0.7)  # 천천히 앞뒤 이동
    
# # # #     def randomize_all(self):
# # # #         """전체 환경을 무작위화"""
# # # #         # 테이블 위치 무작위화
# # # #         table_pos = self.randomize_table_position()
        
# # # #         # Lap joint 각도 무작위화
# # # #         lap_angle = self.randomize_lap_joint_angle()
        
# # # #         # 장애물 무작위화
# # # #         self.randomize_obstacles()
        
# # # #         # Forward kinematics 업데이트
# # # #         mujoco.mj_forward(self.model, self.data)
        
# # # #         # 도달 가능성 확인
# # # #         reachable = all([
# # # #             self.check_reachability("lap_start"),
# # # #             self.check_reachability("lap_waypoint1"),
# # # #             self.check_reachability("lap_waypoint2"),
# # # #             self.check_reachability("lap_waypoint3"),
# # # #             self.check_reachability("lap_end")
# # # #         ])
        
# # # #         return {
# # # #             'table_position': table_pos,
# # # #             'lap_angle': lap_angle,
# # # #             'all_waypoints_reachable': reachable
# # # #         }
    
# # # #     def _euler_to_quat(self, roll, pitch, yaw):
# # # #         """Euler angles to quaternion conversion"""
# # # #         cy = np.cos(yaw * 0.5)
# # # #         sy = np.sin(yaw * 0.5)
# # # #         cp = np.cos(pitch * 0.5)
# # # #         sp = np.sin(pitch * 0.5)
# # # #         cr = np.cos(roll * 0.5)
# # # #         sr = np.sin(roll * 0.5)
        
# # # #         w = cr * cp * cy + sr * sp * sy
# # # #         x = sr * cp * cy - cr * sp * sy
# # # #         y = cr * sp * cy + sr * cp * sy
# # # #         z = cr * cp * sy - sr * sp * cy
        
# # # #         return np.array([w, x, y, z])

# # # # class WeldingMujocoROSBridge(MujocoROSBridge):
# # # #     """용접 환경을 위한 확장된 MujocoROSBridge"""
    
# # # #     def __init__(self, robot_info, camera_info, robot_controller):
# # # #         super().__init__(robot_info, camera_info, robot_controller)
        
# # # #         # 환경 무작위화 초기화
# # # #         self.randomizer = WeldingEnvironmentRandomizer(self.model, self.data)
        
# # # #         # 환경 업데이트 관련 변수
# # # #         self.environment_initialized = False  # 환경 초기화 여부
# # # #         self.obstacle_animation_enabled = True  # 장애물 애니메이션만 유지
        
# # # #         # 초기 환경 설정 (시뮬레이션 시작 시 한 번만)
# # # #         result = self.randomizer.randomize_all()
# # # #         self.environment_initialized = True
        
# # # #         self.get_logger().info(f"🎯 Welding environment initialized once:")
# # # #         self.get_logger().info(f"   Table position: {result['table_position']}")
# # # #         self.get_logger().info(f"   Lap angle: {result['lap_angle']:.3f} rad")
# # # #         self.get_logger().info(f"   All waypoints reachable: {result['all_waypoints_reachable']}")
# # # #         self.get_logger().info(f"🔒 Environment positions are now FIXED for this simulation session")
    
# # # #     def robot_control(self):
# # # #         """원래 robot_control 메서드를 오버라이드하여 용접 환경 업데이트 추가"""
# # # #         self.ctrl_step = 0
# # # #         sync_step = 30  # every 30 ctrl_steps

# # # #         try:
# # # #             while rclpy.ok() and self.running:            
# # # #                 with self.lock:
# # # #                     start_time = time.perf_counter()                        

# # # #                     # 원래 시뮬레이션 스텝
# # # #                     mujoco.mj_step(self.model, self.data)

# # # #                     # 용접 환경 업데이트 (장애물 애니메이션)
# # # #                     self.update_welding_environment()

# # # #                     self.rc.updateModel(self.data, self.ctrl_step)
                    
# # # #                     # -------------------- ADD Controller ---------------------------- #
# # # #                     rclpy.spin_once(self.rc, timeout_sec=0.0001) # for scene monitor
# # # #                     rclpy.spin_once(self, timeout_sec=0.0001) # for robot controller
# # # #                     self.data.ctrl[:self.ctrl_dof] = self.rc.compute()   

# # # #                     # --- publish joint positions ---
# # # #                     js = self.get_joint_state_message()
# # # #                     self.joint_state_pub.publish(js)

# # # #                     # ---------------------------------------------------------------- #
# # # #                     self.ctrl_step += 1

# # # #                 self.time_sync(self.dt, start_time, False)
            
# # # #         except KeyboardInterrupt:
# # # #             self.get_logger().info("\nSimulation interrupted. Closing robot controller ...")
# # # #             self.rc.destroy_node()

# # # #     def get_joint_state_message(self):
# # # #         """JointState 메시지 생성"""
# # # #         from sensor_msgs.msg import JointState
        
# # # #         js = JointState()
# # # #         js.header.stamp = self.get_clock().now().to_msg()
# # # #         js.name = self.joint_names
# # # #         js.position = [
# # # #             float(self.data.qpos[self.model.joint(j).qposadr])
# # # #             for j in self.joint_names
# # # #         ]
# # # #         return js
    
# # # #     def update_welding_environment(self):
# # # #         """용접 환경 업데이트 - 장애물 애니메이션만 실행"""
# # # #         # 장애물 애니메이션만 유지 (매 스텝)
# # # #         if self.obstacle_animation_enabled:
# # # #             self.randomizer.update_obstacles()
        
# # # #         # 환경 재무작위화는 하지 않음 (한 번 설정된 환경 유지)
# # # #         # 필요시 주석 해제: self.manual_randomize_environment()
    
# # # #     def disable_obstacle_animation(self):
# # # #         """장애물 애니메이션 비활성화 (완전 정적 환경)"""
# # # #         self.obstacle_animation_enabled = False
# # # #         self.get_logger().info("🔒 Obstacle animation disabled - fully static environment")
    
# # # #     def enable_obstacle_animation(self):
# # # #         """장애물 애니메이션 활성화"""
# # # #         self.obstacle_animation_enabled = True
# # # #         self.get_logger().info("🔓 Obstacle animation enabled")
    
# # # #     def manual_randomize_environment(self):
# # # #         """수동으로 환경 재무작위화 (필요시 호출)"""
# # # #         if self.environment_initialized:
# # # #             result = self.randomizer.randomize_all()
# # # #             self.get_logger().info(f"🔄 Environment manually randomized:")
# # # #             self.get_logger().info(f"   Table position: {result['table_position']}")
# # # #             self.get_logger().info(f"   All waypoints reachable: {result['all_waypoints_reachable']}")

# # # # def main():
# # # #     current_dir = os.path.dirname(os.path.realpath(__file__))
    
# # # #     # ROS2 초기화
# # # #     rclpy.init()
    
# # # #     # 파일 경로 설정
# # # #     xml_path = os.path.join(current_dir, '../robots', "welding_scene2.xml")
# # # #     urdf_path = os.path.join(current_dir, '../robots', 'fr3/fr3_hand.urdf')
    
# # # #     # 파일 존재 확인
# # # #     if not os.path.exists(xml_path):
# # # #         print(f"❌ XML file not found: {xml_path}")
# # # #         return
# # # #     if not os.path.exists(urdf_path):
# # # #         print(f"❌ URDF file not found: {urdf_path}")
# # # #         return
    
# # # #     try:
# # # #         print("🤖 Initializing Welding Robot Controller...")
        
# # # #         # Fr3 컨트롤러 초기화
# # # #         rc = Fr3Controller(urdf_path)
        
# # # #         # 로봇 및 카메라 정보 설정
# # # #         robot_info = [xml_path, urdf_path, 1000]  # [xml_path, urdf_path, hz]
# # # #         camera_info = ['hand_eye', 320, 240, 30]  # [camera_name, width, height, fps]
        
# # # #         print("🌉 Setting up Welding MuJoCo-ROS Bridge...")
        
# # # #         # 용접 환경용 MuJoCo-ROS 브리지 초기화
# # # #         bridge = WeldingMujocoROSBridge(robot_info, camera_info, rc)
        
# # # #         print("🚀 Starting Welding Simulation with ROS Bridge...")
# # # #         print("💡 Available services:")
# # # #         print("   - /task_move_srv")
# # # #         print("   - /get_site_position")
# # # #         print("   - /get_site_orientation")
# # # #         print("🔧 You can now run the waypoint client!")
# # # #         print("   ros2 run dm_task_manager multi_pose_task_client_waypoints")
# # # #         print("🎮 Environment features:")
# # # #         print("   - Environment randomized ONCE at startup")
# # # #         print("   - Moving obstacle animation (realistic)")
# # # #         print("   - Fixed table and static obstacle positions")
# # # #         print("   - Real-time waypoint reachability checking")
        
# # # #         # 브리지 실행 (ROS 서비스와 시뮬레이션 동시 실행)
# # # #         bridge.run()
        
# # # #     except KeyboardInterrupt:
# # # #         print("\n🛑 Shutting down welding simulation...")
# # # #     except Exception as e:
# # # #         print(f"❌ Error: {str(e)}")
# # # #         import traceback
# # # #         traceback.print_exc()
# # # #     finally:
# # # #         # 정리
# # # #         if 'bridge' in locals():
# # # #             bridge.destroy_node()
# # # #         rclpy.shutdown()
# # # #         print("✅ Welding simulation terminated.")

# # # # if __name__ == "__main__":    
# # # #     main()
    
    
# # # # # # /home/minjun/wr_ws/src/welding_robot/dm_ros/dm_ros/welding_free2.py
# # # # # import rclpy
# # # # # import os
# # # # # import numpy as np
# # # # # import mujoco
# # # # # import threading
# # # # # import time
# # # # # from prc import Fr3Controller
# # # # # from .utils.multi_thread import MujocoROSBridge

# # # # # class WeldingEnvironmentRandomizer:
# # # # #     def __init__(self, model, data):
# # # # #         self.model = model
# # # # #         self.data = data
        
# # # # #         # 테이블과 관련 body들의 ID 저장
# # # # #         self.table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
# # # # #         self.lap_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lap_base")
# # # # #         self.fillet_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fillet_joint_base")
# # # # #         self.curved_pipe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "curved_pipe")
# # # # #         self.moving_obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_obstacle")
# # # # #         self.worker_torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "worker_torso")
        
# # # # #         # 원래 위치 저장 (상대 위치 계산용)
# # # # #         if self.table_id != -1:
# # # # #             self.original_table_pos = model.body_pos[self.table_id].copy()
# # # # #             self.relative_positions = {}
            
# # # # #             if self.lap_base_id != -1:
# # # # #                 self.relative_positions['lap_base'] = model.body_pos[self.lap_base_id] - self.original_table_pos
# # # # #             if self.fillet_joint_id != -1:
# # # # #                 self.relative_positions['fillet_joint'] = model.body_pos[self.fillet_joint_id] - self.original_table_pos
# # # # #             if self.curved_pipe_id != -1:
# # # # #                 self.relative_positions['curved_pipe'] = model.body_pos[self.curved_pipe_id] - self.original_table_pos
# # # # #             if self.moving_obstacle_id != -1:
# # # # #                 self.relative_positions['moving_obstacle'] = model.body_pos[self.moving_obstacle_id] - self.original_table_pos
# # # # #             if self.worker_torso_id != -1:
# # # # #                 self.relative_positions['worker_torso'] = model.body_pos[self.worker_torso_id] - self.original_table_pos
        
# # # # #     def randomize_table_position(self, x_range=(0.4, 0.6), y_range=(-0.1, 0.1), z_range=(0.4, 0.5)):
# # # # #         """테이블 위치를 무작위화하고 관련된 모든 오브젝트도 함께 이동"""
# # # # #         if self.table_id == -1:
# # # # #             return np.array([0.5, 0, 0.45])  # 기본값 반환
            
# # # # #         # 새로운 테이블 위치 생성
# # # # #         new_table_pos = np.array([
# # # # #             np.random.uniform(*x_range),
# # # # #             np.random.uniform(*y_range),
# # # # #             np.random.uniform(*z_range)
# # # # #         ])
        
# # # # #         # 테이블 위치 업데이트
# # # # #         self.model.body_pos[self.table_id] = new_table_pos
        
# # # # #         # 테이블 위의 모든 오브젝트들도 상대 위치 유지하며 이동
# # # # #         if self.lap_base_id != -1 and 'lap_base' in self.relative_positions:
# # # # #             self.model.body_pos[self.lap_base_id] = new_table_pos + self.relative_positions['lap_base']
# # # # #         if self.fillet_joint_id != -1 and 'fillet_joint' in self.relative_positions:
# # # # #             self.model.body_pos[self.fillet_joint_id] = new_table_pos + self.relative_positions['fillet_joint']
# # # # #         if self.curved_pipe_id != -1 and 'curved_pipe' in self.relative_positions:
# # # # #             self.model.body_pos[self.curved_pipe_id] = new_table_pos + self.relative_positions['curved_pipe']
# # # # #         if self.moving_obstacle_id != -1 and 'moving_obstacle' in self.relative_positions:
# # # # #             self.model.body_pos[self.moving_obstacle_id] = new_table_pos + self.relative_positions['moving_obstacle']
# # # # #         if self.worker_torso_id != -1 and 'worker_torso' in self.relative_positions:
# # # # #             self.model.body_pos[self.worker_torso_id] = new_table_pos + self.relative_positions['worker_torso']
        
# # # # #         return new_table_pos
    
# # # # #     def randomize_obstacles(self):
# # # # #         """장애물 위치를 무작위화"""
# # # # #         # 움직이는 장애물의 초기 위치 설정
# # # # #         obstacle_joint_ids = [
# # # # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x"),
# # # # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y"),
# # # # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_z")
# # # # #         ]
        
# # # # #         for joint_id in obstacle_joint_ids:
# # # # #             if joint_id != -1:
# # # # #                 range_min = self.model.jnt_range[joint_id][0]
# # # # #                 range_max = self.model.jnt_range[joint_id][1]
# # # # #                 self.data.qpos[self.model.jnt_qposadr[joint_id]] = np.random.uniform(range_min, range_max)
    
# # # # #     def randomize_lap_joint_angle(self, angle_range=(-0.3, 0.3)):
# # # # #         """Lap joint의 초기 각도를 무작위화"""
# # # # #         lap_base_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lap_base")
# # # # #         if lap_base_joint != -1 and self.lap_base_id != -1:
# # # # #             angle = np.random.uniform(*angle_range)
# # # # #             self.model.body_quat[self.lap_base_id] = self._euler_to_quat(0, 0, angle)
# # # # #             return angle
# # # # #         return 0
    
# # # # #     def check_reachability(self, site_name, robot_reach=0.855):
# # # # #         """특정 사이트가 로봇의 도달 범위 내에 있는지 확인"""
# # # # #         site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
# # # # #         if site_id != -1:
# # # # #             site_pos = self.data.site_xpos[site_id]
# # # # #             # 로봇 베이스를 원점으로 가정
# # # # #             distance = np.linalg.norm(site_pos)
# # # # #             return distance <= robot_reach
# # # # #         return False
    
# # # # #     def update_obstacles(self):
# # # # #         """장애물 애니메이션 업데이트"""
# # # # #         t = self.data.time
        
# # # # #         # 움직이는 장애물 애니메이션
# # # # #         obstacle_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x")
# # # # #         obstacle_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y")
        
# # # # #         if obstacle_x_id != -1:
# # # # #             addr = self.model.jnt_qposadr[obstacle_x_id]
# # # # #             self.data.qpos[addr] = 0.05 * np.sin(t * 0.5)  # 천천히 좌우 이동
            
# # # # #         if obstacle_y_id != -1:
# # # # #             addr = self.model.jnt_qposadr[obstacle_y_id]
# # # # #             self.data.qpos[addr] = 0.03 * np.cos(t * 0.7)  # 천천히 앞뒤 이동
    
# # # # #     def randomize_all(self):
# # # # #         """전체 환경을 무작위화"""
# # # # #         # 테이블 위치 무작위화
# # # # #         table_pos = self.randomize_table_position()
        
# # # # #         # Lap joint 각도 무작위화
# # # # #         lap_angle = self.randomize_lap_joint_angle()
        
# # # # #         # 장애물 무작위화
# # # # #         self.randomize_obstacles()
        
# # # # #         # Forward kinematics 업데이트
# # # # #         mujoco.mj_forward(self.model, self.data)
        
# # # # #         # 도달 가능성 확인
# # # # #         reachable = all([
# # # # #             self.check_reachability("lap_start"),
# # # # #             self.check_reachability("lap_waypoint1"),
# # # # #             self.check_reachability("lap_waypoint2"),
# # # # #             self.check_reachability("lap_waypoint3"),
# # # # #             self.check_reachability("lap_end")
# # # # #         ])
        
# # # # #         return {
# # # # #             'table_position': table_pos,
# # # # #             'lap_angle': lap_angle,
# # # # #             'all_waypoints_reachable': reachable
# # # # #         }
    
# # # # #     def _euler_to_quat(self, roll, pitch, yaw):
# # # # #         """Euler angles to quaternion conversion"""
# # # # #         cy = np.cos(yaw * 0.5)
# # # # #         sy = np.sin(yaw * 0.5)
# # # # #         cp = np.cos(pitch * 0.5)
# # # # #         sp = np.sin(pitch * 0.5)
# # # # #         cr = np.cos(roll * 0.5)
# # # # #         sr = np.sin(roll * 0.5)
        
# # # # #         w = cr * cp * cy + sr * sp * sy
# # # # #         x = sr * cp * cy - cr * sp * sy
# # # # #         y = cr * sp * cy + sr * cp * sy
# # # # #         z = cr * cp * sy - sr * sp * cy
        
# # # # #         return np.array([w, x, y, z])

# # # # # class WeldingMujocoROSBridge(MujocoROSBridge):
# # # # #     """용접 환경을 위한 확장된 MujocoROSBridge"""
    
# # # # #     def __init__(self, robot_info, camera_info, robot_controller):
# # # # #         super().__init__(robot_info, camera_info, robot_controller)
        
# # # # #         # 환경 무작위화 초기화
# # # # #         self.randomizer = WeldingEnvironmentRandomizer(self.model, self.data)
        
# # # # #         # 환경 업데이트 관련 변수
# # # # #         self.last_randomize_time = 0
# # # # #         self.randomize_interval = 10.0  # 10초마다 환경 무작위화
# # # # #         self.randomize_enabled = True
        
# # # # #         # 초기 환경 설정
# # # # #         result = self.randomizer.randomize_all()
# # # # #         self.get_logger().info(f"🎯 Initial welding environment setup:")
# # # # #         self.get_logger().info(f"   Table position: {result['table_position']}")
# # # # #         self.get_logger().info(f"   Lap angle: {result['lap_angle']:.3f} rad")
# # # # #         self.get_logger().info(f"   All waypoints reachable: {result['all_waypoints_reachable']}")
    
# # # # #     def robot_control(self):
# # # # #         """원래 robot_control 메서드를 오버라이드하여 용접 환경 업데이트 추가"""
# # # # #         self.ctrl_step = 0
# # # # #         sync_step = 30  # every 30 ctrl_steps

# # # # #         try:
# # # # #             while rclpy.ok() and self.running:            
# # # # #                 with self.lock:
# # # # #                     start_time = time.perf_counter()                        

# # # # #                     # 원래 시뮬레이션 스텝
# # # # #                     mujoco.mj_step(self.model, self.data)

# # # # #                     # 용접 환경 업데이트 (장애물 애니메이션)
# # # # #                     self.update_welding_environment()

# # # # #                     self.rc.updateModel(self.data, self.ctrl_step)
                    
# # # # #                     # -------------------- ADD Controller ---------------------------- #
# # # # #                     rclpy.spin_once(self.rc, timeout_sec=0.0001) # for scene monitor
# # # # #                     rclpy.spin_once(self, timeout_sec=0.0001) # for robot controller
# # # # #                     self.data.ctrl[:self.ctrl_dof] = self.rc.compute()   

# # # # #                     # --- publish joint positions ---
# # # # #                     js = self.get_joint_state_message()
# # # # #                     self.joint_state_pub.publish(js)

# # # # #                     # ---------------------------------------------------------------- #
# # # # #                     self.ctrl_step += 1

# # # # #                 self.time_sync(self.dt, start_time, False)
            
# # # # #         except KeyboardInterrupt:
# # # # #             self.get_logger().info("\nSimulation interrupted. Closing robot controller ...")
# # # # #             self.rc.destroy_node()

# # # # #     def get_joint_state_message(self):
# # # # #         """JointState 메시지 생성"""
# # # # #         from sensor_msgs.msg import JointState
        
# # # # #         js = JointState()
# # # # #         js.header.stamp = self.get_clock().now().to_msg()
# # # # #         js.name = self.joint_names
# # # # #         js.position = [
# # # # #             float(self.data.qpos[self.model.joint(j).qposadr])
# # # # #             for j in self.joint_names
# # # # #         ]
# # # # #         return js
    
# # # # #     def update_welding_environment(self):
# # # # #         """용접 환경 업데이트"""
# # # # #         # 장애물 애니메이션 (매 스텝)
# # # # #         self.randomizer.update_obstacles()
        
# # # # #         # 주기적 환경 무작위화
# # # # #         if (self.randomize_enabled and 
# # # # #             self.data.time - self.last_randomize_time > self.randomize_interval):
            
# # # # #             result = self.randomizer.randomize_all()
# # # # #             self.get_logger().info(f"🔄 Environment randomized at t={self.data.time:.1f}s")
# # # # #             self.get_logger().info(f"   Table position: {result['table_position']}")
# # # # #             self.get_logger().info(f"   All waypoints reachable: {result['all_waypoints_reachable']}")
# # # # #             self.last_randomize_time = self.data.time
    
# # # # #     def disable_randomization(self):
# # # # #         """환경 무작위화 비활성화"""
# # # # #         self.randomize_enabled = False
# # # # #         self.get_logger().info("🔒 Environment randomization disabled")
    
# # # # #     def enable_randomization(self):
# # # # #         """환경 무작위화 활성화"""
# # # # #         self.randomize_enabled = True
# # # # #         self.get_logger().info("🔓 Environment randomization enabled")

# # # # # def main():
# # # # #     current_dir = os.path.dirname(os.path.realpath(__file__))
    
# # # # #     # ROS2 초기화
# # # # #     rclpy.init()
    
# # # # #     # 파일 경로 설정
# # # # #     xml_path = os.path.join(current_dir, '../robots', "welding_scene2.xml")
# # # # #     urdf_path = os.path.join(current_dir, '../robots', 'fr3/fr3_hand.urdf')
    
# # # # #     # 파일 존재 확인
# # # # #     if not os.path.exists(xml_path):
# # # # #         print(f"❌ XML file not found: {xml_path}")
# # # # #         return
# # # # #     if not os.path.exists(urdf_path):
# # # # #         print(f"❌ URDF file not found: {urdf_path}")
# # # # #         return
    
# # # # #     try:
# # # # #         print("🤖 Initializing Welding Robot Controller...")
        
# # # # #         # Fr3 컨트롤러 초기화
# # # # #         rc = Fr3Controller(urdf_path)
        
# # # # #         # 로봇 및 카메라 정보 설정
# # # # #         robot_info = [xml_path, urdf_path, 1000]  # [xml_path, urdf_path, hz]
# # # # #         camera_info = ['hand_eye', 320, 240, 30]  # [camera_name, width, height, fps]
        
# # # # #         print("🌉 Setting up Welding MuJoCo-ROS Bridge...")
        
# # # # #         # 용접 환경용 MuJoCo-ROS 브리지 초기화
# # # # #         bridge = WeldingMujocoROSBridge(robot_info, camera_info, rc)
        
# # # # #         print("🚀 Starting Welding Simulation with ROS Bridge...")
# # # # #         print("💡 Available services:")
# # # # #         print("   - /task_move_srv")
# # # # #         print("   - /get_site_position")
# # # # #         print("   - /get_site_orientation")
# # # # #         print("🔧 You can now run the waypoint client!")
# # # # #         print("   ros2 run dm_task_manager multi_pose_task_client_waypoints")
# # # # #         print("🎮 Environment features:")
# # # # #         print("   - Automatic obstacle animation")
# # # # #         print("   - Periodic environment randomization (every 10s)")
# # # # #         print("   - Real-time waypoint reachability checking")
        
# # # # #         # 브리지 실행 (ROS 서비스와 시뮬레이션 동시 실행)
# # # # #         bridge.run()
        
# # # # #     except KeyboardInterrupt:
# # # # #         print("\n🛑 Shutting down welding simulation...")
# # # # #     except Exception as e:
# # # # #         print(f"❌ Error: {str(e)}")
# # # # #         import traceback
# # # # #         traceback.print_exc()
# # # # #     finally:
# # # # #         # 정리
# # # # #         if 'bridge' in locals():
# # # # #             bridge.destroy_node()
# # # # #         rclpy.shutdown()
# # # # #         print("✅ Welding simulation terminated.")

# # # # # if __name__ == "__main__":    
# # # # #     main()

# # # # # # # /home/minjun/wr_ws/src/welding_robot/dm_ros/dm_ros/welding_free2.py
# # # # # # import rclpy
# # # # # # import os
# # # # # # import numpy as np
# # # # # # import mujoco
# # # # # # import mujoco.viewer
# # # # # # import threading
# # # # # # import time
# # # # # # from prc import Fr3Controller
# # # # # # from .utils.multi_thread import MujocoROSBridge

# # # # # # class WeldingEnvironmentRandomizer:
# # # # # #     def __init__(self, model, data):
# # # # # #         self.model = model
# # # # # #         self.data = data
        
# # # # # #         # 테이블과 관련 body들의 ID 저장
# # # # # #         self.table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
# # # # # #         self.lap_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lap_base")
# # # # # #         self.fillet_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fillet_joint_base")
# # # # # #         self.curved_pipe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "curved_pipe")
# # # # # #         self.moving_obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_obstacle")
# # # # # #         self.worker_torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "worker_torso")
        
# # # # # #         # 원래 위치 저장 (상대 위치 계산용)
# # # # # #         if self.table_id != -1:
# # # # # #             self.original_table_pos = model.body_pos[self.table_id].copy()
# # # # # #             self.relative_positions = {}
            
# # # # # #             if self.lap_base_id != -1:
# # # # # #                 self.relative_positions['lap_base'] = model.body_pos[self.lap_base_id] - self.original_table_pos
# # # # # #             if self.fillet_joint_id != -1:
# # # # # #                 self.relative_positions['fillet_joint'] = model.body_pos[self.fillet_joint_id] - self.original_table_pos
# # # # # #             if self.curved_pipe_id != -1:
# # # # # #                 self.relative_positions['curved_pipe'] = model.body_pos[self.curved_pipe_id] - self.original_table_pos
# # # # # #             if self.moving_obstacle_id != -1:
# # # # # #                 self.relative_positions['moving_obstacle'] = model.body_pos[self.moving_obstacle_id] - self.original_table_pos
# # # # # #             if self.worker_torso_id != -1:
# # # # # #                 self.relative_positions['worker_torso'] = model.body_pos[self.worker_torso_id] - self.original_table_pos
        
# # # # # #     def randomize_table_position(self, x_range=(0.4, 0.6), y_range=(-0.1, 0.1), z_range=(0.4, 0.5)):
# # # # # #         """테이블 위치를 무작위화하고 관련된 모든 오브젝트도 함께 이동"""
# # # # # #         if self.table_id == -1:
# # # # # #             return np.array([0.5, 0, 0.45])  # 기본값 반환
            
# # # # # #         # 새로운 테이블 위치 생성
# # # # # #         new_table_pos = np.array([
# # # # # #             np.random.uniform(*x_range),
# # # # # #             np.random.uniform(*y_range),
# # # # # #             np.random.uniform(*z_range)
# # # # # #         ])
        
# # # # # #         # 테이블 위치 업데이트
# # # # # #         self.model.body_pos[self.table_id] = new_table_pos
        
# # # # # #         # 테이블 위의 모든 오브젝트들도 상대 위치 유지하며 이동
# # # # # #         if self.lap_base_id != -1 and 'lap_base' in self.relative_positions:
# # # # # #             self.model.body_pos[self.lap_base_id] = new_table_pos + self.relative_positions['lap_base']
# # # # # #         if self.fillet_joint_id != -1 and 'fillet_joint' in self.relative_positions:
# # # # # #             self.model.body_pos[self.fillet_joint_id] = new_table_pos + self.relative_positions['fillet_joint']
# # # # # #         if self.curved_pipe_id != -1 and 'curved_pipe' in self.relative_positions:
# # # # # #             self.model.body_pos[self.curved_pipe_id] = new_table_pos + self.relative_positions['curved_pipe']
# # # # # #         if self.moving_obstacle_id != -1 and 'moving_obstacle' in self.relative_positions:
# # # # # #             self.model.body_pos[self.moving_obstacle_id] = new_table_pos + self.relative_positions['moving_obstacle']
# # # # # #         if self.worker_torso_id != -1 and 'worker_torso' in self.relative_positions:
# # # # # #             self.model.body_pos[self.worker_torso_id] = new_table_pos + self.relative_positions['worker_torso']
        
# # # # # #         return new_table_pos
    
# # # # # #     def randomize_obstacles(self):
# # # # # #         """장애물 위치를 무작위화"""
# # # # # #         # 움직이는 장애물의 초기 위치 설정
# # # # # #         obstacle_joint_ids = [
# # # # # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x"),
# # # # # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y"),
# # # # # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_z")
# # # # # #         ]
        
# # # # # #         for joint_id in obstacle_joint_ids:
# # # # # #             if joint_id != -1:
# # # # # #                 range_min = self.model.jnt_range[joint_id][0]
# # # # # #                 range_max = self.model.jnt_range[joint_id][1]
# # # # # #                 self.data.qpos[self.model.jnt_qposadr[joint_id]] = np.random.uniform(range_min, range_max)
    
# # # # # #     def randomize_lap_joint_angle(self, angle_range=(-0.3, 0.3)):
# # # # # #         """Lap joint의 초기 각도를 무작위화"""
# # # # # #         lap_base_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lap_base")
# # # # # #         if lap_base_joint != -1 and self.lap_base_id != -1:
# # # # # #             angle = np.random.uniform(*angle_range)
# # # # # #             self.model.body_quat[self.lap_base_id] = self._euler_to_quat(0, 0, angle)
# # # # # #             return angle
# # # # # #         return 0
    
# # # # # #     def check_reachability(self, site_name, robot_reach=0.855):
# # # # # #         """특정 사이트가 로봇의 도달 범위 내에 있는지 확인"""
# # # # # #         site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
# # # # # #         if site_id != -1:
# # # # # #             site_pos = self.data.site_xpos[site_id]
# # # # # #             # 로봇 베이스를 원점으로 가정
# # # # # #             distance = np.linalg.norm(site_pos)
# # # # # #             return distance <= robot_reach
# # # # # #         return False
    
# # # # # #     def randomize_all(self):
# # # # # #         """전체 환경을 무작위화"""
# # # # # #         # 테이블 위치 무작위화
# # # # # #         table_pos = self.randomize_table_position()
        
# # # # # #         # Lap joint 각도 무작위화
# # # # # #         lap_angle = self.randomize_lap_joint_angle()
        
# # # # # #         # 장애물 무작위화
# # # # # #         self.randomize_obstacles()
        
# # # # # #         # Forward kinematics 업데이트
# # # # # #         mujoco.mj_forward(self.model, self.data)
        
# # # # # #         # 도달 가능성 확인
# # # # # #         reachable = all([
# # # # # #             self.check_reachability("lap_start"),
# # # # # #             self.check_reachability("lap_waypoint1"),
# # # # # #             self.check_reachability("lap_waypoint2"),
# # # # # #             self.check_reachability("lap_waypoint3"),
# # # # # #             self.check_reachability("lap_end")
# # # # # #         ])
        
# # # # # #         return {
# # # # # #             'table_position': table_pos,
# # # # # #             'lap_angle': lap_angle,
# # # # # #             'all_waypoints_reachable': reachable
# # # # # #         }
    
# # # # # #     def _euler_to_quat(self, roll, pitch, yaw):
# # # # # #         """Euler angles to quaternion conversion"""
# # # # # #         cy = np.cos(yaw * 0.5)
# # # # # #         sy = np.sin(yaw * 0.5)
# # # # # #         cp = np.cos(pitch * 0.5)
# # # # # #         sp = np.sin(pitch * 0.5)
# # # # # #         cr = np.cos(roll * 0.5)
# # # # # #         sr = np.sin(roll * 0.5)
        
# # # # # #         w = cr * cp * cy + sr * sp * sy
# # # # # #         x = sr * cp * cy - cr * sp * sy
# # # # # #         y = cr * sp * cy + sr * cp * sy
# # # # # #         z = cr * cp * sy - sr * sp * cy
        
# # # # # #         return np.array([w, x, y, z])

# # # # # # class WeldingSimulation:
# # # # # #     """용접 시뮬레이션을 관리하는 클래스"""
# # # # # #     def __init__(self, bridge, randomizer):
# # # # # #         self.bridge = bridge
# # # # # #         self.randomizer = randomizer
# # # # # #         self.model = bridge.model
# # # # # #         self.data = bridge.data
# # # # # #         self.running = True
# # # # # #         self.last_randomize_time = 0
# # # # # #         self.randomize_interval = 10.0  # 10초마다 환경 무작위화
        
# # # # # #     def update_obstacles(self):
# # # # # #         """장애물 애니메이션 업데이트"""
# # # # # #         t = self.data.time
        
# # # # # #         # 움직이는 장애물 애니메이션
# # # # # #         obstacle_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x")
# # # # # #         obstacle_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y")
        
# # # # # #         if obstacle_x_id != -1:
# # # # # #             addr = self.model.jnt_qposadr[obstacle_x_id]
# # # # # #             self.data.qpos[addr] = 0.05 * np.sin(t * 0.5)  # 천천히 좌우 이동
            
# # # # # #         if obstacle_y_id != -1:
# # # # # #             addr = self.model.jnt_qposadr[obstacle_y_id]
# # # # # #             self.data.qpos[addr] = 0.03 * np.cos(t * 0.7)  # 천천히 앞뒤 이동
    
# # # # # #     def update(self):
# # # # # #         """시뮬레이션 업데이트"""
# # # # # #         # 주기적 환경 무작위화
# # # # # #         if self.data.time - self.last_randomize_time > self.randomize_interval:
# # # # # #             result = self.randomizer.randomize_all()
# # # # # #             print(f"🔄 Environment randomized at t={self.data.time:.1f}s")
# # # # # #             print(f"   Table position: {result['table_position']}")
# # # # # #             print(f"   All waypoints reachable: {result['all_waypoints_reachable']}")
# # # # # #             self.last_randomize_time = self.data.time
        
# # # # # #         # 장애물 애니메이션 업데이트
# # # # # #         self.update_obstacles()

# # # # # # def main():
# # # # # #     current_dir = os.path.dirname(os.path.realpath(__file__))
    
# # # # # #     # ROS2 초기화
# # # # # #     rclpy.init()
    
# # # # # #     # 파일 경로 설정
# # # # # #     xml_path = os.path.join(current_dir, '../robots', "welding_scene2.xml")
# # # # # #     urdf_path = os.path.join(current_dir, '../robots', 'fr3/fr3_hand.urdf')
    
# # # # # #     # 파일 존재 확인
# # # # # #     if not os.path.exists(xml_path):
# # # # # #         print(f"❌ XML file not found: {xml_path}")
# # # # # #         return
# # # # # #     if not os.path.exists(urdf_path):
# # # # # #         print(f"❌ URDF file not found: {urdf_path}")
# # # # # #         return
    
# # # # # #     try:
# # # # # #         print("🤖 Initializing Welding Robot Controller...")
        
# # # # # #         # Fr3 컨트롤러 초기화
# # # # # #         rc = Fr3Controller(urdf_path)
        
# # # # # #         # 로봇 및 카메라 정보 설정
# # # # # #         robot_info = [xml_path, urdf_path, 1000]  # [xml_path, urdf_path, hz]
# # # # # #         camera_info = ['hand_eye', 320, 240, 30]  # [camera_name, width, height, fps]
        
# # # # # #         print("🌉 Setting up MuJoCo-ROS Bridge...")
        
# # # # # #         # MuJoCo-ROS 브리지 초기화
# # # # # #         bridge = MujocoROSBridge(robot_info, camera_info, rc)
        
# # # # # #         print("🎲 Initializing Environment Randomizer...")
        
# # # # # #         # 환경 무작위화 클래스 초기화
# # # # # #         randomizer = WeldingEnvironmentRandomizer(bridge.model, bridge.data)
        
# # # # # #         # 초기 환경 설정
# # # # # #         result = randomizer.randomize_all()
# # # # # #         print(f"🎯 Initial setup complete:")
# # # # # #         print(f"   Table position: {result['table_position']}")
# # # # # #         print(f"   Lap angle: {result['lap_angle']:.3f} rad")
# # # # # #         print(f"   All waypoints reachable: {result['all_waypoints_reachable']}")
        
# # # # # #         # 용접 시뮬레이션 관리자 초기화
# # # # # #         welding_sim = WeldingSimulation(bridge, randomizer)
        
# # # # # #         # 브리지의 시뮬레이션 업데이트 콜백 설정
# # # # # #         original_update = bridge.update_simulation
        
# # # # # #         def enhanced_update():
# # # # # #             # 원래 시뮬레이션 업데이트
# # # # # #             original_update()
# # # # # #             # 용접 환경 업데이트
# # # # # #             welding_sim.update()
        
# # # # # #         bridge.update_simulation = enhanced_update
        
# # # # # #         print("🚀 Starting Welding Simulation with ROS Bridge...")
# # # # # #         print("💡 Available services:")
# # # # # #         print("   - /task_move_srv")
# # # # # #         print("   - /get_site_position")
# # # # # #         print("   - /get_site_orientation")
# # # # # #         print("🔧 You can now run the waypoint client!")
# # # # # #         print("   ros2 run dm_task_manager multi_pose_task_client_waypoints")
        
# # # # # #         # 브리지 실행 (ROS 서비스와 시뮬레이션 동시 실행)
# # # # # #         bridge.run()
        
# # # # # #     except KeyboardInterrupt:
# # # # # #         print("\n🛑 Shutting down welding simulation...")
# # # # # #     except Exception as e:
# # # # # #         print(f"❌ Error: {str(e)}")
# # # # # #         import traceback
# # # # # #         traceback.print_exc()
# # # # # #     finally:
# # # # # #         # 정리
# # # # # #         if 'bridge' in locals():
# # # # # #             bridge.destroy_node()
# # # # # #         rclpy.shutdown()
# # # # # #         print("✅ Welding simulation terminated.")

# # # # # # if __name__ == "__main__":    
# # # # # #     main()

# # # # # # # # /home/minjun/wr_ws/src/welding_robot/dm_ros/dm_ros/welding_free2.py
# # # # # # # import numpy as np
# # # # # # # import mujoco
# # # # # # # import mujoco.viewer

# # # # # # # class WeldingEnvironmentRandomizer:
# # # # # # #     def __init__(self, model, data):
# # # # # # #         self.model = model
# # # # # # #         self.data = data
        
# # # # # # #         # 테이블과 관련 body들의 ID 저장
# # # # # # #         self.table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
# # # # # # #         self.lap_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lap_base")
# # # # # # #         self.fillet_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fillet_joint_base")
# # # # # # #         self.curved_pipe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "curved_pipe")
# # # # # # #         self.moving_obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving_obstacle")
# # # # # # #         self.worker_torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "worker_torso")
        
# # # # # # #         # 원래 위치 저장 (상대 위치 계산용)
# # # # # # #         self.original_table_pos = model.body_pos[self.table_id].copy()
# # # # # # #         self.relative_positions = {
# # # # # # #             'lap_base': model.body_pos[self.lap_base_id] - self.original_table_pos,
# # # # # # #             'fillet_joint': model.body_pos[self.fillet_joint_id] - self.original_table_pos,
# # # # # # #             'curved_pipe': model.body_pos[self.curved_pipe_id] - self.original_table_pos,
# # # # # # #             'moving_obstacle': model.body_pos[self.moving_obstacle_id] - self.original_table_pos,
# # # # # # #             'worker_torso': model.body_pos[self.worker_torso_id] - self.original_table_pos
# # # # # # #         }
        
# # # # # # #     def randomize_table_position(self, x_range=(0.4, 0.6), y_range=(-0.1, 0.1), z_range=(0.4, 0.5)):
# # # # # # #         """테이블 위치를 무작위화하고 관련된 모든 오브젝트도 함께 이동"""
# # # # # # #         # 새로운 테이블 위치 생성
# # # # # # #         new_table_pos = np.array([
# # # # # # #             np.random.uniform(*x_range),
# # # # # # #             np.random.uniform(*y_range),
# # # # # # #             np.random.uniform(*z_range)
# # # # # # #         ])
        
# # # # # # #         # 테이블 위치 업데이트
# # # # # # #         self.model.body_pos[self.table_id] = new_table_pos
        
# # # # # # #         # 테이블 위의 모든 오브젝트들도 상대 위치 유지하며 이동
# # # # # # #         self.model.body_pos[self.lap_base_id] = new_table_pos + self.relative_positions['lap_base']
# # # # # # #         self.model.body_pos[self.fillet_joint_id] = new_table_pos + self.relative_positions['fillet_joint']
# # # # # # #         self.model.body_pos[self.curved_pipe_id] = new_table_pos + self.relative_positions['curved_pipe']
# # # # # # #         self.model.body_pos[self.moving_obstacle_id] = new_table_pos + self.relative_positions['moving_obstacle']
# # # # # # #         self.model.body_pos[self.worker_torso_id] = new_table_pos + self.relative_positions['worker_torso']
        
# # # # # # #         return new_table_pos
    
# # # # # # #     def randomize_table_rotation(self, angle_range=(-np.pi/6, np.pi/6)):
# # # # # # #         """테이블 회전을 무작위화 (Z축 기준)"""
# # # # # # #         angle = np.random.uniform(*angle_range)
# # # # # # #         # Euler angles: (roll, pitch, yaw)
# # # # # # #         self.model.body_quat[self.table_id] = self._euler_to_quat(0, 0, angle)
# # # # # # #         return angle
    
# # # # # # #     def randomize_obstacles(self):
# # # # # # #         """장애물 위치를 무작위화"""
# # # # # # #         # 움직이는 장애물의 초기 위치 설정
# # # # # # #         obstacle_joint_ids = [
# # # # # # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x"),
# # # # # # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_y"),
# # # # # # #             mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_z")
# # # # # # #         ]
        
# # # # # # #         for joint_id in obstacle_joint_ids:
# # # # # # #             if joint_id != -1:
# # # # # # #                 range_min = self.model.jnt_range[joint_id][0]
# # # # # # #                 range_max = self.model.jnt_range[joint_id][1]
# # # # # # #                 self.data.qpos[self.model.jnt_qposadr[joint_id]] = np.random.uniform(range_min, range_max)
    
# # # # # # #     def randomize_lap_joint_angle(self, angle_range=(-0.3, 0.3)):
# # # # # # #         """Lap joint의 초기 각도를 무작위화"""
# # # # # # #         lap_base_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lap_base")
# # # # # # #         if lap_base_joint != -1:
# # # # # # #             angle = np.random.uniform(*angle_range)
# # # # # # #             self.model.body_quat[self.lap_base_id] = self._euler_to_quat(0, 0, angle)
# # # # # # #             return angle
# # # # # # #         return 0
    
# # # # # # #     def check_reachability(self, site_name, robot_reach=0.855):
# # # # # # #         """특정 사이트가 로봇의 도달 범위 내에 있는지 확인"""
# # # # # # #         site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
# # # # # # #         if site_id != -1:
# # # # # # #             site_pos = self.data.site_xpos[site_id]
# # # # # # #             # 로봇 베이스를 원점으로 가정
# # # # # # #             distance = np.linalg.norm(site_pos)
# # # # # # #             return distance <= robot_reach
# # # # # # #         return False
    
# # # # # # #     def randomize_all(self):
# # # # # # #         """전체 환경을 무작위화"""
# # # # # # #         # 테이블 위치 무작위화
# # # # # # #         table_pos = self.randomize_table_position()
        
# # # # # # #         # 테이블 회전 무작위화 (선택적)
# # # # # # #         # table_angle = self.randomize_table_rotation()
        
# # # # # # #         # Lap joint 각도 무작위화
# # # # # # #         lap_angle = self.randomize_lap_joint_angle()
        
# # # # # # #         # 장애물 무작위화
# # # # # # #         self.randomize_obstacles()
        
# # # # # # #         # Forward kinematics 업데이트
# # # # # # #         mujoco.mj_forward(self.model, self.data)
        
# # # # # # #         # 도달 가능성 확인
# # # # # # #         reachable = all([
# # # # # # #             self.check_reachability("lap_start"),
# # # # # # #             self.check_reachability("lap_waypoint1"),
# # # # # # #             self.check_reachability("lap_waypoint2"),
# # # # # # #             self.check_reachability("lap_waypoint3"),
# # # # # # #             self.check_reachability("lap_end")
# # # # # # #         ])
        
# # # # # # #         return {
# # # # # # #             'table_position': table_pos,
# # # # # # #             'lap_angle': lap_angle,
# # # # # # #             'all_waypoints_reachable': reachable
# # # # # # #         }
    
# # # # # # #     def _euler_to_quat(self, roll, pitch, yaw):
# # # # # # #         """Euler angles to quaternion conversion"""
# # # # # # #         cy = np.cos(yaw * 0.5)
# # # # # # #         sy = np.sin(yaw * 0.5)
# # # # # # #         cp = np.cos(pitch * 0.5)
# # # # # # #         sp = np.sin(pitch * 0.5)
# # # # # # #         cr = np.cos(roll * 0.5)
# # # # # # #         sr = np.sin(roll * 0.5)
        
# # # # # # #         w = cr * cp * cy + sr * sp * sy
# # # # # # #         x = sr * cp * cy - cr * sp * sy
# # # # # # #         y = cr * sp * cy + sr * cp * sy
# # # # # # #         z = cr * cp * sy - sr * sp * cy
        
# # # # # # #         return np.array([w, x, y, z])


# # # # # # # # 사용 예시
# # # # # # # def main():
# # # # # # #     # 모델 로드
# # # # # # #     model = mujoco.MjModel.from_xml_path("/home/minjun/wr_ws/src/welding_robot/dm_ros/robots/welding_scene2.xml")
# # # # # # #     data = mujoco.MjData(model)
    
# # # # # # #     # Randomizer 초기화
# # # # # # #     randomizer = WeldingEnvironmentRandomizer(model, data)
    
# # # # # # #     # 환경 무작위화
# # # # # # #     result = randomizer.randomize_all()
# # # # # # #     print(f"Table position: {result['table_position']}")
# # # # # # #     print(f"Lap angle: {result['lap_angle']}")
# # # # # # #     print(f"All waypoints reachable: {result['all_waypoints_reachable']}")
    
# # # # # # #     # 시뮬레이션 실행
# # # # # # #     with mujoco.viewer.launch_passive(model, data) as viewer:
# # # # # # #         while viewer.is_running():
# # # # # # #             # 매 에피소드마다 환경 무작위화 (예시)
# # # # # # #             if data.time % 10 < 0.01:  # 10초마다
# # # # # # #                 randomizer.randomize_all()
            
# # # # # # #             # 움직이는 장애물 애니메이션 (선택적)
# # # # # # #             t = data.time
# # # # # # #             obstacle_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle_x")
# # # # # # #             if obstacle_x_id != -1:
# # # # # # #                 data.qpos[model.jnt_qposadr[obstacle_x_id]] = 0.05 * np.sin(t)
            
# # # # # # #             mujoco.mj_step(model, data)
# # # # # # #             viewer.sync()

# # # # # # # if __name__=="__main__":    
# # # # # # #     main()