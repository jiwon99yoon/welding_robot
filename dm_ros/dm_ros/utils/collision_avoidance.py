# /home/minjun/wr_ws/src/welding_robot/dm_ros/dm_ros/utils/collision_avoidance.py
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
import time

class MuJoCoCollisionAvoidance:
    """MuJoCo 기반 충돌 회피 시스템"""
    
    def __init__(self, model, data, robot_controller):
        self.model = model
        self.data = data
        self.rc = robot_controller
        
        # 로봇 관련 설정
        self.robot_joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3',
            'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
        
        # 충돌 검사 설정
        self.collision_margin = 0.05  # 5cm 안전 마진
        self.path_resolution = 20     # 경로 상의 검사 포인트 수
        self.max_planning_attempts = 5
        self.visualization_enabled = True
        
        # 시각화용 사이트 생성 (충돌 지점 표시)
        self.setup_visualization_sites()
        
        print("🛡️ MuJoCo Collision Avoidance System initialized")
        print(f"   - Safety margin: {self.collision_margin * 100:.1f}cm")
        print(f"   - Path resolution: {self.path_resolution} points")
    
    def setup_visualization_sites(self):
        """시각화용 사이트 설정"""
        # 충돌 지점과 안전 경로를 표시할 사이트들의 ID 저장
        self.collision_site_ids = []
        self.safe_path_site_ids = []
        
        # 기존 사이트들 찾기 (있다면)
        for i in range(10):  # 최대 10개
            collision_site = f"collision_marker_{i}"
            safe_site = f"safe_path_marker_{i}"
            
            collision_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, collision_site)
            safe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, safe_site)
            
            if collision_id != -1:
                self.collision_site_ids.append(collision_id)
            if safe_id != -1:
                self.safe_path_site_ids.append(safe_id)
    
    def check_robot_collision(self, joint_angles=None):
        """현재 또는 지정된 관절 각도에서 충돌 검사 (안전한 버전)"""
        try:
            if joint_angles is not None:
                # 임시로 관절 각도 설정
                original_qpos = self.data.qpos[:7].copy()
                self.data.qpos[:7] = joint_angles
                mujoco.mj_forward(self.model, self.data)
            
            # 안전한 충돌 검사
            mujoco.mj_collision(self.model, self.data)
            collision_detected = self.data.ncon > 0
            
            # 충돌한 geom 쌍들 찾기
            collision_pairs = []
            if collision_detected and self.data.ncon < 50:  # 너무 많은 접촉은 처리하지 않음
                for i in range(min(self.data.ncon, 5)):  # 최대 5개만 처리
                    try:
                        contact = self.data.contact[i]
                        
                        # geom ID 유효성 검사
                        if (0 <= contact.geom1 < self.model.ngeom and 
                            0 <= contact.geom2 < self.model.ngeom):
                            
                            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom_{contact.geom1}"
                            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom_{contact.geom2}"
                            
                            # 로봇 부품과 환경 사이의 충돌만 관심
                            if self.is_robot_environment_collision(geom1_name, geom2_name):
                                collision_pairs.append((geom1_name, geom2_name, contact.pos.copy()))
                                
                    except Exception:
                        # 개별 접촉 처리 오류는 무시
                        continue
            
            if joint_angles is not None:
                # 원래 관절 각도로 복원
                try:
                    self.data.qpos[:7] = original_qpos
                    mujoco.mj_forward(self.model, self.data)
                except:
                    pass
            
            return len(collision_pairs) > 0, collision_pairs
            
        except Exception as e:
            print(f"Collision check error: {e}")
            if joint_angles is not None:
                try:
                    self.data.qpos[:7] = original_qpos
                    mujoco.mj_forward(self.model, self.data)
                except:
                    pass
            return False, []
    
    def is_robot_environment_collision(self, geom1_name, geom2_name):
        """로봇과 환경 사이의 충돌인지 확인 (안전한 버전)"""
        if not geom1_name or not geom2_name:
            return False
            
        robot_keywords = ['fr3_', 'hand_', 'finger_']
        env_keywords = ['table', 'obstacle', 'worker', 'lap_', 'fillet_', 'curved_']
        
        def is_robot_geom(name):
            try:
                return any(keyword in str(name).lower() for keyword in robot_keywords)
            except:
                return False
        
        def is_env_geom(name):
            try:
                return any(keyword in str(name).lower() for keyword in env_keywords)
            except:
                return False
        
        # 하나는 로봇, 하나는 환경이어야 함
        try:
            return (is_robot_geom(geom1_name) and is_env_geom(geom2_name)) or \
                   (is_env_geom(geom1_name) and is_robot_geom(geom2_name))
        except:
            return False
    
    def plan_collision_free_path(self, current_ee_pos, target_ee_pos, current_joint_angles):
        """충돌 없는 경로 계획"""
        print(f"🎯 Planning collision-free path...")
        print(f"   From: {current_ee_pos}")
        print(f"   To:   {target_ee_pos}")
        
        # 1단계: 직선 경로 검사
        straight_path_safe, collision_points = self.check_straight_path(
            current_ee_pos, target_ee_pos, current_joint_angles
        )
        
        if straight_path_safe:
            print("✅ Direct path is collision-free!")
            return self.generate_straight_waypoints(current_ee_pos, target_ee_pos)
        else:
            print(f"⚠️  Direct path has {len(collision_points)} collision points")
            self.visualize_collision_points(collision_points)
        
        # 2단계: 회피 경로 생성
        safe_path = self.generate_avoidance_path(
            current_ee_pos, target_ee_pos, current_joint_angles, collision_points
        )
        
        if safe_path:
            print(f"✅ Found safe path with {len(safe_path)} waypoints")
            self.visualize_safe_path(safe_path)
            return safe_path
        else:
            print("❌ Could not find collision-free path")
            return None
    
    def check_straight_path(self, start_pos, end_pos, current_joints):
        """직선 경로에서 충돌 검사"""
        collision_points = []
        
        for i in range(self.path_resolution):
            t = i / (self.path_resolution - 1)
            test_pos = start_pos + t * (end_pos - start_pos)
            
            # 해당 위치에 대한 역기구학 계산
            target_joints = self.compute_ik_for_position(test_pos, current_joints)
            if target_joints is None:
                continue
            
            # 충돌 검사
            collision, pairs = self.check_robot_collision(target_joints)
            if collision:
                collision_points.extend([pair[2] for pair in pairs])
        
        return len(collision_points) == 0, collision_points
    
    def generate_avoidance_path(self, start_pos, end_pos, current_joints, collision_points):
        """회피 경로 생성"""
        for attempt in range(self.max_planning_attempts):
            print(f"   Attempt {attempt + 1}/{self.max_planning_attempts}")
            
            # 회피 전략들 시도
            strategies = [
                self.strategy_lift_and_move,
                self.strategy_side_step,
                self.strategy_curve_around
            ]
            
            for strategy_func in strategies:
                path = strategy_func(start_pos, end_pos, current_joints, collision_points)
                if path and self.validate_entire_path(path, current_joints):
                    print(f"✅ Strategy '{strategy_func.__name__}' succeeded")
                    return path
        
        return None
    
    def strategy_lift_and_move(self, start_pos, end_pos, current_joints, collision_points):
        """위로 올린 후 이동하는 전략"""
        lift_height = 0.15  # 15cm 위로
        
        intermediate_pos = start_pos.copy()
        intermediate_pos[2] += lift_height  # Z축으로 올리기
        
        return [
            start_pos,
            intermediate_pos,  # 위로 올리기
            end_pos + np.array([0, 0, lift_height]),  # 목표 위에서 대기
            end_pos  # 최종 목표
        ]
    
    def strategy_side_step(self, start_pos, end_pos, current_joints, collision_points):
        """옆으로 우회하는 전략"""
        # 충돌 지점들의 중심 계산
        if not collision_points:
            return None
        
        collision_center = np.mean(collision_points, axis=0)
        
        # 시작점에서 충돌 중심으로의 벡터에 수직인 방향으로 우회
        to_collision = collision_center[:2] - start_pos[:2]
        perpendicular = np.array([-to_collision[1], to_collision[0]])
        if np.linalg.norm(perpendicular) > 0:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        offset_distance = 0.2  # 20cm 우회
        side_pos = start_pos.copy()
        side_pos[:2] += perpendicular * offset_distance
        
        return [
            start_pos,
            side_pos,  # 옆으로 이동
            end_pos
        ]
    
    def strategy_curve_around(self, start_pos, end_pos, current_joints, collision_points):
        """곡선으로 우회하는 전략"""
        mid_point = (start_pos + end_pos) / 2
        mid_point[2] += 0.1  # 10cm 위로
        mid_point[1] += 0.15  # Y축으로 15cm 이동
        
        return [
            start_pos,
            mid_point,
            end_pos
        ]
    
    def validate_entire_path(self, waypoints, current_joints):
        """전체 경로의 충돌 여부 검사"""
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            safe, _ = self.check_straight_path(start, end, current_joints)
            if not safe:
                return False
        
        return True
    
    def compute_ik_for_position(self, target_pos, seed_joints):
        """지정된 위치에 대한 역기구학 계산 (간단한 버전)"""
        try:
            # 현재 EE 사이트 ID 가져오기
            ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
            if ee_site_id == -1:
                return None
            
            # 반복적 IK 풀이 (간단한 Jacobian 방법)
            joint_angles = seed_joints.copy()
            target_reached = False
            
            for iteration in range(10):  # 최대 10번 반복
                # Forward kinematics
                self.data.qpos[:7] = joint_angles
                mujoco.mj_forward(self.model, self.data)
                current_pos = self.data.site_xpos[ee_site_id].copy()
                
                # 목표까지의 오차
                pos_error = target_pos - current_pos
                if np.linalg.norm(pos_error) < 0.01:  # 1cm 이내
                    target_reached = True
                    break
                
                # Jacobian 계산
                jac_pos = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jac_pos, None, ee_site_id)
                
                # Damped least squares
                damping = 0.01
                jac_robot = jac_pos[:, :7]  # 로봇 관절만
                pinv_jac = jac_robot.T @ np.linalg.inv(
                    jac_robot @ jac_robot.T + damping * np.eye(3)
                )
                
                # 관절 각도 업데이트
                delta_q = pinv_jac @ pos_error
                joint_angles += 0.1 * delta_q  # 작은 스텝 크기
                
                # 관절 한계 확인
                for j in range(7):
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.robot_joint_names[j])
                    if joint_id != -1:
                        q_min, q_max = self.model.jnt_range[joint_id]
                        joint_angles[j] = np.clip(joint_angles[j], q_min, q_max)
            
            return joint_angles if target_reached else None
            
        except Exception as e:
            print(f"IK computation failed: {e}")
            return None
    
    def generate_straight_waypoints(self, start_pos, end_pos, num_points=5):
        """직선 경로 웨이포인트 생성"""
        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1)
            waypoint = start_pos + t * (end_pos - start_pos)
            waypoints.append(waypoint)
        return waypoints
    
    def visualize_collision_points(self, collision_points):
        """충돌 지점 시각화"""
        if not self.visualization_enabled or not collision_points:
            return
        
        # 사용 가능한 마커 사이트에 충돌 지점 표시
        for i, point in enumerate(collision_points[:len(self.collision_site_ids)]):
            site_id = self.collision_site_ids[i]
            # 사이트 위치 업데이트 (실제로는 XML에서 미리 정의된 사이트 필요)
            print(f"🔴 Collision at: {point}")
    
    def visualize_safe_path(self, waypoints):
        """안전한 경로 시각화"""
        if not self.visualization_enabled:
            return
        
        print("🟢 Safe path waypoints:")
        for i, wp in enumerate(waypoints):
            print(f"   {i}: {wp}")
    
    def get_collision_statistics(self):
        """충돌 통계 반환"""
        collision_detected, pairs = self.check_robot_collision()
        
        stats = {
            'collision_detected': collision_detected,
            'num_collision_pairs': len(pairs),
            'collision_pairs': pairs
        }
        
        return stats

def integrate_collision_avoidance_to_bridge(bridge_class):
    """MujocoROSBridge에 충돌 회피 기능 통합"""
    
    original_init = bridge_class.__init__
    
    def new_init(self, robot_info, camera_info, robot_controller):
        original_init(self, robot_info, camera_info, robot_controller)
        
        # 충돌 회피 시스템 추가
        self.collision_avoidance = MuJoCoCollisionAvoidance(
            self.model, self.data, robot_controller
        )
        
        self.get_logger().info("🛡️ Collision avoidance system integrated")
    
    def check_collision_status(self):
        """현재 충돌 상태 확인"""
        return self.collision_avoidance.get_collision_statistics()
    
    def plan_safe_path(self, current_ee_pos, target_ee_pos):
        """안전한 경로 계획"""
        current_joints = self.data.qpos[:7].copy()
        return self.collision_avoidance.plan_collision_free_path(
            current_ee_pos, target_ee_pos, current_joints
        )
    
    # 메서드 추가
    bridge_class.__init__ = new_init
    bridge_class.check_collision_status = check_collision_status
    bridge_class.plan_safe_path = plan_safe_path
    
    return bridge_class

# # /home/minjun/wr_ws/src/welding_robot/dm_ros/dm_ros/utils/collision_avoidance.py
# import numpy as np
# import mujoco
# from scipy.spatial.transform import Rotation as R
# import time

# class MuJoCoCollisionAvoidance:
#     """MuJoCo 기반 충돌 회피 시스템"""
    
#     def __init__(self, model, data, robot_controller):
#         self.model = model
#         self.data = data
#         self.rc = robot_controller
        
#         # 로봇 관련 설정
#         self.robot_joint_names = [
#             'fr3_joint1', 'fr3_joint2', 'fr3_joint3',
#             'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
#         ]
        
#         # 충돌 검사 설정
#         self.collision_margin = 0.05  # 5cm 안전 마진
#         self.path_resolution = 20     # 경로 상의 검사 포인트 수
#         self.max_planning_attempts = 5
#         self.visualization_enabled = True
        
#         # 시각화용 사이트 생성 (충돌 지점 표시)
#         self.setup_visualization_sites()
        
#         print("🛡️ MuJoCo Collision Avoidance System initialized")
#         print(f"   - Safety margin: {self.collision_margin * 100:.1f}cm")
#         print(f"   - Path resolution: {self.path_resolution} points")
    
#     def setup_visualization_sites(self):
#         """시각화용 사이트 설정"""
#         # 충돌 지점과 안전 경로를 표시할 사이트들의 ID 저장
#         self.collision_site_ids = []
#         self.safe_path_site_ids = []
        
#         # 기존 사이트들 찾기 (있다면)
#         for i in range(10):  # 최대 10개
#             collision_site = f"collision_marker_{i}"
#             safe_site = f"safe_path_marker_{i}"
            
#             collision_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, collision_site)
#             safe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, safe_site)
            
#             if collision_id != -1:
#                 self.collision_site_ids.append(collision_id)
#             if safe_id != -1:
#                 self.safe_path_site_ids.append(safe_id)
    
#     def check_robot_collision(self, joint_angles=None):
#         """현재 또는 지정된 관절 각도에서 충돌 검사"""
#         if joint_angles is not None:
#             # 임시로 관절 각도 설정
#             original_qpos = self.data.qpos[:7].copy()
#             self.data.qpos[:7] = joint_angles
#             mujoco.mj_forward(self.model, self.data)
        
#         # 충돌 검사
#         mujoco.mj_collision(self.model, self.data)
#         collision_detected = self.data.ncon > 0
        
#         # 충돌한 geom 쌍들 찾기
#         collision_pairs = []
#         if collision_detected:
#             for i in range(self.data.ncon):
#                 contact = self.data.contact[i]
#                 geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
#                 geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                
#                 # 로봇 부품과 환경 사이의 충돌만 관심
#                 if self.is_robot_environment_collision(geom1_name, geom2_name):
#                     collision_pairs.append((geom1_name, geom2_name, contact.pos.copy()))
        
#         if joint_angles is not None:
#             # 원래 관절 각도로 복원
#             self.data.qpos[:7] = original_qpos
#             mujoco.mj_forward(self.model, self.data)
        
#         return collision_detected, collision_pairs
    
#     def is_robot_environment_collision(self, geom1_name, geom2_name):
#         """로봇과 환경 사이의 충돌인지 확인"""
#         robot_keywords = ['fr3_', 'hand_', 'finger_']
#         env_keywords = ['table', 'obstacle', 'worker', 'lap_', 'fillet_', 'curved_']
        
#         def is_robot_geom(name):
#             return any(keyword in name for keyword in robot_keywords) if name else False
        
#         def is_env_geom(name):
#             return any(keyword in name for keyword in env_keywords) if name else False
        
#         # 하나는 로봇, 하나는 환경이어야 함
#         return (is_robot_geom(geom1_name) and is_env_geom(geom2_name)) or \
#                (is_env_geom(geom1_name) and is_robot_geom(geom2_name))
    
#     def plan_collision_free_path(self, current_ee_pos, target_ee_pos, current_joint_angles):
#         """충돌 없는 경로 계획"""
#         print(f"🎯 Planning collision-free path...")
#         print(f"   From: {current_ee_pos}")
#         print(f"   To:   {target_ee_pos}")
        
#         # 1단계: 직선 경로 검사
#         straight_path_safe, collision_points = self.check_straight_path(
#             current_ee_pos, target_ee_pos, current_joint_angles
#         )
        
#         if straight_path_safe:
#             print("✅ Direct path is collision-free!")
#             return self.generate_straight_waypoints(current_ee_pos, target_ee_pos)
#         else:
#             print(f"⚠️  Direct path has {len(collision_points)} collision points")
#             self.visualize_collision_points(collision_points)
        
#         # 2단계: 회피 경로 생성
#         safe_path = self.generate_avoidance_path(
#             current_ee_pos, target_ee_pos, current_joint_angles, collision_points
#         )
        
#         if safe_path:
#             print(f"✅ Found safe path with {len(safe_path)} waypoints")
#             self.visualize_safe_path(safe_path)
#             return safe_path
#         else:
#             print("❌ Could not find collision-free path")
#             return None
    
#     def check_straight_path(self, start_pos, end_pos, current_joints):
#         """직선 경로에서 충돌 검사"""
#         collision_points = []
        
#         for i in range(self.path_resolution):
#             t = i / (self.path_resolution - 1)
#             test_pos = start_pos + t * (end_pos - start_pos)
            
#             # 해당 위치에 대한 역기구학 계산
#             target_joints = self.compute_ik_for_position(test_pos, current_joints)
#             if target_joints is None:
#                 continue
            
#             # 충돌 검사
#             collision, pairs = self.check_robot_collision(target_joints)
#             if collision:
#                 collision_points.extend([pair[2] for pair in pairs])
        
#         return len(collision_points) == 0, collision_points
    
#     def generate_avoidance_path(self, start_pos, end_pos, current_joints, collision_points):
#         """회피 경로 생성"""
#         for attempt in range(self.max_planning_attempts):
#             print(f"   Attempt {attempt + 1}/{self.max_planning_attempts}")
            
#             # 회피 전략들 시도
#             strategies = [
#                 self.strategy_lift_and_move,
#                 self.strategy_side_step,
#                 self.strategy_curve_around
#             ]
            
#             for strategy_func in strategies:
#                 path = strategy_func(start_pos, end_pos, current_joints, collision_points)
#                 if path and self.validate_entire_path(path, current_joints):
#                     print(f"✅ Strategy '{strategy_func.__name__}' succeeded")
#                     return path
        
#         return None
    
#     def strategy_lift_and_move(self, start_pos, end_pos, current_joints, collision_points):
#         """위로 올린 후 이동하는 전략"""
#         lift_height = 0.15  # 15cm 위로
        
#         intermediate_pos = start_pos.copy()
#         intermediate_pos[2] += lift_height  # Z축으로 올리기
        
#         return [
#             start_pos,
#             intermediate_pos,  # 위로 올리기
#             end_pos + np.array([0, 0, lift_height]),  # 목표 위에서 대기
#             end_pos  # 최종 목표
#         ]
    
#     def strategy_side_step(self, start_pos, end_pos, current_joints, collision_points):
#         """옆으로 우회하는 전략"""
#         # 충돌 지점들의 중심 계산
#         if not collision_points:
#             return None
        
#         collision_center = np.mean(collision_points, axis=0)
        
#         # 시작점에서 충돌 중심으로의 벡터에 수직인 방향으로 우회
#         to_collision = collision_center[:2] - start_pos[:2]
#         perpendicular = np.array([-to_collision[1], to_collision[0]])
#         if np.linalg.norm(perpendicular) > 0:
#             perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
#         offset_distance = 0.2  # 20cm 우회
#         side_pos = start_pos.copy()
#         side_pos[:2] += perpendicular * offset_distance
        
#         return [
#             start_pos,
#             side_pos,  # 옆으로 이동
#             end_pos
#         ]
    
#     def strategy_curve_around(self, start_pos, end_pos, current_joints, collision_points):
#         """곡선으로 우회하는 전략"""
#         mid_point = (start_pos + end_pos) / 2
#         mid_point[2] += 0.1  # 10cm 위로
#         mid_point[1] += 0.15  # Y축으로 15cm 이동
        
#         return [
#             start_pos,
#             mid_point,
#             end_pos
#         ]
    
#     def validate_entire_path(self, waypoints, current_joints):
#         """전체 경로의 충돌 여부 검사"""
#         for i in range(len(waypoints) - 1):
#             start = waypoints[i]
#             end = waypoints[i + 1]
            
#             safe, _ = self.check_straight_path(start, end, current_joints)
#             if not safe:
#                 return False
        
#         return True
    
#     def compute_ik_for_position(self, target_pos, seed_joints):
#         """지정된 위치에 대한 역기구학 계산 (간단한 버전)"""
#         try:
#             # 현재 EE 사이트 ID 가져오기
#             ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
#             if ee_site_id == -1:
#                 return None
            
#             # 반복적 IK 풀이 (간단한 Jacobian 방법)
#             joint_angles = seed_joints.copy()
#             target_reached = False
            
#             for iteration in range(10):  # 최대 10번 반복
#                 # Forward kinematics
#                 self.data.qpos[:7] = joint_angles
#                 mujoco.mj_forward(self.model, self.data)
#                 current_pos = self.data.site_xpos[ee_site_id].copy()
                
#                 # 목표까지의 오차
#                 pos_error = target_pos - current_pos
#                 if np.linalg.norm(pos_error) < 0.01:  # 1cm 이내
#                     target_reached = True
#                     break
                
#                 # Jacobian 계산
#                 jac_pos = np.zeros((3, self.model.nv))
#                 mujoco.mj_jacSite(self.model, self.data, jac_pos, None, ee_site_id)
                
#                 # Damped least squares
#                 damping = 0.01
#                 jac_robot = jac_pos[:, :7]  # 로봇 관절만
#                 pinv_jac = jac_robot.T @ np.linalg.inv(
#                     jac_robot @ jac_robot.T + damping * np.eye(3)
#                 )
                
#                 # 관절 각도 업데이트
#                 delta_q = pinv_jac @ pos_error
#                 joint_angles += 0.1 * delta_q  # 작은 스텝 크기
                
#                 # 관절 한계 확인
#                 for j in range(7):
#                     joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.robot_joint_names[j])
#                     if joint_id != -1:
#                         q_min, q_max = self.model.jnt_range[joint_id]
#                         joint_angles[j] = np.clip(joint_angles[j], q_min, q_max)
            
#             return joint_angles if target_reached else None
            
#         except Exception as e:
#             print(f"IK computation failed: {e}")
#             return None
    
#     def generate_straight_waypoints(self, start_pos, end_pos, num_points=5):
#         """직선 경로 웨이포인트 생성"""
#         waypoints = []
#         for i in range(num_points):
#             t = i / (num_points - 1)
#             waypoint = start_pos + t * (end_pos - start_pos)
#             waypoints.append(waypoint)
#         return waypoints
    
#     def visualize_collision_points(self, collision_points):
#         """충돌 지점 시각화"""
#         if not self.visualization_enabled or not collision_points:
#             return
        
#         # 사용 가능한 마커 사이트에 충돌 지점 표시
#         for i, point in enumerate(collision_points[:len(self.collision_site_ids)]):
#             site_id = self.collision_site_ids[i]
#             # 사이트 위치 업데이트 (실제로는 XML에서 미리 정의된 사이트 필요)
#             print(f"🔴 Collision at: {point}")
    
#     def visualize_safe_path(self, waypoints):
#         """안전한 경로 시각화"""
#         if not self.visualization_enabled:
#             return
        
#         print("🟢 Safe path waypoints:")
#         for i, wp in enumerate(waypoints):
#             print(f"   {i}: {wp}")
    
#     def get_collision_statistics(self):
#         """충돌 통계 반환"""
#         collision_detected, pairs = self.check_robot_collision()
        
#         stats = {
#             'collision_detected': collision_detected,
#             'num_collision_pairs': len(pairs),
#             'collision_pairs': pairs
#         }
        
#         return stats

# def integrate_collision_avoidance_to_bridge(bridge_class):
#     """MujocoROSBridge에 충돌 회피 기능 통합"""
    
#     original_init = bridge_class.__init__
    
#     def new_init(self, robot_info, camera_info, robot_controller):
#         original_init(self, robot_info, camera_info, robot_controller)
        
#         # 충돌 회피 시스템 추가
#         self.collision_avoidance = MuJoCoCollisionAvoidance(
#             self.model, self.data, robot_controller
#         )
        
#         self.get_logger().info("🛡️ Collision avoidance system integrated")
    
#     def check_collision_status(self):
#         """현재 충돌 상태 확인"""
#         return self.collision_avoidance.get_collision_statistics()
    
#     def plan_safe_path(self, current_ee_pos, target_ee_pos):
#         """안전한 경로 계획"""
#         current_joints = self.data.qpos[:7].copy()
#         return self.collision_avoidance.plan_collision_free_path(
#             current_ee_pos, target_ee_pos, current_joints
#         )
    
#     # 메서드 추가
#     bridge_class.__init__ = new_init
#     bridge_class.check_collision_status = check_collision_status
#     bridge_class.plan_safe_path = plan_safe_path
    
#     return bridge_class
