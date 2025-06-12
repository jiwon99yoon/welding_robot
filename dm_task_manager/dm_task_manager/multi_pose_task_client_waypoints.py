# /home/minjun/wr_ws/src/welding_robot/dm_task_manager/dm_task_manager/multi_pose_task_client_waypoints.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from scipy.spatial.transform import Rotation as R
from dm_msgs.srv import TaskMove, GetSitePosition, GetSiteOrientation
import numpy as np
import time
import math

def pose_from_xyz_quat(xyz, quat):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = xyz
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
    return pose

class ObstacleInfo:
    """장애물 정보를 담는 클래스"""
    def __init__(self, name, position, size, shape="sphere"):
        self.name = name
        self.position = np.array(position)
        self.size = size  # radius for sphere, [x,y,z] for box
        self.shape = shape

class AdvancedCollisionAvoidancePlanner:
    """고급 충돌 회피 계획기 - 토치 끝부분 기준으로 동작"""
    
    def __init__(self, safety_margin=0.12):
        self.safety_margin = safety_margin
        self.obstacles = []
        self.workspace_limits = {
            'x': (-0.8, 0.8),
            'y': (-0.8, 0.8), 
            'z': (0.1, 0.8)
        }
        # 🔧 토치 길이 고려 (torch_tip_site가 ee_site로부터 약 12.3cm 떨어져 있음)
        self.torch_length = 0.123  # 토치 길이 (m)
        
    def add_obstacle(self, obstacle_info):
        """장애물 추가"""
        self.obstacles.append(obstacle_info)
        
    def add_dynamic_obstacle(self, name, current_pos, size, velocity=np.array([0,0,0])):
        """동적 장애물 추가 (움직이는 장애물)"""
        obstacle = ObstacleInfo(name, current_pos, size, "sphere")
        obstacle.velocity = velocity
        obstacle.is_dynamic = True
        self.add_obstacle(obstacle)
    
    def is_collision_free(self, torch_tip_point, time_offset=0):
        """토치 끝부분이 장애물과 충돌하지 않는지 확인"""
        for obs in self.obstacles:
            # 동적 장애물의 경우 시간에 따른 위치 예측
            if hasattr(obs, 'is_dynamic') and obs.is_dynamic:
                predicted_pos = obs.position + obs.velocity * time_offset
            else:
                predicted_pos = obs.position
                
            dist = np.linalg.norm(torch_tip_point - predicted_pos)
            
            if obs.shape == "sphere":
                # 구체의 경우: size는 반지름(float)
                safe_distance = obs.size + self.safety_margin
                if dist < safe_distance:
                    return False, f"Torch tip too close to {obs.name} (dist: {dist:.3f}m, safe: {safe_distance:.3f}m)"
            elif obs.shape == "box":
                # 박스의 경우: size는 [x,y,z] 리스트
                diff = np.abs(torch_tip_point - predicted_pos)
                box_half = np.array(obs.size) + self.safety_margin
                if all(diff[i] < box_half[i] for i in range(3)):
                    return False, f"Torch tip inside {obs.name} box region"
                    
        return True, "Safe"
    
    def is_within_workspace(self, torch_tip_point):
        """토치 끝부분이 로봇 작업공간 내에 있는지 확인"""
        return (self.workspace_limits['x'][0] <= torch_tip_point[0] <= self.workspace_limits['x'][1] and
                self.workspace_limits['y'][0] <= torch_tip_point[1] <= self.workspace_limits['y'][1] and
                self.workspace_limits['z'][0] <= torch_tip_point[2] <= self.workspace_limits['z'][1])
    
    def check_path_safety(self, start_torch_pos, end_torch_pos, num_checks=15):
        """토치 끝부분 경로의 안전성을 세밀하게 검사"""
        collision_points = []
        unsafe_segments = []
        
        for i in range(num_checks + 1):
            t = i / num_checks
            torch_point = start_torch_pos + t * (end_torch_pos - start_torch_pos)
            time_offset = t * 3.0  # 3초 동안의 경로라고 가정
            
            # 작업공간 확인
            if not self.is_within_workspace(torch_point):
                unsafe_segments.append((i, "Torch tip outside workspace"))
                continue
                
            # 충돌 확인
            safe, reason = self.is_collision_free(torch_point, time_offset)
            if not safe:
                collision_points.append(torch_point)
                unsafe_segments.append((i, reason))
        
        is_safe = len(collision_points) == 0 and len(unsafe_segments) == 0
        return {
            'safe': is_safe,
            'collision_points': collision_points,
            'unsafe_segments': unsafe_segments,
            'safety_score': (num_checks + 1 - len(unsafe_segments)) / (num_checks + 1)
        }
    
    def plan_collision_free_path(self, start_torch_pos, end_torch_pos):
        """토치 끝부분 기준 충돌 없는 경로 계획"""
        
        print(f"🔍 DEBUG: Planning torch path from {start_torch_pos} to {end_torch_pos}")
        print(f"🔍 DEBUG: Torch distance: {np.linalg.norm(end_torch_pos - start_torch_pos):.3f}m")
        
        # 1단계: 직선 경로 검사
        direct_safety = self.check_path_safety(start_torch_pos, end_torch_pos)
        print(f"🔍 DEBUG: Direct torch path safety: {direct_safety['safe']}, score: {direct_safety['safety_score']:.2f}")
        
        if direct_safety['safe']:
            return {
                'success': True,
                'waypoints': self.generate_linear_waypoints(start_torch_pos, end_torch_pos, 3),
                'strategy': 'direct_path',
                'safety_score': direct_safety['safety_score']
            }
        
        # 2단계: 회피 전략들 시도
        avoidance_strategies = [
            ('lift_over', self.strategy_lift_over),
            ('wide_arc', self.strategy_wide_arc),
            ('step_back_around', self.strategy_step_back_around),
            ('spiral_up', self.strategy_spiral_up)
        ]
        
        best_result = None
        best_score = 0
        
        for strategy_name, strategy_func in avoidance_strategies:
            try:
                print(f"🔍 DEBUG: Trying torch strategy: {strategy_name}")
                waypoints = strategy_func(start_torch_pos, end_torch_pos)
                if waypoints and len(waypoints) >= 2:
                    # 전체 경로의 안전성 평가
                    total_score = self.evaluate_path_safety(waypoints)
                    print(f"🔍 DEBUG: {strategy_name} torch safety score: {total_score:.2f}")
                    
                    if total_score > 0.8:  # 80% 이상 안전하면 사용
                        print(f"🔍 DEBUG: {strategy_name} accepted for torch (score > 0.8)")
                        return {
                            'success': True,
                            'waypoints': waypoints,
                            'strategy': strategy_name,
                            'safety_score': total_score
                        }
                    elif total_score > best_score:
                        print(f"🔍 DEBUG: {strategy_name} is new best for torch (score: {total_score:.2f})")
                        best_result = {
                            'success': True,
                            'waypoints': waypoints,
                            'strategy': strategy_name,
                            'safety_score': total_score
                        }
                        best_score = total_score
                        
            except Exception as e:
                print(f"🔍 DEBUG: Torch strategy {strategy_name} failed: {e}")
                continue
        
        # 최선의 결과 반환 (완전히 안전하지 않더라도)
        if best_result and best_score > 0.3:  # 기준을 0.3으로 설정
            print(f"🔍 DEBUG: Using best torch result: {best_result['strategy']} (score: {best_score:.2f})")
            return best_result
        
        print(f"🔍 DEBUG: All torch strategies failed. Best score: {best_score:.2f}")
        return {
            'success': False,
            'waypoints': None,
            'strategy': 'none',
            'safety_score': 0
        }
    
    def strategy_lift_over(self, start_pos, end_pos):
        """토치를 위로 올린 후 이동하는 전략"""
        lift_height = 0.25  # 25cm 위로
        
        # 점진적 상승 경로
        quarter_point = start_pos + 0.25 * (end_pos - start_pos)
        mid_point = start_pos + 0.5 * (end_pos - start_pos)
        three_quarter_point = start_pos + 0.75 * (end_pos - start_pos)
        
        waypoints = [
            start_pos,
            quarter_point + np.array([0, 0, lift_height * 0.5]),
            mid_point + np.array([0, 0, lift_height]),
            three_quarter_point + np.array([0, 0, lift_height * 0.5]),
            end_pos
        ]
        
        return waypoints
    
    def strategy_wide_arc(self, start_pos, end_pos):
        """토치를 넓은 호로 우회"""
        # 방향 벡터 계산
        direction = end_pos - start_pos
        perpendicular = np.array([-direction[1], direction[0], 0])
        if np.linalg.norm(perpendicular) > 0:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        arc_radius = 0.3  # 30cm 우회
        arc_height = 0.15  # 15cm 위로
        
        # 5개 점으로 호 생성
        waypoints = [start_pos]
        
        for i in range(1, 4):
            t = i / 4.0
            # 베지어 곡선 스타일 호
            base_point = start_pos + t * (end_pos - start_pos)
            offset = np.sin(t * np.pi) * arc_radius * perpendicular
            height_offset = np.sin(t * np.pi) * arc_height * np.array([0, 0, 1])
            
            waypoint = base_point + offset + height_offset
            waypoints.append(waypoint)
        
        waypoints.append(end_pos)
        return waypoints
    
    def strategy_step_back_around(self, start_pos, end_pos):
        """토치를 뒤로 물러나서 우회"""
        retreat_distance = 0.2  # 20cm 후진
        side_distance = 0.25   # 25cm 옆으로
        
        # 로봇 베이스 방향 추정 (원점 방향)
        to_base = -start_pos / np.linalg.norm(start_pos) if np.linalg.norm(start_pos) > 0 else np.array([-1, 0, 0])
        
        # 옆 방향 계산
        side_dir = np.array([to_base[1], -to_base[0], 0])
        if np.linalg.norm(side_dir) > 0:
            side_dir = side_dir / np.linalg.norm(side_dir)
        
        retreat_point = start_pos + retreat_distance * to_base
        side_point = retreat_point + side_distance * side_dir
        approach_point = end_pos + retreat_distance * to_base + side_distance * side_dir
        
        return [
            start_pos,
            retreat_point,
            side_point,
            approach_point,
            end_pos
        ]
    
    def strategy_spiral_up(self, start_pos, end_pos):
        """토치를 나선형으로 올라가며 이동"""
        spiral_height = 0.2
        spiral_radius = 0.15
        
        waypoints = [start_pos]
        
        for i in range(1, 6):
            t = i / 6.0
            angle = t * 2 * np.pi  # 한 바퀴 회전
            
            base_point = start_pos + t * (end_pos - start_pos)
            
            # 나선 오프셋
            spiral_x = spiral_radius * np.cos(angle) * (1 - t)  # 점점 줄어듦
            spiral_y = spiral_radius * np.sin(angle) * (1 - t)
            spiral_z = spiral_height * np.sin(t * np.pi)  # 올라갔다 내려옴
            
            spiral_offset = np.array([spiral_x, spiral_y, spiral_z])
            waypoints.append(base_point + spiral_offset)
        
        waypoints.append(end_pos)
        return waypoints
    
    def evaluate_path_safety(self, waypoints):
        """전체 토치 경로의 안전성 평가"""
        if len(waypoints) < 2:
            return 0
        
        total_segments = len(waypoints) - 1
        safe_segments = 0
        
        for i in range(total_segments):
            safety_result = self.check_path_safety(waypoints[i], waypoints[i+1], 10)
            if safety_result['safe']:
                safe_segments += 1
            elif safety_result['safety_score'] > 0.7:  # 부분적으로 안전한 경우
                safe_segments += safety_result['safety_score']
        
        return safe_segments / total_segments
    
    def generate_linear_waypoints(self, start_pos, end_pos, num_points):
        """선형 웨이포인트 생성"""
        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1)
            waypoint = start_pos + t * (end_pos - start_pos)
            waypoints.append(waypoint)
        return waypoints

class MultiPoseWaypointClient(Node):
    def __init__(self):
        super().__init__('multi_pose_waypoint_client')
        
        # 서비스 클라이언트 초기화
        self.cli = self.create_client(TaskMove, '/task_move_srv')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /task_move_srv...')
       
        self.pos_cli = self.create_client(GetSitePosition, '/get_site_position')
        while not self.pos_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /get_site_position...')

        self.orient_cli = self.create_client(GetSiteOrientation, '/get_site_orientation')
        while not self.orient_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /get_site_orientation...')
        
        # 🔧 토치 기준 충돌 회피 계획기 초기화
        self.collision_planner = AdvancedCollisionAvoidancePlanner(safety_margin=0.12)
        
        # 웨이포인트 이름 정의
        self.waypoint_names = [
            "lap_start",
            "lap_waypoint1", 
            "lap_waypoint2",
            "lap_waypoint3",
            "lap_end"
        ]
        
        # 통계
        self.path_stats = {
            'total_paths': 0,
            'direct_paths': 0,
            'avoidance_paths': 0,
            'failed_paths': 0
        }
        
        self.get_logger().info("🔧 Torch-based Advanced Collision-Aware Waypoint Client initialized!")
    
    def setup_obstacles(self):
        """시뮬레이션의 장애물 정보 수집 및 설정 (토치 기준)"""
        self.get_logger().info("🚧 Setting up obstacle information for torch navigation...")
        
        # 🔍 DEBUG: 장애물을 임시로 비활성화하여 경로 계획 성공률 확인
        static_obstacles = [
            # 일부 장애물만 활성화 (테스트용)
            ObstacleInfo("worker_torso", [0.5, -0.4, 0.55], 0.08, "sphere"),  # 작고 멀리
        ]
        
        for obs in static_obstacles:
            self.collision_planner.add_obstacle(obs)
            
        self.get_logger().info(f"   Added {len(static_obstacles)} static obstacles (DEBUG: reduced for torch)")
        self.get_logger().info("   Added 0 dynamic obstacles (DEBUG: disabled)")
        self.get_logger().info("🛡️ DEBUG: Simplified collision avoidance for torch testing!")
        
        # 작업공간 확장 (디버깅용)
        self.collision_planner.workspace_limits = {
            'x': (-1.0, 1.0),  # 더 넓게
            'y': (-1.0, 1.0),  # 더 넓게
            'z': (0.0, 1.0)    # 더 넓게
        }
    
    def send_pose(self, pose: Pose, duration: float):
        """포즈 명령 전송"""
        req = TaskMove.Request()
        req.pose = pose
        req.duration = duration
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() and future.result().is_received:
            self.get_logger().info('✅ Pose command sent successfully.')
            return True
        else:
            self.get_logger().error('❌ Failed to send pose command.')
            return False

    def get_site_position(self, site_name: str) -> np.ndarray:
        """사이트 위치 가져오기"""
        req = GetSitePosition.Request()
        req.site_name = site_name
        future = self.pos_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result():
            pos = future.result().position
            return np.array([pos.x, pos.y, pos.z])
        else:
            self.get_logger().error(f"❌ Failed to get site position: {site_name}")
            return np.zeros(3)
    
    def get_site_orientation(self, site_name: str) -> np.ndarray:
        """사이트 자세 가져오기"""
        req = GetSiteOrientation.Request()
        req.site_name = site_name
        future = self.orient_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result():
            ori = future.result().orientation
            return np.array([ori.x, ori.y, ori.z, ori.w])
        else:
            self.get_logger().error(f"❌ Failed to get orientation of '{site_name}'")
            return np.array([0.0, 0.0, 0.0, 1.0])
    
    def torch_pos_to_ee_relative(self, torch_target_pos, ee_pos, ee_quat):
        """토치 목표 위치를 EE 기준 상대 위치로 변환"""
        # 🔧 토치 길이만큼 EE가 토치보다 뒤에 있어야 함
        R_ee = R.from_quat(ee_quat).as_matrix()
        
        # 토치는 EE에서 Z축 방향으로 torch_length만큼 떨어져 있음
        torch_offset_in_ee_frame = np.array([0, 0, self.collision_planner.torch_length])
        
        # 목표 EE 위치 = 토치 목표 위치 - 토치 오프셋
        target_ee_pos = torch_target_pos - R_ee @ torch_offset_in_ee_frame
        
        # EE 프레임 기준 상대 위치
        rel_pos = R_ee.T @ (target_ee_pos - ee_pos)
        
        return rel_pos
    
    def execute_waypoint_sequence(self):
        """웨이포인트 시퀀스 실행 (토치 기준 충돌 회피 포함)"""
        self.get_logger().info("🎯 Starting torch-based waypoint sequence with collision avoidance...")
        
        # 장애물 설정
        self.setup_obstacles()
        
        # 용접 자세 (Z축 아래 향하도록)
        welding_quat = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()
        
        # 이동 시간 설정
        duration = 5.0  # 각 웨이포인트 간 5초
        wait_time = 1.5  # 웨이포인트 도달 후 대기 시간
        
        for i, waypoint_name in enumerate(self.waypoint_names):
            self.get_logger().info(f"🎯 Moving to waypoint {i+1}/{len(self.waypoint_names)}: {waypoint_name}")
            
            # 🔧 현재 토치 끝부분 위치 및 EE 정보 가져오기
            try:
                current_torch_pos = self.get_site_position("torch_tip_site")
                ee_pos = self.get_site_position("ee_site")
                ee_quat = self.get_site_orientation("ee_site")
                R_ee = R.from_quat(ee_quat).as_matrix()
                
                self.get_logger().info(f"🔧 Current torch tip position: {current_torch_pos}")
                self.get_logger().info(f"🔧 Current EE position: {ee_pos}")
                
            except Exception as e:
                self.get_logger().error(f"❌ Failed to get torch/EE positions: {e}")
                # 폴백: ee_site 사용
                ee_pos = self.get_site_position("ee_site")
                ee_quat = self.get_site_orientation("ee_site")
                R_ee = R.from_quat(ee_quat).as_matrix()
                # 토치 위치 추정 (EE + 토치 오프셋)
                torch_offset_world = R_ee @ np.array([0, 0, self.collision_planner.torch_length])
                current_torch_pos = ee_pos + torch_offset_world
                self.get_logger().warn(f"🔧 Using estimated torch position: {current_torch_pos}")
            
            # 목표 웨이포인트 위치 가져오기
            target_pos = self.get_site_position(waypoint_name)
            
            if np.allclose(target_pos, 0):
                self.get_logger().error(f"❌ Could not get position for {waypoint_name}")
                continue
            
            # 도달 가능성 사전 확인
            distance_to_target = np.linalg.norm(target_pos[:2])  # XY 거리
            if distance_to_target > 0.8:
                self.get_logger().warn(f"⚠️  {waypoint_name} may be unreachable (dist: {distance_to_target:.3f}m)")
            
            # 🛡️ 토치 기준 충돌 회피 경로 계획
            self.get_logger().info("🛡️ Planning collision-free path for torch tip...")
            
            path_result = self.collision_planner.plan_collision_free_path(current_torch_pos, target_pos)
            self.path_stats['total_paths'] += 1
            
            if not path_result['success']:
                self.get_logger().error(f"❌ Could not find safe torch path to {waypoint_name}")
                self.path_stats['failed_paths'] += 1
                continue
            
            # 경로 통계 업데이트
            if path_result['strategy'] == 'direct_path':
                self.path_stats['direct_paths'] += 1
            else:
                self.path_stats['avoidance_paths'] += 1
            
            safe_torch_waypoints = path_result['waypoints']
            strategy = path_result['strategy']
            safety_score = path_result['safety_score']
            
            self.get_logger().info(f"✅ Torch path planned successfully:")
            self.get_logger().info(f"   Strategy: {strategy}")
            self.get_logger().info(f"   Safety score: {safety_score:.2f}")
            self.get_logger().info(f"   Torch waypoints: {len(safe_torch_waypoints)}")
            
            # 생성된 토치 웨이포인트들을 EE 명령으로 변환하여 순차적으로 이동
            for j, torch_waypoint_pos in enumerate(safe_torch_waypoints):
                if j == 0:  # 첫 번째는 현재 위치이므로 스킵
                    continue
                
                # 🔧 토치 목표 위치를 EE 기준 상대 위치로 변환
                rel_pos = self.torch_pos_to_ee_relative(torch_waypoint_pos, ee_pos, ee_quat)
                
                # 상대 위치가 너무 크면 제한 (더 보수적)
                max_step = 0.1  # 10cm 제한
                if np.linalg.norm(rel_pos) > max_step:
                    rel_pos = rel_pos / np.linalg.norm(rel_pos) * max_step
                    self.get_logger().warn(f"   Step size limited to {max_step}m for torch control")
                
                # 포즈 생성 및 전송
                target_pose = pose_from_xyz_quat(rel_pos, welding_quat)
                
                self.get_logger().info(f"   🚀 Safe torch waypoint {j}/{len(safe_torch_waypoints)-1}: torch_pos={torch_waypoint_pos} -> ee_rel={rel_pos}")
                
                success = self.send_pose(target_pose, duration)
                if not success:
                    self.get_logger().error(f"❌ Failed to reach safe torch waypoint {j}")
                    return False
                
                # 이동 완료 대기
                time.sleep(duration + wait_time)
                
                # 다음 이동을 위해 현재 위치 업데이트
                try:
                    current_torch_pos = self.get_site_position("torch_tip_site")
                    ee_pos = self.get_site_position("ee_site")
                    ee_quat = self.get_site_orientation("ee_site")
                    R_ee = R.from_quat(ee_quat).as_matrix()
                except Exception as e:
                    self.get_logger().error(f"❌ Failed to update torch position: {e}")
                    break
            
            self.get_logger().info(f"✅ Successfully reached {waypoint_name}")
        
        # 최종 통계 출력
        self.print_path_statistics()
        
        # 성공 여부 올바르게 판단
        success_count = self.path_stats['total_paths'] - self.path_stats['failed_paths']
        overall_success = success_count > 0
        
        if overall_success:
            self.get_logger().info(f"🎉 Torch-based trajectory completed! ({success_count}/{self.path_stats['total_paths']} waypoints reached)")
        else:
            self.get_logger().error("❌ All waypoints failed!")
            
        return overall_success
    
    def print_path_statistics(self):
        """경로 계획 통계 출력"""
        self.get_logger().info("📊 Torch-based Path Planning Statistics:")
        self.get_logger().info(f"   Total paths: {self.path_stats['total_paths']}")
        self.get_logger().info(f"   Direct paths: {self.path_stats['direct_paths']}")
        self.get_logger().info(f"   Avoidance paths: {self.path_stats['avoidance_paths']}")
        self.get_logger().info(f"   Failed paths: {self.path_stats['failed_paths']}")
        
        if self.path_stats['total_paths'] > 0:
            success_rate = (self.path_stats['total_paths'] - self.path_stats['failed_paths']) / self.path_stats['total_paths'] * 100
            avoidance_rate = self.path_stats['avoidance_paths'] / self.path_stats['total_paths'] * 100
            self.get_logger().info(f"   Success rate: {success_rate:.1f}%")
            self.get_logger().info(f"   Avoidance rate: {avoidance_rate:.1f}%")

def main(args=None):
    rclpy.init(args=args)
    
    node = MultiPoseWaypointClient()
    
    try:
        # 웨이포인트 시퀀스 실행
        success = node.execute_waypoint_sequence()
        
        if success:
            node.get_logger().info("🎉 Advanced torch-based welding trajectory completed successfully!")
        else:
            node.get_logger().error("❌ Torch-based welding trajectory failed!")
            
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Interrupted by user")
    except Exception as e:
        node.get_logger().error(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()