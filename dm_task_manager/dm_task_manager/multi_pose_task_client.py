import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
from dm_msgs.srv import TaskMove, GetSitePosition, GetSiteOrientation # <- 새로운 서비스 메세지 필요

import numpy as np
import time

def pose_from_xyz_quat(xyz, quat):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = xyz
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
    return pose
# pose_from_xyz_quat 함수는 주어진 위치(xyz)와 쿼터니언(quat)을 사용하여 Pose 메시지를 생성합니다.

class MultiPoseTaskClient(Node):
    def __init__(self):
        super().__init__('multi_pose_task_client')
        # TaskMove 서비스 
        self.cli = self.create_client(TaskMove, '/task_move_srv')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /task_move_srv...')
       
        # Site 위치 서비스
        self.pos_cli = self.create_client(GetSitePosition, '/get_site_position')
        while not self.pos_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /get_site_position...')

        # Site 자세 서비스
        self.orient_cli = self.create_client(
            GetSiteOrientation, '/get_site_orientation'
        )
        while not self.orient_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /get_site_orientation...')
    
    def send_pose(self, pose: Pose, duration: float):
        req = TaskMove.Request()
        req.pose = pose
        req.duration = duration
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().is_received:
            self.get_logger().info('Pose command sent successfully.')
        else:
            self.get_logger().error('Failed to send pose command.')

    # 사이트 위치를 가져오는 메소드
    def get_site_position(self, site_name: str) -> np.ndarray:
        req = GetSitePosition.Request()
        req.site_name = site_name
        future = self.pos_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result():
            pos = future.result().position
            return np.array([pos.x, pos.y, pos.z])
        else:
            self.get_logger().error(f"Failed to get site position: {site_name}")
            return np.zeros(3)
    
    # 사이트 자세를 가져오는 메소드
    def get_site_orientation(self, site_name: str) -> np.ndarray:
        req = GetSiteOrientation.Request()
        req.site_name = site_name
        future = self.orient_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result():
            ori = future.result().orientation
            return np.array([ori.x, ori.y, ori.z, ori.w])
        else:
            self.get_logger().error(f"Failed to get orientation of '{site_name}'")
            return np.array([0.0, 0.0, 0.0, 1.0])
        
def main(args=None):
    rclpy.init(args=args)
    node = MultiPoseTaskClient()

    '''# lap_start ~ lap_end 방향으로 선형 이동
    lap_start = np.array([0.6 - 0.2 + 0.03 - 0.15,  # table + lap_base + lap_top + site offset
                          0.0 - 0.15 + 0.02 - 0.05,
                          0.3 + 0.04 + 0.005 + 0.005])
    lap_end = np.array([0.6 - 0.2 + 0.03 + 0.15,
                        0.0 - 0.15 + 0.02 - 0.05,
                        0.3 + 0.04 + 0.005 + 0.005])
    '''

    '''처음 짠거 - 얜 로테이션 이랑 반영 안됨, 그냥 쭉 직진 /
    # 1) EE의 절대 위치/자세
    ee_pos = node.get_site_position("ee_site")
    ee_quat = node.get_site_orientation("ee_site")
    R_ee_world = R.from_quat(ee_quat).as_matrix()

    # 2) lap joint 절대 시작/끝 위치
    lap_start_abs = node.get_site_position("lap_start")
    lap_end_abs   = node.get_site_position("lap_end")

    # 3) EE 프레임 기준의 상대 오프셋으로 변환
    lap_start_rel = R_ee_world.T.dot(lap_start_abs - ee_pos)
    lap_end_rel   = R_ee_world.T.dot(lap_end_abs   - ee_pos)

    # 4) EE가 지향할 orientation (z-axis down)
    target_quat = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()

    # 5) quintic 스플라인 대신 단순 선형보간 웨이포인트
    n_points = 5
    waypoints = []
    for i in range(n_points):
        alpha = i / (n_points - 1)
        pt = lap_start_rel + (lap_end_rel - lap_start_rel) * alpha
        waypoints.append(pose_from_xyz_quat(pt, target_quat))

    # 6) 순차 전송 + 대기
    duration = 2.0  # seconds per waypoint
    for pose in waypoints:
        node.send_pose(pose, duration)
        time.sleep(duration + 0.05)'''
    
    # 아래 수정 코드

    duration = 2.0  # 각 이동에 2초
    
    # 1) EE 현재 위치/자세
    ee_pos  = node.get_site_position("ee_site")
    ee_quat = node.get_site_orientation("ee_site")
    R_ee    = R.from_quat(ee_quat).as_matrix()

    # 2) lap_start 위치/자세
    start_pos  = node.get_site_position("lap_start")
    start_quat = node.get_site_orientation("lap_start")
    # → EE 프레임 기준 상대 오프셋
    rel_start = R_ee.T.dot(start_pos - ee_pos)
    # (orientation은 Z축 아래로 향하도록 고정)
    target_quat = R.from_euler('xyz', [np.pi,0,0]).as_quat()

    # 3) 첫 이동: relative offset → lap_start
    node.send_pose(pose_from_xyz_quat(rel_start, target_quat), duration)
    time.sleep(duration + 10.0) # 1.0초 대기 - 0.1초 대기하면 막 이상해짐
    #10초 대기 -> jointstate값뽑기위해서 echo joint_state

    # 4) 움직인 뒤 EE 위치/자세 다시 읽기
    ee_pos  = node.get_site_position("ee_site")
    ee_quat = node.get_site_orientation("ee_site")
    R_ee    = R.from_quat(ee_quat).as_matrix()

    # 5) lap_end 위치/자세
    end_pos  = node.get_site_position("lap_end")
    end_quat = node.get_site_orientation("lap_end")
    rel_end  = R_ee.T.dot(end_pos - ee_pos)

    # 6) 두 번째 이동: relative offset → lap_end
    node.send_pose(pose_from_xyz_quat(rel_end, target_quat), duration)
    time.sleep(duration + 10.0) # 1.0초 대기 - 0.1초 대기하면 막 이상해짐 
    #10초 대기 -> jointstate값뽑기위해서 echo joint_state

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
