from geometry_msgs.msg import Pose
from dm_msgs.srv import TaskMove
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import numpy as np

class MultiPoseTaskClient(Node):
    def __init__(self):
        super().__init__('multi_pose_task_client')
        self.cli = self.create_client(TaskMove, '/task_move_srv')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /task_move_srv...')

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

def pose_from_xyz_quat(xyz, quat):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = xyz
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
    return pose

def main(args=None):
    rclpy.init(args=args)
    node = MultiPoseTaskClient()

    # lap_start ~ lap_end 방향으로 선형 이동
    lap_start = np.array([0.6 - 0.2 + 0.03 - 0.15,  # table + lap_base + lap_top + site offset
                          0.0 - 0.15 + 0.02 - 0.05,
                          0.3 + 0.04 + 0.005 + 0.005])
    lap_end = np.array([0.6 - 0.2 + 0.03 + 0.15,
                        0.0 - 0.15 + 0.02 - 0.05,
                        0.3 + 0.04 + 0.005 + 0.005])
    
    # orientation: z-axis down (identity quaternion rotated 180 deg on x-axis)
    quat = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()

    waypoints = []
    n_points = 5
    for i in range(n_points):
        interp = lap_start + (lap_end - lap_start) * (i / (n_points - 1))
        pose = pose_from_xyz_quat(interp, quat)
        waypoints.append(pose)

    duration = 2.0  # seconds per waypoint

    for pose in waypoints:
        node.send_pose(pose, duration)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
