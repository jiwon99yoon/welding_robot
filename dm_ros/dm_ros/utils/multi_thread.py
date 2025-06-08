import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer as mj_view
import threading
import time
from .scene_monitor import SceneMonitor
from .image_publisher import MujocoCameraBridge
import numpy as np


# import 추가
from dm_msgs.srv import GetSitePosition, GetSiteOrientation
from geometry_msgs.msg import Point, Quaternion
from scipy.spatial.transform import Rotation as R

# jointstate 뽑기 위해 -> 렉 많이 걸려서 비활성화
#JointState publisher 추가
from sensor_msgs.msg import JointState

class MujocoROSBridge(Node):
    def __init__(self, robot_info, camera_info, robot_controller):
        super().__init__('mujoco_ros_bridge')

        # joint names & publisher
        self.joint_names = [
            'fr3_joint1','fr3_joint2','fr3_joint3',
            'fr3_joint4','fr3_joint5','fr3_joint6','fr3_joint7'
        ]
        self.joint_state_pub = self.create_publisher(
            JointState, '/fr3/joint_states', 10
        )

        # robot_info = [xml, urdf, hz]
        self.xml_path = robot_info[0]
        self.urdf_path = robot_info[1]
        self.ctrl_freq = robot_info[2]

        # camera_info = [name, width, height, fps]
        self.camera_name = camera_info[0]
        self.width = camera_info[1]
        self.height = camera_info[2]
        self.fps = camera_info[3]
          
        self.rc = robot_controller

        # Mujoco 모델 로드
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.dt = 1 / self.ctrl_freq
        self.model.opt.timestep = self.dt
       
        self.sm = SceneMonitor(self.model, self.data)

        self.hand_eye = MujocoCameraBridge(self.model, camera_info)
        self.top_down_cam = MujocoCameraBridge(self.model, ['top_down_cam', 640, 480, self.fps])
        
        # self.renderer = mujoco.Renderer(self.model, width=self.width, height=self.height)

        self.ctrl_dof = 8 # 7 + 1
        self.ctrl_step = 0

        # 스레드 실행
        self.running = True
        self.lock = threading.Lock()
        self.robot_thread = threading.Thread(target=self.robot_control, daemon=True)
        # self.hand_eye_thread = threading.Thread(target=self.hand_eye_control, daemon=True)
        # self.td_cam_thread = threading.Thread(target=self.td_cam_control, daemon=True)
        
        # GetSitePosition 서비스 등록
        self.get_logger().info("Registering /get_site_position service")
        self.site_srv = self.create_service(
            GetSitePosition,
            '/get_site_position',
            self.get_site_position_callback  # 아래에 정의할 콜백
        )

        # --- GetSiteOrientation 서비스 등록 추가 ---
        self.get_logger().info("Registering /get_site_orientation service")
        self.orient_srv = self.create_service(
            GetSiteOrientation,
            '/get_site_orientation',
            self.get_site_orientation_callback
        )


    # visualize thread = main thread
    def run(self):        
        try:     
            with mj_view.launch_passive(self.model, self.data) as viewer:            
                # self.sm.getAllObject()        
                # self.sm.getTargetObject()       
                # self.sm.getSensor() 
                self.robot_thread.start()    
                # self.hand_eye_thread.start()
                # self.td_cam_thread.start()

                while self.running and viewer.is_running():   
                    start_time = time.perf_counter()       

                    with self.lock:                        

                        viewer.sync()  # 화면 업데이트          

                    self.time_sync(1/self.fps, start_time, False)

                    # print(time.perf_counter())
                    # print("Time in main node: ", self.ctrl_step)

                    
        except KeyboardInterrupt:
            print("\nSimulation interrupted. Closing viewer...")
            self.running = False
            self.robot_thread.join()
            # self.hand_eye_thread.join()
            # self.td_cam_thread.join()
            self.sm.destroy_node()


    def robot_control(self):
        self.ctrl_step = 0
        # sync_step = int(1/self.fps*self.ctrl_freq)
        sync_step = 30 # evey 30 ctrl_steps

        try:
            while rclpy.ok() and self.running:            
                with self.lock:
                    start_time = time.perf_counter()                        

                    mujoco.mj_step(self.model, self.data)  # 시뮬레이션 실행

                    self.rc.updateModel(self.data, self.ctrl_step)
                    
                    # -------------------- ADD Controller ---------------------------- #
                    rclpy.spin_once(self.rc, timeout_sec=0.0001) # for scene monitor
                    rclpy.spin_once(self, timeout_sec=0.0001) # for robot controller
                    self.data.ctrl[:self.ctrl_dof] = self.rc.compute()   

                    # --- publish joint positions ---
                    js = JointState()
                    js.header.stamp = self.get_clock().now().to_msg()
                    js.name = self.joint_names
                    js.position = [
                        float(self.data.qpos[self.model.joint(j).qposadr])
                        for j in self.joint_names
                    ]
                    self.joint_state_pub.publish(js)

                    # ---------------------------------------------------------------- #
                    # if self.ctrl_step % sync_step == 0:
                    #     self.hand_eye.allow_to_pub = True
                    #     self.top_down_cam.allow_to_pub = True

                    # print("Time in robot node: ", self.syn_cnt)
                    self.ctrl_step += 1

                self.time_sync(self.dt, start_time, False)
            
        except KeyboardInterrupt:
            self.get_logger().into("\nSimulation interrupted. Closing robot controller ...")
            # self.rc.tm.destroy_node()
            self.rc.destroy_node()


    def hand_eye_control(self):
        renderer = mujoco.Renderer(self.model, width=self.width, height=self.height)
        # renderer = self.hand_eye_renderer
        hand_eye_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)

        while rclpy.ok() and self.running:            
            with self.lock:
                start_time = time.perf_counter()  

                if self.hand_eye.allow_to_pub: 
                    # hand eye view
                    renderer.update_scene(self.data, camera=hand_eye_id)
                    self.hand_eye.getImage(renderer.render(), self.ctrl_step)
                    rclpy.spin_once(self.hand_eye, timeout_sec=0.001) # for hand eye

            # self.time_sync(1/self.fps, start_time, False)
            self.time_sync(self.dt, start_time, False)
        self.hand_eye.destroy_node()

    def td_cam_control(self):
        # 카메라 렌더러 설정
        renderer = mujoco.Renderer(self.model, width=640, height=480)
        # renderer = self.td_cam_renderer

        top_down_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_down_cam")

        while rclpy.ok() and self.running:            
            with self.lock:
                start_time = time.perf_counter()    
                
                if self.top_down_cam.allow_to_pub: 
                    # top down view
                    renderer.update_scene(self.data, camera=top_down_cam_id)
                    self.top_down_cam.getImage(renderer.render(), self.ctrl_step)     
                    rclpy.spin_once(self.top_down_cam, timeout_sec=0.001) # for hand eye

            # self.time_sync(1/self.fps, start_time, False)
            self.time_sync(self.dt, start_time, False)

        self.top_down_cam.destroy_node()


    def time_sync(self, target_dt, t_0, verbose=False):
        elapsed_time = time.perf_counter() - t_0
        sleep_time = target_dt - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        if verbose:
            print(f'Time {elapsed_time*1000:.4f} + {sleep_time*1000:.4f} = {(elapsed_time + sleep_time)*1000} ms')
    
    #pose 헬퍼 메서드 - site의 위치/자세 가져오기
    def get_site_pos(self, site_name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return np.array(self.data.site_xpos[site_id])
        
    #pose 콜벡 함수
    def get_site_position_callback(self, request, response):
        site_name = request.site_name
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        if site_id == -1:
            self.get_logger().error(f"[GetSitePosition] '{site_name}' site not found!")
            response.position.x = 0.0
            response.position.y = 0.0
            response.position.z = 0.0
        else:
            pos = self.data.site_xpos[site_id]
            response.position.x = float(pos[0])
            response.position.y = float(pos[1])
            response.position.z = float(pos[2])
            self.get_logger().info(f"[GetSitePosition] '{site_name}' → {pos}")

        return response
    
    #orientation 헬퍼 메서드 - site의 자세(orientation) 가져오기
    def get_site_orient(self, site_name):
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        # site_xmat 에는 3×3 행렬이 9개 요소(flat)로 들어있습니다
        mat = np.array(self.data.site_xmat[site_id]).reshape(3,3)
        # scipy 로 쿼터니언 변환
        from scipy.spatial.transform import Rotation as R
        quat = R.from_matrix(mat).as_quat()  # [x, y, z, w]
        return quat
    
    #orientation 콜백 함수
    def get_site_orientation_callback(self, request, response):
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, request.site_name)
        if site_id == -1:
            self.get_logger().error(f"[GetSiteOrientation] '{request.site_name}' not found")
            response.orientation = Quaternion(w=1.0)  # identity
        else:
            # data.site_xmat 는 3x3 행렬(flat array) 입니다
            mat = np.array(self.data.site_xmat[site_id]).reshape(3,3)
            quat = R.from_matrix(mat).as_quat()  # [x,y,z,w]
            response.orientation.x = float(quat[0])
            response.orientation.y = float(quat[1])
            response.orientation.z = float(quat[2])
            response.orientation.w = float(quat[3])
        return response