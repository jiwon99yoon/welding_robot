import mujoco
import mujoco.viewer as mj_view
import time

# xml_path = "two_box_full.xml"
xml_path = "robot.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

try:
    with mj_view.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)  
            viewer.sync()  
            time.sleep(0.001) 

except KeyboardInterrupt:
    print("\nSimulation interrupted. Closing viewer...")
