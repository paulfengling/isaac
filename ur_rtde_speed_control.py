import gym
import time
import threading
import numpy as np
from packaging import version

import frankx
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

import rtde_control
import rtde_receive

def normalize_angle( q):
        q = q % (2 * np.pi) 
        q = np.where(q > np.pi, q - 2 * np.pi, q)
        q = np.where(q < -np.pi, q + 2 * np.pi, q)
        return q     
class ReferenceSelector():
    def __init__(self, dof, len_buffer=16):
        self.euclidean_threshold = 0.15
        self.dof = dof
        self.trajectory_buffer = []
        self.cur_pos = np.zeros(dof)
        self.max_buffer_len = 16
    
    def normalize_angle(self, q):
        q = q % (2 * np.pi) 
        q = np.where(q > np.pi, q - 2 * np.pi, q)
        q = np.where(q < -np.pi, q + 2 * np.pi, q)
        return q       

    def _refresh_trajectory(self, action=None):        
        self.trajectory_buffer = []                                                                     
    
    def step(self, action, cur_pos):
        if len(self.trajectory_buffer) < self.max_buffer_len:
            self.trajectory_buffer.append(action)
            reset_signal = False
        else:
            reset_signal = True
        tar = self.normalize_angle(self.trajectory_buffer[0])     
        cur = self.normalize_angle(cur_pos)   
        err = tar - cur
        err = self.normalize_angle(err)
        euclidean_distance = np.linalg.norm(err)
        # print(err)
        while euclidean_distance < self.euclidean_threshold:
            # print("go on")
            if len(self.trajectory_buffer) > 0:   
                self.trajectory_buffer.pop(0)
            else: 
                break
            if len(self.trajectory_buffer) > 0:          
                tar = self.normalize_angle(self.trajectory_buffer[0])     
                cur = self.normalize_angle(cur_pos)   
                err = tar - cur
                err = self.normalize_angle(err)
                euclidean_distance = np.linalg.norm(err)
            else:
                break
            
        
        if len(self.trajectory_buffer) > 0:
            qr = self.trajectory_buffer[0]
        else:
            qr = action

        
        return qr,  reset_signal




class ReachingURRos():
    def __init__(self, len_buffer=16,  robot_ip="172.16.0.2", device="cuda:0", control_space="joint", motion_type="waypoint", camera_tracking=False):
        # gym API
        self._drepecated_api = version.parse(gym.__version__) < version.parse(" 0.25.0")

        self.device = device
        self.control_space = control_space  # joint or cartesian
        self.motion_type = motion_type  # waypoint or impedance

        if self.control_space == "cartesian" and self.motion_type == "impedance":
            # The operation of this mode (Cartesian-impedance) was adjusted later without being able to test it on the real robot.
            # Dangerous movements may occur for the operator and the robot.
            # Comment the following line of code if you want to proceed with this mode.
            raise ValueError("See comment in the code to proceed with this mode")
            pass

        self.motion = None
        self.motion_thread = None

        self.dt = 1 / 120.0
        self.action_scale = 5.0
        self.dof_vel_scale = 0.1
        self.max_episode_length = 100
        self.robot_dof_speed_scales = 1
        self.target_pos = np.array([0.65, 0.2, 0.2])
        self.robot_default_dof_pos = np.array([0, -1.6, 2.4, 0.77, 1.57, 2.34])
        # self.robot_default_dof_pos = np.radians([0, 0, 0, 0, 0, 0])

        self.progress_buf = 1
        self.obs_buf = np.zeros((18,), dtype=np.float32)

        """
        FCI Interface    |libfranka node cpp |  -- joint tate & eef state --> | this python | <-- isaacgym step -- FrankaCubeStack
                         |velocity execution |  <---- velocity command ------ |  pd control |
        """

        self.reference_selector = ReferenceSelector(6, len_buffer)
        self.joint_positions = self.robot_default_dof_pos
        self.joint_velocities = np.zeros_like(self.joint_positions)
        self.eef_pos = np.zeros(3)
        self.prev_err = np.zeros(6)

        ROBOT_IP = "172.17.0.2"
        self.rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

        # self.go_home()
    
    
    def go_home(self):
        target = self.robot_default_dof_pos
        target = normalize_angle(target)

        cur_pos = np.array(self.rtde_r.getActualQ())
        cur_pos = normalize_angle(cur_pos)
        print(cur_pos)
        dis = target - cur_pos
        norm_dis = np.linalg.norm(dis)

        print(norm_dis)

        while(norm_dis > 0.1):
            cur_pos = self.rtde_r.getActualQ()
            cur_pos = np.array(cur_pos)

            

            cur_pos = normalize_angle(cur_pos)
            # print(cur_pos)
            print(cur_pos)
            print(target)

            dis = target - cur_pos
            norm_dis = np.linalg.norm(dis)
            print(norm_dis)

            err = target - cur_pos
            d_err = err - self.prev_err

            self.prev_err = err

            kp = 0.2
            kd = 0.001
            vel_cmd = kp * err + kd * d_err

            vel_cmd = vel_cmd.tolist()

            # print(vel_cmd)
 
            self.rtde_c.speedJ(vel_cmd, acceleration=0.5, time=2.0)
            time.sleep(0.1)
       
        
        self.prev_err = np.zeros(6)


    def reset(self):
        self.reference_selector._refresh_trajectory()

    def step(self, action):
        cur_positions = self.rtde_r.getActualQ()
        cur_positions = np.array(cur_positions)
        # print(type(cur_positions))
        action, reset_signal = self.reference_selector.step(action, cur_positions)

        action = normalize_angle(action)
        cur_positions = normalize_angle(cur_positions)

        err = action - cur_positions
        d_err = err - self.prev_err

        self.prev_err = err

        kp = 0.85
        kd = 0.01
        vel_cmd = kp * err + kd * d_err

        vel_cmd = vel_cmd.tolist()
        # print(vel_cmd)
        self.rtde_c.speedJ(vel_cmd, acceleration=0.5, time=2.0)

        # print(len(self.reference_selector.trajectory_buffer) / self.reference_selector.max_buffer_len)
        eef_state = np.array(self.rtde_r.getActualTCPPose())
        
        return cur_positions, eef_state,  reset_signal

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    robot = ReachingURRos()
    robot.reset()
    robot.go_home()
