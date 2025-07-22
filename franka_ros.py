import gym
import time
import threading
import numpy as np
from packaging import version

import frankx
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState


class ReachingFrankaRos(gym.Env):
    def __init__(self, robot_ip="172.16.0.2", device="cuda:0", control_space="joint", motion_type="waypoint", camera_tracking=False):
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

        # camera tracking (disabled by default)
        self.camera_tracking = camera_tracking
        if self.camera_tracking:
            threading.Thread(target=self._update_target_from_camera).start()

        # spaces
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(18,), dtype=np.float32)
        if self.control_space == "joint":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        elif self.control_space == "cartesian":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        else:
            raise ValueError("Invalid control space:", self.control_space)

        self.motion = None
        self.motion_thread = None

        self.dt = 1 / 120.0
        self.action_scale = 5.0
        self.dof_vel_scale = 0.1
        self.max_episode_length = 100
        self.robot_dof_speed_scales = 1
        self.target_pos = np.array([0.65, 0.2, 0.2])
        self.robot_default_dof_pos = np.radians([0, -45, 0, -135, 0, 90, 45])
        self.robot_dof_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.robot_dof_upper_limits = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

        self.progress_buf = 1
        self.obs_buf = np.zeros((18,), dtype=np.float32)

        rospy.init_node("franka_env", anonymous=True)
        self.vel_publisher = rospy.Publisher("desired_velocity", Float32MultiArray, queue_size=10)
        rospy.Subscriber("base_feedback/joint_state", JointState, self.js_callback)
        rospy.Subscriber("/eef_position/", Float32MultiArray, self.eef_callback)
        self.joint_positions = self.robot_default_dof_pos
        self.joint_velocities = np.zeros_like(self.joint_positions)
        self.prev_err = np.zeros(7)
        self.eef_pos = np.zeros(3)
    
    def eef_callback(self, msg:Float32MultiArray):
        self.eef_pos = np.array(msg.data)
    
    def js_callback(self, msg:JointState):
        self.joint_states = np.array(msg.position[:7])
        self.joint_velocities = np.array(msg.velocity[:7])

    def _update_target_from_camera(self):
        pixel_to_meter = 1.11 / 375  # m/px: adjust for custom cases

        import cv2
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # convert to HSV and remove noise
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.medianBlur(hsv, 15)

            # color matching in HSV
            mask = cv2.inRange(hsv, np.array([80, 100, 100]), np.array([100, 255, 255]))
            M = cv2.moments(mask)
            if M["m00"]:
                x = M["m10"] / M["m00"]
                y = M["m01"] / M["m00"]

                # real-world position (fixed z to 0.2 meters)
                pos = np.array([pixel_to_meter * (y - 185), pixel_to_meter * (x - 320), 0.2])
                if self is not None:
                    self.target_pos = pos

                # draw target
                frame = cv2.circle(frame, (int(x), int(y)), 30, (0,0,255), 2)
                frame = cv2.putText(frame, str(np.round(pos, 4).tolist()), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            # show images
            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                cap.release()

    def _get_observation_reward_done(self):
        robot_dof_pos = self.joint_positions
        robot_dof_vel = self.joint_velocities
        end_effector_pos = self.eef_pos

        dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        dof_vel_scaled = robot_dof_vel * self.dof_vel_scale

        self.obs_buf[0] = self.progress_buf / float(self.max_episode_length)
        self.obs_buf[1:8] = dof_pos_scaled
        self.obs_buf[8:15] = dof_vel_scaled
        self.obs_buf[15:18] = self.target_pos

        # reward
        distance = np.linalg.norm(end_effector_pos - self.target_pos)
        reward = -distance

        # done
        done = self.progress_buf >= self.max_episode_length - 1
        done = done or distance <= 0.075

        print("Distance:", distance)
        if done:
            print("Target or Maximum episode length reached")
            time.sleep(1)

        return self.obs_buf, reward, done

    def reset(self):
        print("Resetting...")

        # end current motion
        if self.motion is not None:
            self.motion.finish()
            self.motion_thread.join()

        self.motion = None
        self.motion_thread = None

        # open/close gripper
        # self.gripper.open()
        # self.gripper.clamp()
        # get target position from prompt
        if not self.camera_tracking:
            while True:
                try:
                    print("Enter target position (X, Y, Z) in meters")
                    raw = input("or press [Enter] key for a random target position: ")
                    if raw:
                        self.target_pos = np.array([float(p) for p in raw.replace(' ', '').split(',')])
                    else:
                        noise = (2 * np.random.rand(3) - 1) * np.array([0.25, 0.25, 0.10])
                        self.target_pos = np.array([0.5, 0.0, 0.2]) + noise
                    print("Target position:", self.target_pos)
                    break
                except ValueError:
                    print("Invalid input. Try something like: 0.65, 0.0, 0.2")
        input("Press [Enter] to continue")

        self.progress_buf = 0
        observation, reward, done = self._get_observation_reward_done()

        if self._drepecated_api:
            return observation
        else:
            return observation, {}

    def step(self, action):
        action /= 100
        self.progress_buf += 1

        kp=0.1
        kd = 0.005

        action *= self.action_scale
        cur_velo = self.joint_velocities
        err = action - cur_velo
        d_error = err - self.prev_err
        
        self.prev_err = err

        action = cur_velo + kp * err + kd * d_error
    
        msg = Float32MultiArray()
        msg.data = action[:7].tolist()
        self.vel_publisher.publish(msg)

        # the use of time.sleep is for simplicity. This does not guarantee control at a specific frequency
        time.sleep(0.1)  # lower frequency, at 30Hz there are discontinuities

        observation, reward, done = self._get_observation_reward_done()

        if self._drepecated_api:
            return observation, reward, done, {}
        else:

            return observation, reward, done, done, {}

    def render(self, *args, **kwargs):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    robot = ReachingFrankaRos()
    robot.reset()
    action = np.array([0.1, 0, 0, 0, 0, 0, 0 ,0])
    for _ in range(50):
        robot.step(action)
