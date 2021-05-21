from abc import ABC
import gym
from src import Network
from . processing import *
import time


class RaspEnv(gym.Env, ABC):
    def __init__(self):
        # The Discrete space allows a fixed range of non-negative numbers
        self.action_space = gym.spaces.Discrete(4)
        # Shape of observation space
        self.observation_shape = (8,)
        # The Box space represents an n-dimensional box
        self.observation_space = gym.spaces.Box(shape=self.observation_shape,
                                                low=np.array([-30.0, -30.0, -30.0, -30.0, -30.0, -30.0, 0.0, 0.0]),
                                                high=np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 1280.0, 720.0]),
                                                dtype=np.float64)
        # Coordinates of red dot on pc screen
        self.square_ = [0.0, 0.0]
        # Link with our Raspberry to receive observations
        self.network = Network()
        # Record video from webcam number 0
        self.cap = cv2.VideoCapture(0)
        # Count the number of time_steps
        self.counter = 0
        # Monitor the existence of a red dot
        self.trainable = True

    @property
    def frame(self):
        # try to get the first frame
        if self.cap.isOpened():
            # frame has shape (720, 1280, 3)
            readSuccessful, frame = self.cap.read()
        else:
            raise (Exception("failed to open camera."))
        return frame

    @property
    def square(self):
        get_red_dot(self, False)
        return self.square_

    @square.setter
    def square(self, s: list):
        self.square_ = s
        if s[0] is None or s[1] is None:
            self.trainable = False
        else:
            self.trainable = True

    def step(self, action: int):
        """
        The environment’s step function returns exactly what we need. In fact, step returns four values. These are:
        - observation (object): an environment-specific object representing your observation of the environment.
                                For example, pixel data from a camera, joint angles and joint velocities of a robot,
                                or the board state in a board game.
        - reward (float): amount of reward achieved by the previous action. The scale varies between environments,
                          but the goal is always to increase your total reward.
        - done (boolean): whether it’s time to reset the environment again. Most (but not all) tasks are divided up
                          into well-defined episodes, and done being True indicates the episode has terminated.
                          (For example, perhaps the pole tipped too far, or you lost your last life.)
        - info (dict): diagnostic information useful for debugging. It can sometimes be useful for learning
                       (for example, it might contain the raw probabilities behind the environment’s last state change).
                       However, official evaluations of your agent are not allowed to use this for learning.
        :param action: int
        :return: Tuple[np.ndarray, float, bool, dict]
        """

        # The default resulting outcome is: keep training !
        done = False

        # Send action to Raspberry Pi
        self.network.send(action)

        # Let a small time laps to PC before fetching the env's response
        time.sleep(0.2)

        # Get ax, ay, az, gx, gy, gz from Raspberry Pi
        obs = self.network.recv()

        # Update location of red dot on PC screen
        get_red_dot(self.square, self.frame, False)

        # Stop training if no red dot is detected
        if self.square[0] is None or self.square[1] is None:
            obs += [e for e in [0.0, 0.0]]
            obs = np.asarray(obs)
            reward = 0.0
            done = True

        elif self.counter >= 10000:
            obs += [e for e in [0.0, 0.0]]
            obs = np.asarray(obs)
            reward = 0.0
            done = True

        else:
            # Add red dot position to observation
            for e in self.square:
                obs.append(e)
            obs = np.asarray(obs)

            # Compute reward from red dot position
            reward = -self.compute_reward()

        # To allow cap to be used in a loop
        cv2.waitKey(10)

        return obs, reward, done, {}

    def compute_reward(self):
        distance = np.sqrt(((self.square[0] - 640) / 640) ** 2 + ((self.square[1] - 360) / 360) ** 2)
        print("Reward value:", distance)
        return distance

    def reset(self) -> np.ndarray:
        '''
        The process gets started by calling reset(), which returns an initial observation
        :return: list
        '''

        # Reset counter and red dot position
        self.counter = 0
        pos = np.random.randint(0, 8, 2)

        # Put red dot at random initial locations
        self.network.send([pos])

        # Get Raspberry acceleration and gyroscopic data
        obs = self.network.recv()

        for e in pos:
            obs.append(e)
        obs = np.asarray(obs)

        print("Environment has been reset !")

        return obs

    def render(self, mode='human'):
        pass
