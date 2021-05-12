from abc import ABC
import gym
from src import Network
from . processing import *


class RaspEnv(gym.Env, ABC):
    def __init__(self):
        # The Discrete space allows a fixed range of non-negative numbers
        self.action_space = gym.spaces.Discrete(4)
        # Shape of observation space
        self.observation_shape = (8,)
        # The Box space represents an n-dimensional box
        self.observation_space = gym.spaces.Box(shape=self.observation_shape,
                                                low=np.array([-30.0, -30.0, -30.0, -30.0, -30.0, -30.0, 0.0, 0.0]),
                                                high=np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 710.0, 1280.0]),
                                                dtype=np.float64)
        # Coordinates of red dot on pc screen
        self.square = [0.0, 0.0]
        # Link with our Raspberry to receive observations
        self.network = Network()
        # Record video from webcam number 0
        self.cap = cv2.VideoCapture(0)

    def step(self, action: int):
        '''
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
        '''

        # Read video recording
        ret, frame = self.cap.read()

        # Send action to Raspberry Pi
        self.network.send(action)

        # Get ax, ay, az, gx, gy, gz from Raspberry Pi
        obs = self.network.recv()

        # Update location of red dot on PC screen
        get_red_dot(self.square, frame, False)
        self.square = np.random.rand(2)

        # Add red dot position to observation
        obs += [e for e in self.square]
        obs = np.asarray(obs)

        # Compute reward from red dot position
        reward = np.random.rand(1)[0]#self.compute_reward()

        # To allow cap to be used in a loop
        cv2.waitKey(10)

        return obs, reward, False, {}

    def compute_reward(self):
        distance = np.sqrt((self.square[0] - 355) ** 2 + (self.square[1] - 640) ** 2)
        return distance

    def reset(self) -> np.ndarray:
        '''
        The process gets started by calling reset(), which returns an initial observation
        :return: list
        '''
        return np.ones(self.observation_shape) * 1

    def render(self, mode='human'):
        pass
