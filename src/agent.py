from . processing import *
from .environment import RaspEnv
from . callback import RaspCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import time


class Agent:

    def __init__(self):
        self.env = RaspEnv()
        check_env(self.env)
        self.running = False

    def run(self):
        # open a window to show debugging images
        cv2.namedWindow("Robotics Project")
        # Initialize the default camera
        vc = cv2.VideoCapture(0)

        try:
            # try to get the first frame
            if vc.isOpened():
                # frame has shape (720, 1280, 3)
                (readSuccessful, frame) = vc.read()
                self.env.frame = frame
            else:
                raise (Exception("failed to open camera."))

            # Define callback
            callback = RaspCallback(self.env)
            # Define new RL model
            model = PPO("MlpPolicy", self.env, verbose=1)
            model.learn(total_timesteps=10000, callback=callback)

            while readSuccessful:
                # When a red dot is detected on our pc window
                if self.env.square[0] is not None and not self.running:
                    print("Started testing the agent !")
                    obs = self.env.reset()
                    self.running = True

                # Stop training the algorithm if the red dot disappears
                if self.env.square[0] is None and self.running:
                    print("Resetting window after loosing the red dot")
                    self.running = False
                    obs = self.env.reset()

                if self.running:
                    # Compute action based on previous observation
                    action, _states = model.predict(obs)
                    # Send action to server to update location of red dot
                    self.env.network.send(action)
                    # Give the Rasp 0.1s to move red dot
                    time.sleep(0.1)
                    # Measure the resulting effect of the selected action
                    get_red_dot(self.env.square, frame, False)
                    if self.env.square[0] is not None and self.env.square[1] is not None:
                        # Compute observation and reward from the action that was sent
                        obs, rewards, done, info = self.env.step(action)
                        # Does not do anything
                        self.env.render()
                    else:
                        print("Red dot lost during training !")

                else:
                    # Do Network Stuff
                    self.env.network.send(4)
                    self.env.network.recv()
                    get_red_dot(self.env.square, frame, False)

                key = cv2.waitKey(10)  # Set refreshing time
                if key == 27:  # exit on ESC
                    break

                # Get Image from camera
                readSuccessful, frame = vc.read()
        finally:
            vc.release()  # close the camera
            cv2.destroyWindow("Robotics Project")  # close the window
