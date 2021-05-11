from . processing import *
from .environment import RaspEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2


class Agent:

    def __init__(self):
        self.env = RaspEnv()
        self.running = False

    def run(self):

        WINDOW_NAME = "Robotics Project"  # Define new window name
        cv2.namedWindow(WINDOW_NAME)  # open a window to show debugging images
        vc = cv2.VideoCapture(0)  # Initialize the default camera

        # Define new RL model
        model = PPO2(MlpPolicy, self.env, verbose=1)
        model.learn(total_timesteps=10000)

        try:
            if vc.isOpened():  # try to get the first frame
                (readSuccessful, frame) = vc.read() # frame has shape (720, 1280, 3)
            else:
                raise (Exception("failed to open camera."))

            while readSuccessful:

                # Update location of red dot and display image
                get_red_dot(self.env.square, frame, False)

                # When a red dot is detected on our pc window
                if self.env.square.x is not None and not self.running:
                    obs = self.env.reset()
                    self.running = True

                # Stop training the algorithm if the red dot disappears
                if self.env.square.x is None and self.running:
                    self.running = False

                if self.running:
                    # Compute action based on previous observation
                    action, _states = model.predict(obs)
                    print(action)
                    # Send action to server to update location of red dot
                    self.env.network.send(action)
                    # Compute observation and reward from the action that was sent
                    obs, rewards, done, info = self.env.step(action)
                    # Does not do anything
                    self.env.render()

                else:
                    # Do Network Stuff
                    self.env.network.send(4)

                key = cv2.waitKey(10)  # Set refreshing time
                if key == 27:  # exit on ESC
                    break

                # Get Image from camera
                readSuccessful, frame = vc.read()
        finally:
            vc.release()  # close the camera
            cv2.destroyWindow(WINDOW_NAME)  # close the window
