from . processing import *
from .environment import RaspEnv
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback, CallbackList, CheckpointCallback
import time
import os
from stable_baselines.common.vec_env import DummyVecEnv


class Agent:

    def __init__(self):
        # Define new environment
        self.env = RaspEnv()
        # Check if environment is ok
        check_env(self.env)
        # Define empty model
        self.model = None
        self.running = False

    def train(self):

        # Load latest model if available
        try:
            path = os.getcwd()
            os.chdir(os.getcwd() + '/model_checkpoints')
            files = [x for x in os.listdir() if x.endswith(".zip")]
            num = []
            for file in files:
                num.append([int(x) for x in file.split('_') if x.isdigit()][0])
            filename = "rl_model_" + str(max(num)) + "_steps.zip"
            print("Tentative: " + filename)
            self.model = PPO2.load(load_path=filename, env=DummyVecEnv([lambda: self.env]), tensorboard_log='./a2c_rasp_tensorboard/')
            print("Successfully loaded the previous model: " + filename)
            os.chdir(path)
        except:
            # Vector-encode our new environment
            env = DummyVecEnv([lambda: self.env])
            # Create new model
            self.model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log='./a2c_rasp_tensorboard/')
            print("Successfully created new model")

        # Stop training if reward get close to zero
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-1.0, verbose=1)
        eval_callback = EvalCallback(self.env, callback_on_new_best=callback_on_best, verbose=1)

        # Save model at regular time intervals
        checkpoint_callback = CheckpointCallback(save_freq=3000, save_path='./model_checkpoints/')

        # Chain callbacks together
        callback = CallbackList([eval_callback, checkpoint_callback])

        # Train model
        self.model.learn(total_timesteps=int(1e10), callback=callback, tb_log_name="run", reset_num_timesteps=False)

    def evaluate(self):

        # open a window
        cv2.namedWindow("Robotics Project")
        # Initialize the default camera
        vc = cv2.VideoCapture(0)

        try:
            # try to get the first frame
            if vc.isOpened():
                # frame has shape (720, 1280, 3)
                (readSuccessful, frame) = vc.read()
                # Pass the frame to the environment
                self.env.frame = frame
            else:
                raise (Exception("failed to open camera."))

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
                    action, _states = self.model.predict(obs)
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

