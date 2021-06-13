from . processing import *
from .environment import RaspEnv
import tensorflow as tf
import warnings
import abc
import os
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback, CallbackList, CheckpointCallback
from stable_baselines.common.vec_env import DummyVecEnv
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


class Agent:

    def __init__(self, model: abc.ABCMeta):
        # Define new environment
        self.env = RaspEnv()
        # Check if environment is ok
        check_env(self.env)
        # Define empty model
        self.model = model
        # Configure model hyperparameters
        self.config = {'learning_starts': 32, 'target_network_update_freq': 100, 'learning_rate': 0.001}

    def load_model(self, tensorboard_log: str) -> None:

        # Get path of latest model
        path = os.getcwd()
        os.chdir(os.getcwd() + '/model_checkpoints')

        # Process all the files in the folder
        files = [x for x in os.listdir() if x.endswith(".zip")]
        num = []
        for file in files:
            num.append([int(x) for x in file.split('_') if x.isdigit()][0])
        filename = "rl_model_" + str(max(num)) + "_steps.zip"

        # Load most recent model
        self.model = self.model.load(load_path=filename,
                                     env=DummyVecEnv([lambda: self.env]),
                                     tensorboard_log=tensorboard_log,
                                     **self.config)
        print("Successfully loaded the previous model: " + filename)

        # Return to root path
        os.chdir(path)

    def create_model(self, tensorboard_log: str) -> None:

        # Vector-encode our new environment
        env = DummyVecEnv([lambda: self.env])

        # Create new model
        self.model = self.model('MlpPolicy', env, verbose=1,
                                tensorboard_log=tensorboard_log,
                                **self.config)
        print(type(self.model))
        print("Successfully created new model")

    def train(self, tensorboard_log: str) -> None:

        try:
            self.load_model(tensorboard_log=tensorboard_log)

        except:
            self.create_model(tensorboard_log=tensorboard_log)

        # Stop training if reward gets close to zero
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-0.1, verbose=1)
        eval_callback = EvalCallback(self.env, callback_on_new_best=callback_on_best, verbose=1)

        # Save model at regular time intervals
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./model_checkpoints/')

        # Chain callbacks together
        callback = CallbackList([eval_callback, checkpoint_callback])

        # Train model
        self.model.learn(total_timesteps=int(1e10), callback=callback, tb_log_name="run")

        # Save trained model
        print("Training is finished!")

    def evaluate(self, tensorboard_log: str) -> None:

        self.create_model(tensorboard_log)
        obs = self.env.reset()

        while True:

            # Compute action based on previous observation
            action, _states = self.model.predict(obs, deterministic=True)
            # Compute observation and reward from the action that was sent
            obs, rewards, done, info = self.env.step(action)
            if done:
                obs = self.env.reset()
