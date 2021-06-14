from src import Agent
from stable_baselines import PPO2
from stable_baselines import DQN

if __name__ == "__main__":
    agent = Agent(model=PPO2)
    # If one wishes to train the agent
    # agent.train(tensorboard_log='./trial_3/')
    # If one wishes to simply run the agent
    agent.evaluate(tensorboard_log='./evaluation/', model_path='/checkpoint_2')
