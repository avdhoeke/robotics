from src import Agent
from stable_baselines import PPO2
from stable_baselines import DQN

if __name__ == "__main__":
    agent = Agent(model=DQN)
    #agent.train(tensorboard_log='./trial_3/')
    agent.train(tensorboard_log='./evaluation/')