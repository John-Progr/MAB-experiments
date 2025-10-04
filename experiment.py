import numpy as np
import matplotlib.pyplot as plt

class Experiment:

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.rewards = [] 
        self.actions = []

    def run(self, n_trials):
        for _ in range(n_trials):
            arm = self.agent.select_arm()
            channel = self.env.channels[arm]
            reward = self.env.get_reward(channel)
            self.agent.update(arm, reward)
            self.actions.append(channel)
            self.rewards.append(reward)

    
    def plot(self):

        plt.plot(np.cumsum(self.rewards) / (np.arange(1, len(self.rewards)+1)))
        plt.xlabel("Iteration")
        plt.ylabel("Average Reward")
        plt.title("Epsilon-Greedy Wireless Channel Selection")
        plt.show()

