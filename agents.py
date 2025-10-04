import numpy as np


class EpsilonGreedy:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)   # Number of times each arm is pulled, basically how often each channel is used 
        self.values = np.zeros(n_arms)   # Estimated values of each arm, basically estimated throughput per channel

    def select_arm(self):
        """Choose an arm (explore or exploit)."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)   # Explore
        else:
            return np.argmax(self.values)           # Exploit best so far

    def update(self, chosen_arm, reward):
        """Update estimated value of the chosen arm using incremental mean."""
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = value + (reward - value) / n


