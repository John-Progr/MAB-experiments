import numpy as np
import matplotlib.pyplot as plt

class Experiment:

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.rewards = []
        self.actions = []

    def run(self, n_trials):
        for i in range(n_trials):
            print(f"Entering trial number: {i + 1}")
            arm = self.agent.select_arm()
            channel = self.env.channels[arm]
            reward = self.env.get_reward(channel)
            self.agent.update(arm, reward)
            self.actions.append(channel)
            self.rewards.append(reward)

            # halfway checkpoint
            if i + 1 == n_trials // 2:
                print("\n📊 Halfway through — generating mid-run plot...\n")
                self.plot_half()  # show the first-half plot

    def plot(self):
        if len(self.rewards) == 0:
            raise ValueError("No data to plot. Run experiment.run(n_trials) first.")

        steps = np.arange(1, len(self.rewards) + 1)
        avg_reward = np.cumsum(self.rewards) / steps  # cumulative average reward

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle("Epsilon-Greedy Wireless Channel Selection", fontsize=14)

        axs[0].plot(steps, avg_reward, color='tab:orange', linewidth=2)
        axs[0].set_title("Average Reward vs Steps (Iterations)")
        axs[0].set_xlabel("Step (Iteration)")
        axs[0].set_ylabel("Average Reward")
        axs[0].grid(True, linestyle='--', alpha=0.5)

        unique_channels = list(dict.fromkeys(self.actions))
        avg_rewards_per_channel = []
        for ch in unique_channels:
            ch_rewards = [r for (a, r) in zip(self.actions, self.rewards) if a == ch]
            avg_rewards_per_channel.append(np.mean(ch_rewards))

        axs[1].bar([str(ch) for ch in unique_channels], avg_rewards_per_channel, color='tab:blue')
        axs[1].set_title("Average Reward per Channel (Arm)")
        axs[1].set_xlabel("Channel (Arm)")
        axs[1].set_ylabel("Average Reward")
        axs[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_half(self):
        if len(self.rewards) == 0:
            raise ValueError("No data to plot. Run experiment.run(n_trials) first.")

        half = len(self.rewards) // 2
        half_rewards = self.rewards[:half]
        half_actions = self.actions[:half]

        steps = np.arange(1, len(half_rewards) + 1)
        avg_reward = np.cumsum(half_rewards) / steps  # cumulative average reward

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle("Epsilon-Greedy Wireless Channel Selection (First Half Only)", fontsize=14)

        axs[0].plot(steps, avg_reward, color='tab:green', linewidth=2)
        axs[0].set_title("Average Reward vs Steps (First Half)")
        axs[0].set_xlabel("Step (Iteration)")
        axs[0].set_ylabel("Average Reward")
        axs[0].grid(True, linestyle='--', alpha=0.5)

        unique_channels = list(dict.fromkeys(half_actions))
        avg_rewards_per_channel = []
        for ch in unique_channels:
            ch_rewards = [r for (a, r) in zip(half_actions, half_rewards) if a == ch]
            avg_rewards_per_channel.append(np.mean(ch_rewards))

        axs[1].bar([str(ch) for ch in unique_channels], avg_rewards_per_channel, color='tab:purple')
        axs[1].set_title("Average Reward per Channel (First Half)")
        axs[1].set_xlabel("Channel (Arm)")
        axs[1].set_ylabel("Average Reward")
        axs[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
