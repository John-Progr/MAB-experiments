import numpy as np
import matplotlib.pyplot as plt
from logging_utils import save_to_csv
from datetime import datetime
import time

class Experiment:

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.rewards = []
        self.actions = []



        # we added  live plot handles 
        self.live_fig = None
        self.ax_avg = None
        self.ax_bar = None

        self.arm_fig = None
        self.arm_axes = {}



    def run(self, n_trials):
        for i in range(n_trials):
            print(f"Entering trial number: {i + 1}")
            arm = self.agent.select_arm()
            channel = self.env.channels[arm]
            print(f"DEBUG - selected arm index: {arm}, channel value: {channel}")  # Add this too
            time.sleep(2)
            reward = self.env.get_reward(channel)
            self.agent.update(arm, reward)

            self.actions.append(channel)
            self.rewards.append(reward)
            row = [{
                "Iteration": i,
                "Channel": channel,
                "Reward": reward,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }]
            save_to_csv(row) # we save each iteration as a row of a data bank so we can replay the experiment again if we desire


            self.update_live_main_plot()
            self.update_live_arm_plots()

            # halfway checkpoint
            """
            if i + 1 == n_trials // 2:
                print("\n📊 Halfway through — generating mid-run plot...\n")
                print("\n We are on the iteration: ", i)
                self.plot()
                self.plot_avg_reward_per_arm_over_time()  # show the first-half plot
            """

    def plot(self):
        # 1. Apply a modern style
        plt.style.use('ggplot')
        
        if len(self.rewards) == 0:
            raise ValueError("No data to plot. Run experiment.run(n_trials) first.")

        steps = np.arange(1, len(self.rewards) + 1)
        avg_reward = np.cumsum(self.rewards) / steps  # cumulative average reward

        # 2. Increase figure size for better clarity
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle("Epsilon-Greedy Wireless Channel Selection: Final Results", 
                     fontsize=18, fontweight='bold', color='#444444')

        # --- Subplot 1: Cumulative Average Reward ---
        axs[0].plot(steps, avg_reward, color='#E69F00', linewidth=3) # Use a distinctive color and thicker line
        axs[0].set_title("Average Reward vs Steps (Iterations)", loc='left', fontsize=14, fontweight='bold')
        axs[0].set_xlabel("Step (Iteration)", fontsize=12)
        axs[0].set_ylabel("Cumulative Average Reward", fontsize=12)
        axs[0].grid(True, linestyle='-', alpha=0.4)
        
        # Remove top and right spines
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['top'].set_visible(False)
        
        # Add final value annotation
        axs[0].text(steps[-1], avg_reward[-1], f'Final: {avg_reward[-1]:.2f}', 
                    color='black', fontsize=11, ha='right', va='bottom', fontweight='bold')


        # --- Subplot 2: Average Reward per Channel (Bar Chart) ---
        unique_channels = list(dict.fromkeys(self.actions))
        avg_rewards_per_channel = []
        for ch in unique_channels:
            ch_rewards = [r for (a, r) in zip(self.actions, self.rewards) if a == ch]
            avg_rewards_per_channel.append(np.mean(ch_rewards))
            
        # Sort channels by their average reward for better visual comparison
        sorted_data = sorted(zip(avg_rewards_per_channel, unique_channels), reverse=True)
        sorted_rewards = [d[0] for d in sorted_data]
        sorted_channels = [str(d[1]) for d in sorted_data]

        bars = axs[1].bar(sorted_channels, sorted_rewards, color='#56B4E9') # Use a complementary color
        axs[1].set_title("Average Reward per Channel (Arm)", loc='left', fontsize=14, fontweight='bold')
        axs[1].set_xlabel("Channel (Arm)", fontsize=12)
        axs[1].set_ylabel("Average Reward", fontsize=12)
        axs[1].grid(axis='y', linestyle='-', alpha=0.4)
        
        # Add reward labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            axs[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)

        # Remove top and right spines
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)


        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
        # Restore default style
        plt.style.use('default')


    # line plots for each arm
    def plot_avg_reward_per_arm_over_time(self):
        # 1. Apply a modern style
        plt.style.use('ggplot')
        
        if len(self.rewards) == 0:
            raise ValueError("No data to plot. Run experiment.run(n_trials) first.")

        arms = list(dict.fromkeys(self.actions))  # unique arm labels
        n_arms = len(arms)

        # 2. Adjust figure size and apply main title styling
        fig, axs = plt.subplots(n_arms, 1, figsize=(12, 5 * n_arms), sharex=True)
        fig.suptitle("Average Reward per Arm Over Time: Individual Performance", 
                     fontsize=18, fontweight='bold', color='#444444')

        # If there is only one arm, axs is not a list → convert it
        if n_arms == 1:
            axs = [axs]

        # 3. Get distinct colors for each arm
        colors = plt.cm.get_cmap('Dark2', n_arms) 

        for idx, arm in enumerate(arms):
            cumulative_sum = 0
            count = 0
            arm_avg_rewards = []
            steps = []

            # Build the arm-specific average reward over time
            for t, (a, r) in enumerate(zip(self.actions, self.rewards), start=1):
                if a == arm:
                    count += 1
                    cumulative_sum += r
                    arm_avg_rewards.append(cumulative_sum / count)
                    steps.append(t)

            ax = axs[idx]
            color = colors(idx)
            
            # 4. Use styled plot line and fill
            ax.plot(steps, arm_avg_rewards, 
                    color=color, 
                    linewidth=3, 
                    marker='o' if count < 10 else None, # Only show markers for small number of selections
                    markersize=6
            )
            ax.fill_between(steps, arm_avg_rewards, color=color, alpha=0.15)

            # 5. Enhanced Title, Labels, and Spines
            ax.set_title(f"Arm {arm} (Selections: {count})", loc='left', fontsize=14, fontweight='bold')
            ax.set_ylabel("Avg Reward", fontsize=12)
            ax.grid(True, linestyle='-', alpha=0.4)
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # Add final average text
            if count > 0:
                 ax.text(steps[-1], arm_avg_rewards[-1] + 0.05 * (max(arm_avg_rewards) - min(arm_avg_rewards) if len(arm_avg_rewards) > 1 else 1), 
                        f'{arm_avg_rewards[-1]:.2f}',
                        color='black', fontsize=10, ha='right', fontweight='bold')


        axs[-1].set_xlabel("Step (Iteration)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
        # Restore default style
        plt.style.use('default')
    
    def update_live_main_plot(self):
        # 1. Apply a modern style
        plt.style.use('ggplot')
        plt.ion()

        # Get all channels from environment (needed for consistent bar colors)
        all_arms = self.env.channels
        colors = plt.cm.get_cmap('Dark2', len(all_arms))
        
        if self.live_fig is None:
            # Increase figure size for better visual separation
            self.live_fig, (self.ax_avg, self.ax_bar) = plt.subplots(2, 1, figsize=(12, 9))
            self.live_fig.suptitle("Live Channel Performance", 
                                   fontsize=18, fontweight='bold', color='#444444')

        steps = np.arange(1, len(self.rewards)+ 1)
        avg_reward = np.cumsum(self.rewards) / steps

        # --- Update line plot (Cumulative Average) ---
        self.ax_avg.clear()
        
        # 2. Add bold markers and use fill
        self.ax_avg.plot(steps, avg_reward, 
                         color='#E69F00', 
                         linewidth=3, 
                         marker='o',          # Add marker
                         markersize=7,        # Bold marker size
                         markeredgecolor='black',
                         markerfacecolor='#E69F00',
                         label="Cumulative Avg Reward"
        )
        self.ax_avg.fill_between(steps, avg_reward, color='#E69F00', alpha=0.1)
        
        # 3. Add text annotation above each marker
        for s, avg in zip(steps, avg_reward):
            # Only annotate every 5th step or the last step to prevent clutter
            if s % 5 == 0 or s == steps[-1]:
                self.ax_avg.text(s, avg + (max(avg_reward) - min(avg_reward)) * 0.02, # Offset text slightly above
                                 f'{avg:.2f}',
                                 color='black', fontsize=9, ha='center', va='bottom', fontweight='bold')


        # Refined titles, labels, and grid
        self.ax_avg.set_title("Cumulative Average Reward Over Time", loc='left', fontsize=14, fontweight='bold')
        self.ax_avg.set_xlabel("Step (Iteration)", fontsize=12)
        self.ax_avg.set_ylabel("Avg Reward", fontsize=12)
        self.ax_avg.grid(True, linestyle='-', alpha=0.4)
        
        # Remove top and right spines
        self.ax_avg.spines['right'].set_visible(False)
        self.ax_avg.spines['top'].set_visible(False)


        # --- Update bar chart (Channel Comparison) ---
        unique = list(dict.fromkeys(self.actions))
        bar_values = [
            np.mean([r for (a, r) in zip(self.actions, self.rewards) if a == ch])
            for ch in unique
        ]

        # Sort bars by value for easy comparison
        sorted_data = sorted(zip(bar_values, unique), reverse=True)
        sorted_rewards = [d[0] for d in sorted_data]
        sorted_channels = [str(d[1]) for d in sorted_data]
        
        self.ax_bar.clear()
        
        # 4. Determine colors for the bars based on their original channel value position in all_arms
        bar_colors = [colors(all_arms.index(float(ch))) for ch in sorted_channels]
        
        # Plot bars using the consistent colors
        bars = self.ax_bar.bar(sorted_channels, sorted_rewards, color=bar_colors)
        
        # Add reward labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            self.ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                             f'{height:.2f}',
                             ha='center', va='bottom', fontsize=10)


        self.ax_bar.set_title("Average Reward per Channel (Current)", loc='left', fontsize=14, fontweight='bold')
        self.ax_bar.set_xlabel("Channel", fontsize=12)
        self.ax_bar.set_ylabel("Avg Reward", fontsize=12)
        self.ax_bar.grid(axis='y', linestyle='-', alpha=0.4)

        # Remove top and right spines
        self.ax_bar.spines['right'].set_visible(False)
        self.ax_bar.spines['top'].set_visible(False)

        
        self.live_fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.live_fig.canvas.draw()
        self.live_fig.canvas.flush_events()
        
        # Restore default style
        plt.style.use('default')
    def update_live_arm_plots(self):
        # 1. Apply a modern style
        plt.style.use('ggplot') # Use a clean, stylish theme for better aesthetics
        plt.ion()

        # Get all channels from environment
        all_arms = self.env.channels
        
        # Create the figure once with all arms
        if self.arm_fig is None:
            n_arms = len(all_arms)
            # Make the figure larger/taller for better visual separation
            self.arm_fig, axs = plt.subplots(n_arms, 1, figsize=(12, 5 * n_arms), sharex=True) 
            if n_arms == 1:
                axs = [axs]
            
            self.arm_axes = {}
            for idx, ax in enumerate(axs):
                self.arm_axes[all_arms[idx]] = ax
            
            self.arm_fig.suptitle("Live Average Reward per Arm", 
                                  fontsize=18, fontweight='bold', color='#444444')

        # Get a list of distinct colors for the lines
        colors = plt.cm.get_cmap('Dark2', len(all_arms)) 

        # Update each arm's plot
        for idx, (arm, ax) in enumerate(self.arm_axes.items()):
            cumulative = 0
            count = 0
            avg_vals = []
            steps = []

            # Data calculation remains the same
            for t, (a, r) in enumerate(zip(self.actions, self.rewards), start=1):
                if a == arm:
                    count += 1
                    cumulative += r
                    avg_vals.append(cumulative / count)
                    steps.append(t)

            # --- DEBUG line removed for final elegant code ---
            # print(f"Plotting for Arm (Channel Value) {arm}: Steps: {steps}, Avg Values: {avg_vals}")
            # --------------------------------------------------

            ax.clear()
            
            if steps:
                # 2. Use a unique color, thicker line, and semi-transparent fill
                color = colors(idx)
                ax.plot(steps, avg_vals, 
                        label=f"Avg Reward (N={count})",
                        color=color, 
                        linewidth=3, # Thicker line
                        marker='o', 
                        markersize=6, 
                        markeredgecolor='black', # Differentiate marker edge
                        markeredgewidth=0.5
                ) 
                
                # Optional: Add a subtle fill below the line
                ax.fill_between(steps, avg_vals, color=color, alpha=0.1) 
                
                # 3. Add a final value marker for quick reading
                ax.axhline(avg_vals[-1], color='gray', linestyle='--', linewidth=1, alpha=0.6)
                ax.text(steps[-1] + 0.5, avg_vals[-1], f'{avg_vals[-1]:.2f}', 
                        color='black', fontsize=10, verticalalignment='center', fontweight='bold')
                
            else:
                ax.text(0.5, 0.5, 'Not yet selected', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='darkred', style='italic') # Use a more noticeable color
                
            # 4. Refined titles and labels
            ax.set_title(f"Channel {arm}", loc='left', fontsize=14, fontweight='bold')
            ax.set_ylabel("Avg Reward", fontsize=12)
            ax.grid(True, linestyle='-', alpha=0.4) # Slightly darker grid
            
            # Remove top and right spines (box lines) for a cleaner look
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)


        # 5. Final Layout adjustments
        list(self.arm_axes.values())[-1].set_xlabel("Step (Iteration)", fontsize=14)
        self.arm_fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        self.arm_fig.canvas.draw()
        self.arm_fig.canvas.flush_events()
        # Restore default style after drawing to avoid affecting other plots
        plt.style.use('default')

  