from agents import EpsilonGreedy
from environments import WirelessChannelEnv
from experiment import Experiment


source_ip ="192.168.2.30"
dest_ip = "192.168.2.40"
channels = [1,2,3,165] 
agent = EpsilonGreedy(n_arms=len(channels), epsilon=0.25,update_rule= "exponential_smoothing", alpha = 0.7)
env = WirelessChannelEnv(source_ip,dest_ip,channels,)

exp = Experiment(agent, env)
exp.run(150)
exp.plot()
exp.plot_avg_reward_per_arm_over_time()



