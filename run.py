from agents import EpsilonGreedy
from environments import WirelessChannelEnv
from experiment import Experiment


source_ip ="192.168.2.10"
dest_ip = "192.168.2.20"
channels = [1, 2, 3, 36, 40, 44, 48, 149, 165] # wihout 3 and 13
agent = EpsilonGreedy(n_arms=len(channels), epsilon=0.25,update_rule= "exponential_smoothing", alpha = 0.7)
env = WirelessChannelEnv(source_ip,dest_ip,channels,"http://localhost:8000/network/data-transfer-rate")

exp = Experiment(agent, env)
exp.run(200)
exp.plot()



