from agents import EpsilonGreedy
from environments import WirelessChannelEnv
from experiment import Experiment



channels = [1, 6, 11, 36, 40, 44, 149, 153, 157]
agent = EpsilonGreedy(n_arms=len(channels), epsilon=0.1)
env = WirelessChannelEnv(channels)


exp = Experiment(agent, env)
exp.run(1000)
exp.plot()