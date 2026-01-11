from agents import EpsilonGreedy
from environments import WirelessChannelEnv, WirelessRouteEnv
from experiment import Experiment
from dotenv import load_dotenv


source_ip ="192.168.2.10"
dest_ip = "192.168.2.20"
channels = [1,2,12,36,149,165] 
devices = [10,20,30,40,50]
agent = EpsilonGreedy(n_arms=len(channels), epsilon=0.25,update_rule= "exponential_smoothing", alpha = 0.7)
envChannel = WirelessChannelEnv(source_ip,dest_ip,channels, "http://localhost:8000/network/data-transfer-rate", 2)
envRoute = WirelessRouteEnv(source_ip,dest_ip,devices, "http://localhost:8000/network/data-transfer-rate", 2)

# Here we will put type of experiment. its either optimal channel experiment or optimal route. 
exp = Experiment(agent, envRoute, 'optimal_route')
exp.run(10)
exp.plot()
exp.plot_avg_reward_per_arm_over_time()



