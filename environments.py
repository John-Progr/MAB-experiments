import numpy as np 
import requests

class WirelessChannelEnv:

    def __init__(self, channels, reward_endpoint=None):
        self.channels = channels
        self.reward_endpoint = reward_endpoint

    def get_reward(self, channel):
        if self.reward_endpoint:
            response = requests.get(self.reward_endpoint, params={"channel": channel})
            response.raise_for_status()
            return response.json()["reward"]

        return np.random.normal(loc=channel*2, scale=3.0)