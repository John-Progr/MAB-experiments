import numpy as np 
import requests

class WirelessChannelEnv:

    def __init__(self, channels, reward_endpoint=None, timeout=5):

        """
        Request to a testbed as a service environment to get throughput


        Args:
            channels: List of available channels.
            reward_endpoint: API endpoint to fetch rewards from.
            timeout: Timeout for API requests (in seconds)

        """
        self.channels = channels
        self.reward_endpoint = reward_endpoint
        self.timeout = timeout


    def get_reward(self, channel):
        """
        Fetch reward for a given channel, either via API or simulation

        """
        if self.reward_endpoint:
            try:
                response = requests.get(
                    self.reward_endpoint,
                    params={"channel": channel},
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                return data.get("reward", 0.0)
            except requests.RequestException as e:
                print(f"Error Fetching reward: {e}")
                return 0.0
                
        return np.random.normal(loc=channel*2, scale=3.0)