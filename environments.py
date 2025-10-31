import numpy as np 
import requests
import time

class WirelessChannelEnv:

    def __init__(self, source_ip, dest_ip, channels, reward_endpoint=None,):

        """
        Request to a testbed as a service environment to get throughput (our reward)


        Args:
            channels: List of available channels.
            reward_endpoint: API endpoint to fetch rewards from.

        """
      
        self.source_ip = source_ip
        self.dest_ip = dest_ip
        self.channels = channels
        self.reward_endpoint = reward_endpoint
       

    def send_request(self, channel):
        request_data = {
            "source": self.source_ip,
            "destination": self.dest_ip,
            "path": [],
            "wireless_channel": channel
        }


        max_retries = 3
        delay = 2

        for attempt in range (1, max_retries + 1):

            try:
                time.sleep(2)
                response = requests.post(self.reward_endpoint, json=request_data, timeout = 100)

                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"❌ Attempt {attempt}: HTTP {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Attempt {attempt}: Request error: {e}")
            

            if attempt < max_retries:
                sleep_time = delay * ( 2 ** (attempt - 1))
                print(f"⏳ Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("🚫 All retries failed.")
                return None

        


    def get_reward(self, channel):
        """
        Fetch reward for a given channel, either via API or simulation

        """
        if self.reward_endpoint:
             json_response = self.send_request(channel)
             print(json_response)
             reward = json_response["rate_mbps"]

             return reward
        else:

                
            return np.random.normal(loc=channel*2, scale=3.0)


