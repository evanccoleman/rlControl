# agents.py

from stable_baselines3 import PPO, DDPG, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import RecurrentPPO

def create_agent(agent_type, **kwargs):
    """
    Returns a new agent.
    """

    agent = None
    if agent_type == "ppo":
        agent = PPO(**kwargs)
    elif agent_type == "ddpg":
        agent = DDPG(**kwargs)
    elif agent_type == "td3":
        agent = TD3(**kwargs)
    elif agent_type == "sac":
        agent = SAC(**kwargs)
    elif agent_type == "rppo":
        agent = RecurrentPPO(**kwargs)
    else:
        raise Exception(f"Agent {agent_type} is not implemented.")

    # return fresh agent
    return agent
