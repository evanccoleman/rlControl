# environments.py

import gymnasium as gym

from envs.pomdp_wrapper import POMDPWrapper

def create_env(env_type: str = None,
               quiet: bool = False,
               pomdp_type: str = None,
               render_mode: str = None,
               trackbodyid: int = None,
               ):
    """
    Creates a Gymnasium environment.

    Can make the environment partially observable
    using the POMDPWrapper class.

    If render_mode is None, it is derived from quiet
    (quiet=True -> None, quiet=False -> "human"). Pass
    render_mode explicitly (e.g. "rgb_array") to override.

    quiet=True always suppresses a live "human" viewer even when
    render_mode was explicitly set to "human". Offscreen modes like
    "rgb_array" pass through, so video recording still works when
    both quiet and save_video are requested.

    If trackbodyid is provided, the underlying MuJoCo env is
    created with a tracking camera following that body.
    """

    # derive render_mode from quiet if not explicitly provided
    if render_mode is None:
        render_mode = "human" if not quiet else None

    # quiet always wins over a live viewer; offscreen modes are fine
    if quiet and render_mode == "human":
        render_mode = None

    env_kwargs = {"render_mode": render_mode}

    # attach tracking camera if a body id was provided
    # (type 1 = mjCAMERA_TRACKING)
    if trackbodyid is not None:
        env_kwargs["default_camera_config"] = {
            "type": 1,
            "trackbodyid": trackbodyid,
            "distance": 4.0,
        }

    # make pomdp or mdp env
    if pomdp_type is not None:
        env = POMDPWrapper(env_type, pomdp_type=pomdp_type, **env_kwargs)
    else:
        env = gym.make(env_type, **env_kwargs)

    # track non-discounted returns automatically
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env
