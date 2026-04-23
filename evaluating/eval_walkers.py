# eval_walkers.py

import sys
import os
from datetime import datetime as dt

# add project root to path so shared packages are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym

from evaluating.config import read_command
from agents.factory import create_agent
from agents.episodes import run_many_episodes, set_seed
from envs.environments import create_env


def main() -> None:
    """
    Runs eval_walkers.py.

    Loads a trained agent from a zip file and runs it for a specified
    number of evaluation episodes. Optionally records mp4 video of
    each episode to ../outputs/videos/.
    """

    # read in the options from the command line
    args = read_command(sys.argv[1:])

    # filename format: {agent_type}_{env_type}_{ispomdp}_{datetime}/...
    # parse: agent_type from the grandparent dir of the zip
    filename = args.load_agent.split("/")[-3]
    agent_type = filename.split("_")[0]

    # stamp this eval run so output paths don't collide
    current_datetime = dt.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ensure output directory structure exists
    os.makedirs("../outputs/videos", exist_ok=True)

    # seed packages so eval is deterministic within a run
    set_seed(0)

    # create env; wrap with RecordVideo if video saving was requested.
    # render_mode="rgb_array" + quiet=True guarantees no live viewer
    # even when -v and -q are combined; frames still flow to the mp4.
    if args.save_video:
        agent_dir = args.load_agent.split("/")[-2]
        zip_stem = os.path.splitext(
            os.path.basename(args.load_agent))[0]
        video_dir = (f"../outputs/videos/"
                     f"{filename}/"
                     f"{agent_dir}/"
                     f"{zip_stem}_eval_{current_datetime}")
        env = create_env(env_type=args.env_type,
                         quiet=args.quiet,
                         pomdp_type=args.pomdp_type,
                         render_mode="rgb_array",
                         trackbodyid=args.trackbodyid,
                         )
        env = gym.wrappers.RecordVideo(env,
                                        video_folder=video_dir,
                                        episode_trigger=lambda episode_id: True,
                                        disable_logger=True,
                                        )
    else:
        env = create_env(env_type=args.env_type,
                         quiet=args.quiet,
                         pomdp_type=args.pomdp_type,
                         )
    env.reset(seed=0)

    # load agent (agent_type is inferred from the zip's path)
    agent = create_agent(agent_type=agent_type,
                         load_agent=args.load_agent,
                         env=env,
                         seed=0,
                         )

    # run evaluation episodes (with a tqdm progress bar)
    avg_reward = run_many_episodes(agent,
                                   env,
                                   num_episodes=args.num_test,
                                   agent_type=agent_type,
                                   discount=args.discount,
                                   progress=True,
                                   )
    print(f"average return over {args.num_test} episodes: {avg_reward}")

    # close env
    env.close()


if __name__ == "__main__":

    """
    Note to self: by defining a main() for this file,
    other main functions in the same directory
    are isolated from each other, so I can run
    particular .py files when I want.
    """
    main()
