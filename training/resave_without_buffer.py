# resave_without_buffer.py

import argparse
from argparse import Namespace
import os
import sys
import torch


def read_command(argv) -> Namespace:
    """
    Reads in command line options that specify which agent
    zip files to resave without their replay buffer experiences.
    """

    usage_str = """
    USAGE:      python resave_without_buffer.py -p {path} [{path} ...]
    NOTE:       Produces a lean copy named '{stem}_no_buffer_exp.zip'
                next to each original. Setting replay_buffer and
                replay_position to None (rather than removing them)
                keeps the existing CustomDDPG/FrameDDPG load()
                compatible.

    EXAMPLE:    python resave_without_buffer.py
                    -p ../outputs/saved_agents/frameddpg_halfcheetah_mdp_2026-03-19_23-45-45/agent2_seed88/ver_980.zip
    """

    parser = argparse.ArgumentParser(usage=usage_str)

    parser.add_argument("-p", "--paths",
                        type=str, nargs="+", required=True,
                        metavar="P", help="One or more paths to \
                                CustomDDPG/FrameDDPG agent zip files \
                                to resave without their replay buffer.")

    return parser.parse_args(argv)


def resave_without_buffer(zip_path: str) -> str:
    """
    Loads an agent zip saved by CustomDDPG/FrameDDPG, clears the
    replay buffer entry, and writes a lean copy to the same directory
    named '{original stem}_no_buffer_exp.zip'.
    """
    saved = torch.load(zip_path, weights_only=False)

    saved["replay_buffer"] = None
    saved["replay_position"] = None

    directory, filename = os.path.split(zip_path)
    stem, _ = os.path.splitext(filename)
    out_path = os.path.join(directory, f"{stem}_no_buffer_exp.zip")

    torch.save(saved, out_path)
    return out_path


def main() -> None:
    """
    Runs resave_without_buffer.py.
    """

    args = read_command(sys.argv[1:])

    for path in args.paths:
        if not os.path.isfile(path):
            print(f"SKIP (not a file): {path}", file=sys.stderr)
            continue

        before_mb = os.path.getsize(path) / (1024 * 1024)
        out_path = resave_without_buffer(path)
        after_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"{path}\n  -> {out_path}"
              f"\n  {before_mb:.1f} MB -> {after_mb:.1f} MB")


if __name__ == "__main__":
    main()
