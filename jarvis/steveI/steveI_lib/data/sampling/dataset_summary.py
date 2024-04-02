import argparse
from jarvis.steveI.steveI_lib.data.minecraft_dataset import get_valid_episodes


def main(args):
    valid_episodes = get_valid_episodes(args.data_dir, args.T, args.min_btwn_goals)
    # a list of (episode_dirpath, episode_len) tuples
    total_len = 0
    for _, episode_len in valid_episodes:
        total_len += episode_len

    print(f'Total length in frames: {total_len / 1e6:.2f}M')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/ssd005/projects/mc_trajs/datasets/dataset_contractor')
    parser.add_argument('--min_btwn_goals', type=int, default=15)
    parser.add_argument('--T', type=int, default=640)

    args = parser.parse_args()
    main(args)
