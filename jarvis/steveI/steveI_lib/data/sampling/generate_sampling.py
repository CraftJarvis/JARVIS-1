import argparse
import random
import os

from jarvis.steveI.steveI_lib.data.minecraft_dataset import get_valid_episodes

DATASET_DIRS = {
    'contractor': 'data/dataset_contractor',
}


def generate_train_val_split(episodes, args):
    """Episodes is a list of (episode_dirpath, episode_len) tuples."""
    train_frames, val_frames = args.train_frames, args.val_frames
    # First, assert that the total number of frames is enough
    total_len = 0
    for _, episode_len in episodes:
        total_len += episode_len
    if train_frames == -1:
        train_frames = total_len - val_frames
    assert total_len >= train_frames + val_frames, f'Total length of episodes is {total_len}, ' \
                                                   f'but train_frames + val_frames is {train_frames + val_frames}'
    if args.val_from is not None:
        print(f'Using validation set from {args.val_from}')
        val_path = os.path.join(args.output_dir, f'{args.val_from}_val.txt')
        with open(val_path, 'r') as f:
            val_episodes = [line.strip() for line in f.readlines()]
        val_episodes = [(episode, 0) for episode in val_episodes]

        # Get rid of val episodes from episodes
        episodes = [episode for episode in episodes if episode[0] not in val_episodes]

        # Shuffle the remaining episodes
        random.shuffle(episodes)

        # Get the train episodes
        train_episodes = []
        train_len = 0
        for episode_dirpath, episode_len in episodes:
            train_episodes.append((episode_dirpath, episode_len))
            train_len += episode_len
            if train_len >= train_frames:
                break

        return train_episodes, val_episodes
    else:
        # Shuffle episodes
        random.shuffle(episodes)
        # Keep adding episodes until we have enough frames
        train_episodes = []
        train_len = 0
        for episode_dirpath, episode_len in episodes:
            train_episodes.append((episode_dirpath, episode_len))
            train_len += episode_len
            if train_len >= train_frames:
                break
        # Now add episodes to val until we have enough frames
        val_episodes = []
        val_len = 0
        for episode_dirpath, episode_len in episodes[len(train_episodes):]:
            val_episodes.append((episode_dirpath, episode_len))
            val_len += episode_len
            if val_len >= val_frames:
                break
        return train_episodes, val_episodes


def get_total_frames(episodes):
    total_len = 0
    for _, episode_len in episodes:
        total_len += episode_len
    return total_len


def get_first_n_frames(episodes, n):
    first_n_episodes = []
    first_n_len = 0
    for episode_dirpath, episode_len in episodes:
        first_n_episodes.append((episode_dirpath, episode_len))
        first_n_len += episode_len
        if first_n_len >= n:
            break
    return first_n_episodes


def get_mixture(datasets_episodes, ps, total_frames):
    # shuffle each dataset
    for dataset_episodes in datasets_episodes:
        random.shuffle(dataset_episodes)

    # Compute frames per dataset
    frames_per_dataset = [int(p * total_frames) for p in ps]

    # Get that many frames from each dataset
    mixture_episodes = []
    for dataset_episodes, frames in zip(datasets_episodes, frames_per_dataset):
        mixture_episodes += get_first_n_frames(dataset_episodes, frames)

    return mixture_episodes


def main(args):
    random.seed(42)
    if args.type == 'neurips':
        episodes_seed0 = get_valid_episodes(DATASET_DIRS['contractor'], args.T, args.min_btwn_goals)
        train_episodes, val_episodes = generate_train_val_split(episodes_seed0, args)
    else:
        raise NotImplementedError
    print(f'Number of train episodes: {len(train_episodes)}')
    print(f'Number of val episodes: {len(val_episodes)}')
    print(f'Total train length in frames: {sum([episode_len for _, episode_len in train_episodes]) / 1e6:.2f}M')
    print(f'Total val length in frames: {sum([episode_len for _, episode_len in val_episodes]) / 1e6:.2f}M')

    # Write to two separate text files: {name}_train.txt and {name}_val.txt
    with open(f'{args.output_dir}/{args.name}_train.txt', 'w') as f:
        for episode_dirpath, _ in train_episodes:
            f.write(f'{episode_dirpath}\n')
    with open(f'{args.output_dir}/{args.name}_val.txt', 'w') as f:
        for episode_dirpath, _ in val_episodes:
            f.write(f'{episode_dirpath}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='seed0')
    parser.add_argument('--name', type=str, default='seed0')
    parser.add_argument('--output_dir', type=str, default='/ssd005/projects/mc_trajs/samplings')
    parser.add_argument('--val_frames', type=int, default=1_300_000)
    parser.add_argument('--val_from', type=str, default=None)
    parser.add_argument('--train_frames', type=int, default=11_000_000)
    parser.add_argument('--min_btwn_goals', type=int, default=15)
    parser.add_argument('--T', type=int, default=640)

    args = parser.parse_args()
    main(args)
