import json
import os
import pickle
from typing import Optional, Tuple

import cv2
import numpy as np
import math

import jarvis.steveI.steveI_lib.utils.video_utils


class EpisodeStorage:
    def __init__(self, episode_dirpath, check_valid=False):
        """filepath cannot have a '.'"""
        assert '.' not in episode_dirpath
        self.update_episode_dirpath(episode_dirpath)
        self.frame_count = 0
        self.frames = []
        self.actions = []
        self.embeds_attn = []

        if check_valid:
            assert self.is_valid()

    def update_episode_dirpath(self, episode_dirpath):
        self.episode_dirpath = episode_dirpath
        self.frames_dirpath = os.path.join(self.episode_dirpath, 'frames.mp4')
        self.actions_filepath = os.path.join(self.episode_dirpath, 'actions.pkl')
        self.embeds_attn_filepath = os.path.join(self.episode_dirpath, 'embeds_attn.pkl')
        self.metadata_filepath = os.path.join(self.episode_dirpath, 'metadata.json')
        self.len_filepath = os.path.join(self.episode_dirpath, 'len.txt')
        self.augmented_embeds_filepath = os.path.join(episode_dirpath, 'augmented_embeds.pkl')

    def append(self, agent_input_pov: np.ndarray, action: dict, video_embed_attn: Optional[np.ndarray]):
        assert agent_input_pov.shape == (128, 128, 3)
        im_save = cv2.cvtColor(agent_input_pov, cv2.COLOR_RGB2BGR)
        self.frames.append(im_save)
        self.actions.append(action)
        self.embeds_attn.append(video_embed_attn)
    
    def load_frames(self, only_range=None, to_rgb=True):
        """Loads the frames by reading from the MP4 file."""
        frames = steve1.utils.video_utils.load_video_to_lst(self.frames_dirpath, to_rgb=to_rgb,
                                                         only_range=only_range, length=len(self))
        return frames

    def load_first_frame(self):
        if not os.path.exists(self.frames_dirpath):
            raise FileNotFoundError(f'Could not find frames at {self.frames_dirpath}')
        cap = cv2.VideoCapture(self.frames_dirpath)
        ret, frame = cap.read()
        assert ret, 'Frames MP4 exists but could not read first frame'
        cap.release()
        return frame

    def load_actions(self) -> list[dict]:
        with open(self.actions_filepath, 'rb') as f:
            actions = pickle.load(f)
        return actions

    def save_actions(self, actions: list[dict]):
        try:
            with open(self.actions_filepath, 'wb') as f:
                pickle.dump(actions, f)
        except KeyboardInterrupt:
            print('\033[93mPlease wait for save to finish (do not send KeyboardInterrupt '
                  'again unless you know what you\'re doing!)\033[0m')
            with open(self.actions_filepath, 'wb') as f:
                pickle.dump(actions, f)
            raise KeyboardInterrupt

    def load_embeds_attn(self) -> list[np.ndarray]:
        """Note: first 16 are always None."""
        with open(self.embeds_attn_filepath, 'rb') as f:
            embeds_attn = pickle.load(f)
        # Fix since some are saved as lists in the wrong dtype
        for i in range(len(embeds_attn)):
            if embeds_attn[i] is not None and not isinstance(embeds_attn[i], np.ndarray):
                embeds_attn[i] = np.array(embeds_attn[i], dtype=np.float32)
                embeds_attn[i] = embeds_attn[i].reshape(1, -1)
        return embeds_attn
    
    def save_embeds_attn(self, embeds_attn: list[np.ndarray]):
        with open(self.embeds_attn_filepath, 'wb') as f:
            pickle.dump(embeds_attn, f)

    def load_metadata(self) -> dict:
        with open(self.metadata_filepath, 'r') as f:
            metadata = json.load(f)
        return metadata

    def save_metadata(self, metadata: dict):
        with open(self.metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=4)

    def is_valid(self, for_training: bool = False, min_frames_training: int = 720,
                 expensive: bool = False) -> Tuple[bool, str]:
        """Returns True if the video is valid, False otherwise.

        To be valid, the episode must:
            - have at least 16 frames
            - have no missing frames in the frames directory (i.e., no gaps in the frame numbers)
            - have metadata['num_frames'] == len(os.listdir(self.frames_dirpath))

        If for_training is True, then the video must also be valid for training.
        To be valid for training, the episode must:
            - have at least min_frames_training frames (currently 30 seconds at 24 fps)
        """
        if not os.path.exists(self.episode_dirpath):
            return False, 'dirpath does not exist'
        if not os.path.exists(self.frames_dirpath):
            return False, 'frames_dirpath does not exist'
        if not os.path.exists(self.metadata_filepath):
            return False, 'metadata_filepath does not exist'
        if not os.path.exists(self.actions_filepath):
            return False, 'actions_filepath does not exist'
        if not os.path.exists(self.embeds_attn_filepath):
            return False, 'embeds_attn_filepath does not exist'
        if not os.path.exists(self.len_filepath):
            return False, 'len_filepath does not exist'

        if len(self) < 16:
            return False, 'less than 16 frames'

        if for_training:
            if len(self) < min_frames_training:
                return False, f'for_training: less than min_frames_training ({min_frames_training}) frames'

        return True, 'valid'

    def save_episode(self):
        """Save the episode"""
        if os.path.exists(self.episode_dirpath):
            raise ValueError('episode_dirpath cannot already exist')
        os.makedirs(self.episode_dirpath)
        # Save the frames as an MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.frames_dirpath, fourcc, 20.0, (128, 128))

        for frame in self.frames:
            out.write(frame)
        out.release()

        # Save the actions
        with open(self.actions_filepath, 'wb') as f:
            pickle.dump(self.actions, f)

        # Save the embeds_attn
        with open(self.embeds_attn_filepath, 'wb') as f:
            pickle.dump(self.embeds_attn, f)

        # Save length text
        min_len = min(len(self.frames), len(self.actions), len(self.embeds_attn))
        with open(self.len_filepath, 'w') as f:
            f.write(str(min_len))
    
    def __len__(self):
        with open(self.len_filepath, 'r') as f:
            return int(f.read())

    def check_length(self):
        """Checks that the length of the episode is the same for all components (including len.txt)."""
        frames = self.load_frames()
        actions = self.load_actions()
        embeds_attn = self.load_embeds_attn()
        same_len = len(frames) == len(actions) == len(embeds_attn)
        if not same_len:
            raise ValueError(f'Component lengths do not match: {len(frames)}, {len(actions)}, {len(embeds_attn)}'
                             f' (episode: {os.path.basename(self.episode_dirpath)})')
        if len(frames) != len(self):
            raise ValueError(f'len.txt ({len(self)}) does not match actual length {len(frames)}'
                             f' (episode: {os.path.basename(self.episode_dirpath)})')

    def get_num_chunks(self, chunk_size: int) -> int:
        """Returns the number of chunks that the episode can be split into.
        """
        return math.ceil(len(self) / chunk_size)

    def get_chunk_embeds_at_idx(self, idx: int, chunk_size: int, episode_embeds: list[np.ndarray]) -> list[np.ndarray]:
        """Returns the embeds_attn for the chunk at the given idx."""
        assert idx < self.get_num_chunks(chunk_size), "idx must be less than the number of chunks."

        if idx == 0:
            # Ignore first 16 frames
            return episode_embeds[16: (idx + 1) * chunk_size]
        else:
            return episode_embeds[idx * chunk_size: (idx + 1) * chunk_size]

    def get_chunk_frames_at_idx(self, chunk_idx: int, chunk_size: int) -> list[np.ndarray]:
        """Returns the frames for the chunk at the given idx."""
        assert chunk_idx < self.get_num_chunks(chunk_size), "idx must be less than the number of chunks."

        # No need to ignore first 16 frames here, since this is used for saving purposes, not checking closeness
        indices = list(range(chunk_idx * chunk_size, (chunk_idx + 1) * chunk_size))
        frames = self.load_frames()
        chunk_frames = [frames[i] for i in indices]
        return chunk_frames

    def get_size_gb(self) -> float:
        """Returns the size of the episode in GB."""
        # Use os, dir only contains files
        total_size = 0
        for f in os.listdir(self.episode_dirpath):
            assert os.path.isfile(os.path.join(self.episode_dirpath, f))
            total_size += os.path.getsize(os.path.join(self.episode_dirpath, f))
        return total_size / 1000000000

    def is_term(self) -> bool:
        """Returns True if the episode was terminated, False otherwise."""
        return 'term' in os.path.basename(self.episode_dirpath)

    def get_term_reason(self) -> str:
        """Get the termination reason from the episode dirname."""
        if self.is_term():
            return os.path.basename(self.episode_dirpath).split('term')[1].split('_')[0]
        else:
            return 'Incomplete'
