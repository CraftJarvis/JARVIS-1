import numpy as np
import torch


class FrameBuffer:
    def __init__(self):
        self.frames = []

    def add_frame(self, frame: np.ndarray):
        self.frames.append(frame)
        if self.__len__() > 16:
            self.frames.pop(0)

    def to_numpy(self):
        return np.array(self.frames)

    def to_torch(self, device):
        return torch.Tensor(np.array(self.frames)).to(device).unsqueeze(0)

    def reset(self):
        self.frames = []

    def ready(self):
        return self.__len__() >= 16

    def __len__(self):
        return len(self.frames)

class QueueFrameBuffer:
    """Consumes frames via add_frame, but instead of deleting frames when it
    reaches capacity, it stores all frames. It can be used as an iterator to return batches
    of frames of a specified size."""

    seq_len = 16

    def __init__(self):
        self.frames = []
        self.idx = 0

    def add_frame(self, frame: np.ndarray):
        self.frames.append(frame)

    def __iter__(self):
        self.idx = 0
        return self

    def __len__(self):
        return len(self.frames) - self.seq_len + 1

    def __next__(self):
        if self.idx > len(self.frames) - self.seq_len:
            raise StopIteration
        else:
            frames = torch.Tensor(np.array(self.frames[self.idx: self.idx + self.seq_len])).unsqueeze(0)
            # print(frames.shape)
            self.idx += 1
            return frames