import cv2
import torch
import pickle
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from copy import deepcopy
from rich import print

from jarvis.arm.utils.impala_lib.impala_cnn import ImpalaCNN
from jarvis.arm.utils.impala_lib.goal_impala_cnn import GoalImpalaCNN
from jarvis.arm.utils.impala_lib.util import FanInInitReLULayer

def resize_image(img, target_resolution=(128, 128)):
    if type(img) == np.ndarray:
        img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    elif type(img) == torch.Tensor:
        img = F.interpolate(img, size=target_resolution, mode='bilinear')
    else:
        raise ValueError
    return img

class ImpalaCNNWrapper(nn.Module):
    
    def __init__(self, scale='1x', **kwargs):
        super().__init__()
        if scale == '1x':
            net_config = {
                'hidsize': 1024,
                'img_shape': [128, 128, 3],
                'impala_chans': [16, 32, 32],
                'impala_kwargs': {'post_pool_groups': 1},
                'impala_width': 4,
                'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
            }
        elif scale == '3x':
            net_config = {
                'hidsize': 3072,
                'img_shape': [128, 128, 3],
                'impala_chans': [16, 32, 32],
                'impala_kwargs': {'post_pool_groups': 1},
                'impala_width': 12,
                'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
            }
        else:
            assert False
        
        hidsize = net_config['hidsize']
        img_shape = net_config['img_shape']
        impala_width = net_config['impala_width']
        impala_chans = net_config['impala_chans']
        impala_kwargs = net_config['impala_kwargs']
        init_norm_kwargs = net_config['init_norm_kwargs']
        
        chans = tuple(int(impala_width * c) for c in impala_chans)
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True
        
        self.cnn = ImpalaCNN(
            outsize=256,
            output_size=hidsize,
            inshape=img_shape,
            chans=chans,
            nblock=2,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            first_conv_norm=False,
            **impala_kwargs,
            **kwargs,
        )

        self.linear = FanInInitReLULayer(
            256,
            hidsize,
            layer_type="linear",
            **self.dense_init_norm_kwargs,
        )

    def forward(self, img, goal_embeddings):
        assert len(img.shape) == 4
        img = resize_image(img, (128, 128))
        # print(img)
        img = img.to(dtype=torch.float32)  / 255.
        return self.linear(self.cnn(img))

class GoalImpalaCNNWrapper(nn.Module):
    
    def __init__(self, scale='1x', **kwargs):
        super().__init__()
        if scale == '1x':
            net_config = {
                'hidsize': 1024,
                'img_shape': [128, 128, 3],
                'impala_chans': [16, 32, 32],
                'impala_kwargs': {'post_pool_groups': 1},
                'impala_width': 4,
                'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
            }
        elif scale == '3x':
            net_config = {
                'hidsize': 3072,
                'img_shape': [128, 128, 3],
                'impala_chans': [16, 32, 32],
                'impala_kwargs': {'post_pool_groups': 1},
                'impala_width': 12,
                'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
            }
        else:
            assert False
        
        hidsize = net_config['hidsize']
        img_shape = net_config['img_shape']
        impala_width = net_config['impala_width']
        impala_chans = net_config['impala_chans']
        impala_kwargs = net_config['impala_kwargs']
        init_norm_kwargs = net_config['init_norm_kwargs']
        
        chans = tuple(int(impala_width * c) for c in impala_chans)
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True
        
        self.cnn = GoalImpalaCNN(
            outsize=256,
            output_size=hidsize,
            inshape=img_shape,
            chans=chans,
            nblock=2,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            first_conv_norm=False,
            **impala_kwargs,
            **kwargs,
        )

        self.linear = FanInInitReLULayer(
            256,
            hidsize,
            layer_type="linear",
            **self.dense_init_norm_kwargs,
        )

    def forward(self, img, goal_embeddings):
        '''
        img: BxT, 3, H, W, without normalization
        goal_embeddings: BxT, C
        '''
        img = resize_image(img, (128, 128))
        assert img.max() <= 1.0, "Input image should be normalized to [0, 1]"
        # img = img.to(dtype=torch.float32) / 255.
        return self.linear(self.cnn(img, goal_embeddings))

    def get_cam_layer(self):
        return [ block.conv1 for block in self.cnn.stacks[-1].blocks ] + [ block.conv0 for block in self.cnn.stacks[-1].blocks ]


def build_backbone(name, model_path = "", weight_path = "", **kwargs):
    assert name in ['impala_1x' 'impala_3x', 'goal_impala_1x', 'goal_impala_3x'], \
                    f"[x] backbone {name} is not surpported!"
    if name == 'impala_1x':
        return ImpalaCNNWrapper('1x', **kwargs)
    elif name == 'impala_3x':
        return ImpalaCNNWrapper('3x', **kwargs)
    elif name == 'goal_impala_1x':
        return GoalImpalaCNNWrapper('1x', **kwargs)
    elif name == 'goal_impala_3x':
        return GoalImpalaCNNWrapper('3x', **kwargs)


if __name__ == '__main__':
    vpt = create_backbone("goal_impala_1x")
    inp = torch.ones(2, 3, 128, 128)
    opt = vpt(inp)
    print(opt.shape)
