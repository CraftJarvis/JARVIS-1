import torch
import cv2

MINECLIP_CONFIG = {
    'arch': "vit_base_p16_fz.v2.t2",
    'hidden_dim': 512,
    'image_feature_dim': 512,
    'mlp_adapter_spec': 'v0-2.t0',
    'pool_type': "attn.d2.nh8.glusw",
    'resolution': [160, 256],
    'ckpt': {
        'path': "data/weights/mineclip/attn.pth",
        'checksum': 'b5ece9198337cfd117a3bfbd921e56da'
    }
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PRIOR_INFO = {
    'mineclip_dim': 512,
    'latent_dim': 512,
    'hidden_dim': 512,
    'model_path': 'data/weights/steve1/steve1_prior.pt',
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
