import hashlib
from mineclip import MineCLIP


def load(cfg, device):
    cfg = cfg.copy()
    ckpt = cfg.pop("ckpt")
    assert hashlib.md5(open(ckpt['path'], "rb").read()).hexdigest() == ckpt['checksum'], "broken ckpt"

    model = MineCLIP(**cfg).to(device)
    model.load_ckpt(ckpt['path'], strict=True)
    return model
