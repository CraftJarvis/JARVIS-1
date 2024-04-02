from pathlib import Path

VPT_MODEL_PATH = Path(__file__).parent / "weights" / "vpt" / "2x.model"
VPT_WEIGHT_PATH = Path(__file__).parent / "weights" / "steve1" / "steve1.weights"
PRIOR_WEIGHT_PATH = Path(__file__).parent / "weights" / "steve1" / "steve1_prior.pt"
MINECLIP_WEIGHT_PATH = Path(__file__).parent / "weights" / "mineclip" / "attn.pth"