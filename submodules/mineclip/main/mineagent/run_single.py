import torch
import hydra

from mineclip.mineagent import features as F
from mineclip import SimpleFeatureFusion, MineAgent, MultiCategoricalActor
from mineclip.mineagent.batch import Batch


@torch.no_grad()
@hydra.main(config_name="conf", config_path=".", version_base="1.1")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_net_kwargs = cfg.feature_net_kwargs

    feature_net = {}
    for k, v in feature_net_kwargs.items():
        v = dict(v)
        cls = v.pop("cls")
        cls = getattr(F, cls)
        feature_net[k] = cls(**v, device=device)

    feature_fusion_kwargs = cfg.feature_fusion
    feature_net = SimpleFeatureFusion(
        feature_net, **feature_fusion_kwargs, device=device
    )

    actor = MultiCategoricalActor(
        feature_net,
        action_dim=[3, 3, 4, 25, 25, 8],
        device=device,
        **cfg.actor,
    )

    mine_agent = MineAgent(
        actor=actor,
    ).to(device)

    B = 32
    obs = {
        "compass": torch.rand((B, 4), device=device),
        "gps": torch.rand((B, 3), device=device),
        "voxels": torch.randint(
            low=0, high=26, size=(B, 3 * 3 * 3), dtype=torch.long, device=device
        ),
        "biome_id": torch.randint(
            low=0, high=167, size=(B,), dtype=torch.long, device=device
        ),
        "prev_action": torch.randint(
            low=0, high=88, size=(B,), dtype=torch.long, device=device
        ),
        "prompt": torch.rand((B, 512), device=device),
        "rgb": torch.rand((B, 512), device=device),
    }
    pi_out = mine_agent(Batch(obs=obs))
    print(pi_out.act)
    print("Inference successful")


if __name__ == "__main__":
    main()
