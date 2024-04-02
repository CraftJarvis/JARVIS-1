import numpy as np
import torch
import hydra

from tqdm import tqdm
from mineclip.mineagent import features as F
from mineclip import SimpleFeatureFusion, MineAgent, MultiCategoricalActor
from mineclip.mineagent.batch import Batch
from mineclip import CombatSpiderDenseRewardEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_obs(env_obs):
    """
    Here you preprocess the raw env obs to pass to the agent.
    Preprocessing includes, for example, use MineCLIP to extract image feature and prompt feature,
    flatten and embed voxel names, mask unused obs, etc.

    Here we just use random vectors for demo purpose.
    """
    B = 1
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
    return Batch(obs=obs)


def transform_action(action):
    """
    Map agent action to env action.
    """
    assert action.ndim == 2
    action = action[0]
    action = action.cpu().numpy()
    if action[-1] != 0 or action[-1] != 1 or action[-1] != 3:
        action[-1] = 0
    action = np.concatenate([action, np.array([0, 0])])
    return action


@torch.no_grad()
@hydra.main(config_name="conf", config_path=".", version_base="1.1")
def main(cfg):

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

    env = CombatSpiderDenseRewardEnv(
        step_penalty=0,
        attack_reward=1,
        success_reward=10,
    )

    for i in tqdm(range(2), desc="Episode"):
        obs = env.reset()
        done = False
        pbar = tqdm(desc="Step")
        while not done:
            obs = preprocess_obs(obs)
            action = transform_action(mine_agent(obs).act)
            obs, reward, done, info = env.step(action)
            pbar.update(1)
        print(f"{i+1}-th episode ran successful!")
    env.close()


if __name__ == "__main__":
    main()
