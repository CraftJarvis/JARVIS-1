from tqdm import tqdm
from mineclip import CombatSpiderDenseRewardEnv


if __name__ == "__main__":
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
            action = env.action_space.no_op()
            obs, reward, done, info = env.step(action)
            pbar.update(1)
        print(f"{i+1}-th episode ran successful!")
    env.close()
