from tqdm import tqdm
from mineclip import HuntCowDenseRewardEnv


if __name__ == "__main__":
    env = HuntCowDenseRewardEnv(
        step_penalty=0,
        nav_reward_scale=0.1,
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
