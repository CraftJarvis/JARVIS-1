from __future__ import annotations

from collections import deque

from gym import Wrapper


class MobCombatDenseRewardWrapper(Wrapper):
    def __init__(
        self,
        env,
        step_penalty: float | int,
        attack_reward: float | int,
    ):
        super().__init__(env)

        assert step_penalty >= 0, f"penalty must be non-negative"
        self._step_penalty = step_penalty
        self._attack_reward = attack_reward

        self._weapon_durability_deque = deque(maxlen=2)

    def reset(self, **kwargs):
        self._weapon_durability_deque.clear()
        obs = super().reset(**kwargs)
        self._weapon_durability_deque.append(obs["inventory"]["cur_durability"][0])

        return obs

    def step(self, action):
        obs, _reward, done, info = super().step(action)

        self._weapon_durability_deque.append(obs["inventory"]["cur_durability"][0])
        valid_attack = (
            self._weapon_durability_deque[0] - self._weapon_durability_deque[1]
        )
        # when dying, the weapon is gone and durability changes to 0
        valid_attack = 1.0 if valid_attack == 1.0 else 0.0

        # attack reward
        attack_reward = valid_attack * self._attack_reward
        # total reward
        reward = attack_reward - self._step_penalty + _reward

        done = done or bool(obs["inventory"]["cur_durability"][0] <= 0)
        return obs, reward, done, info
