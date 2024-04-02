from __future__ import annotations

from typing import Literal
from collections import deque

import numpy as np
from gym import Wrapper


class AnimalZooDenseRewardWrapper(Wrapper):
    def __init__(
        self,
        env,
        entity: Literal["cow", "sheep"],
        step_penalty: float | int,
        nav_reward_scale: float | int,
        attack_reward: float | int,
    ):
        assert (
            "rays" in env.observation_space.keys()
        ), "Dense reward function requires lidar observation"
        super().__init__(env)

        self._entity = entity
        assert step_penalty >= 0, f"penalty must be non-negative"
        self._step_penalty = step_penalty
        self._nav_reward_scale = nav_reward_scale
        self._attack_reward = attack_reward

        self._weapon_durability_deque = deque(maxlen=2)
        self._consecutive_distances = deque(maxlen=2)
        self._distance_min = np.inf

    def reset(self, **kwargs):
        self._weapon_durability_deque.clear()
        self._consecutive_distances.clear()
        self._distance_min = np.inf

        obs = super().reset(**kwargs)

        entity_in_sight, distance = self._find_distance_to_entity_if_in_sight(obs)
        if entity_in_sight:
            distance = self._distance_min = min(distance, self._distance_min)
            self._consecutive_distances.append(distance)
        else:
            self._consecutive_distances.append(0)
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
        # nav reward
        entity_in_sight, distance = self._find_distance_to_entity_if_in_sight(obs)
        nav_reward = 0
        if entity_in_sight:
            distance = self._distance_min = min(distance, self._distance_min)
            self._consecutive_distances.append(distance)
            nav_reward = self._consecutive_distances[0] - self._consecutive_distances[1]
        nav_reward = max(0, nav_reward)
        nav_reward *= self._nav_reward_scale
        # reset distance min if attacking the entity because entity will run away
        if valid_attack > 0:
            self._distance_min = np.inf
        # total reward
        reward = attack_reward + nav_reward - self._step_penalty + _reward
        return obs, reward, done, info

    def _find_distance_to_entity_if_in_sight(self, obs):
        in_sight, min_distance = False, None
        entities, distances = (
            obs["rays"]["entity_name"],
            obs["rays"]["entity_distance"],
        )
        entity_idx = np.where(entities == self._entity)[0]
        if len(entity_idx) > 0:
            in_sight = True
            min_distance = np.min(distances[entity_idx])
        return in_sight, min_distance
