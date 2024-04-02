from __future__ import annotations


import minedojo
from minedojo.sim.inventory import InventoryItem
import numpy as np

from .dense_reward import MobCombatDenseRewardWrapper


class CombatSpiderDenseRewardEnv(MobCombatDenseRewardWrapper):
    def __init__(
        self,
        step_penalty: float | int,
        attack_reward: float | int,
        success_reward: float | int,
    ):
        max_spawn_range = 10
        distance_to_axis = int(max_spawn_range / np.sqrt(2))
        spawn_range_low = (-distance_to_axis, 1, -distance_to_axis)
        spawn_range_high = (distance_to_axis, 1, distance_to_axis)

        env = minedojo.make(
            "Combat",
            target_names="spider",
            target_quantities=1,
            reward_weights=success_reward,
            # start_position=pos,
            initial_inventory=[
                InventoryItem(slot=0, name="diamond_sword", variant=None, quantity=1),
                InventoryItem(slot=36, name="diamond_boots", variant=None, quantity=1),
                InventoryItem(
                    slot=37, name="diamond_leggings", variant=None, quantity=1
                ),
                InventoryItem(
                    slot=38, name="diamond_chestplate", variant=None, quantity=1
                ),
                InventoryItem(slot=39, name="diamond_helmet", variant=None, quantity=1),
                InventoryItem(slot=40, name="shield", variant=None, quantity=1),
            ],
            initial_mobs="spider",
            initial_mob_spawn_range_low=spawn_range_low,
            initial_mob_spawn_range_high=spawn_range_high,
            image_size=(160, 256),
            world_seed=123,
            specified_biome="sunflower_plains",
            fast_reset=True,
            fast_reset_random_teleport_range=0,
            allow_mob_spawn=False,
            use_voxel=True,
            always_night=True,
            initial_weather="clear",
        )
        super().__init__(
            env=env,
            step_penalty=step_penalty,
            attack_reward=attack_reward,
        )

        # reset cmds, call before `env.reset()`
        self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]

        self._episode_len = 500
        self._elapsed_steps = 0
        self._first_reset = True

    def reset(self, **kwargs):
        self._elapsed_steps = 0

        if not self._first_reset:
            for cmd in self._reset_cmds:
                self.env.unwrapped.execute_cmd(cmd)
            self.unwrapped.set_time(18000)
            self.unwrapped.set_weather("clear")
        self._first_reset = False

        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._episode_len:
            done = True
        return obs, reward, done, info
