from typing import (
    Dict, List, Tuple, Union, Callable, 
    Sequence, Mapping, Any, Optional
)

import cv2
import gymnasium as gym
from jarvis.assembly.scripts import CraftScript, SmeltScript, EquipScript
from jarvis.assembly.base import MarkBase
from jarvis.assembly.env import RenderWrapper
from jarvis.stark_tech.env_interface import MinecraftWrapper

class MarkCrafter(MarkBase):
    
    def __init__(
        self,
        env: gym.Env,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.craft_script = CraftScript(env=env)
        self.smelt_script = SmeltScript(env=env)
        self.equip_script = EquipScript(env=env)

    def reset(self): 
        super().reset()
        self.craft_script.reset()
        self.smelt_script.reset()
        self.equip_script.reset()

    def do(self, condition: str = '', *args, **kwargs):
        # TODO: handle the corner case that the inventory is open at the beginning
        # need a signal: close inventory(when crafting/smelting for the last time)
        self.reset()
        if condition == 'craft':
            result, error_message = self.craft_script.crafting(*args, **kwargs)
            self.record_frames += self.craft_script.outframes
            self.record_infos += self.craft_script.outinfos
        elif condition == 'smelt':
            result, error_message = self.smelt_script.smelting(*args, **kwargs)
            self.record_frames += self.smelt_script.outframes
            self.record_infos += self.smelt_script.outinfos
        elif condition == 'equip':
            result, error_message = self.equip_script.equip_item(*args, **kwargs)
            self.record_frames += self.equip_script.outframes
            self.record_infos += self.equip_script.outinfos
        else:
            raise ValueError("Condition must be `craft`, `smelt` or `equip`. ")
        return result, error_message

from jarvis.steveI.steveI_text import SteveIText as SteveI
class MarkI(MarkBase):
    
    def __init__(self, env, **kwargs):
        self.env = env
        self.kwargs = kwargs
        self.REC_FRAMES = False
        self.mark_policy = SteveI(env=self.env)
        self.mark_policy.reset()
        self.mark_shell = MarkCrafter(env=self.env)

    def reset(self):
        # self.record_frames = []
        self.record_infos = []

        self.mark_policy.reset()
        self.mark_shell.reset()

    def post_infos(self, infos):
        discard_info_keys = ['location_stats', 'pickup', 'break_item', 'craft_item', 'mine_block', 'kill_entity']
        if not self.REC_FRAMES:
            discard_info_keys.append('pov')
        record_infos = []
        for info in infos:
            record_info = {}
            for k, v in info.items():
                if k not in discard_info_keys:
                    record_info[k] = v
            record_infos += [record_info]
        return record_infos
    
    def do(self, condition: str = '', **kwargs):
        if condition in ['craft', 'smelt', 'equip']:
            # self.mark_crafter.reset()
            # self.latest_mark = self.mark_crafter
            res, msg = self.mark_shell.do(condition, **kwargs)
            # self.record_frames += self.mark_crafter.record_frames
            # self.record_infos += self.mark_crafter.record_infos
            self.record_infos += self.post_infos(self.mark_shell.record_infos)
            return res, msg
        else:
            # self.mark_miner.reset()
            # self.latest_mark = self.mark_miner
            res, msg = self.mark_policy.do(condition, **kwargs)
            # self.record_frames += self.mark_miner.record_frames
            # self.record_infos += self.mark_miner.record_infos
            self.record_infos += self.post_infos(self.mark_policy.record_infos)
            return res, msg
    
    # def make_traj_video(self, file_name = 'dummy'):
    #     container = av.open(f'{file_name}.mp4', mode='w', format='mp4')
    #     stream = container.add_stream('h264', rate=30)
    #     stream.width = 640 
    #     stream.height = 360
    #     stream.pix_fmt = 'yuv420p'

    #     curr_prompt = "null"
    #     curr_goal = {}
    #     # for i,frame in enumerate(self.record_frames):
    #     for i,info in enumerate(self.record_infos):
    #         frame = info['pov']
    #         if i in self.record_goals.keys():
    #             curr_goal = self.record_goals[i]
    #         if i in self.record_prompts.keys():
    #             curr_prompt = self.record_prompts[i]
    #         frame = frame.copy()
    #         cv2.putText(frame, f"Task: {self.current_task['task']}", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (234, 53, 70), 2)
    #         cv2.putText(frame, f"Step: {i}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (248, 102, 36), 2)
    #         cv2.putText(frame, f"Pos: [{int(info['location_stats']['xpos'])}, {int(info['location_stats']['ypos'])}, {int(info['location_stats']['zpos'])}]", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (248, 102, 36), 2)
    #         cv2.putText(frame, f"Goal: {curr_goal['goal']}", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (248, 102, 36), 2)
    #         cv2.putText(frame, f"Prompt: {curr_prompt}", (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (248, 102, 36), 2)
    #         frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
    #         for packet in stream.encode(frame):
    #             container.mux(packet)
    #     for packet in stream.encode():
    #         container.mux(packet)
    #     container.close()
    
    def noop_step(self):
        return self.env.step(self.env.noop_action())

if __name__ == '__main__':
    env = MinecraftWrapper("test")
    env = RenderWrapper(env)
    env.reset()
    
    mark = MarkI(env=env)
    mark.reset()


    result, error_message = mark.do('equip', target_item='diamond_boots')
    print(result, error_message)

    # result, error_message = mark.do('equip', target_item='diamond_helmet')
    result, error_message = mark.do('chop tree')

    # crafting
    result, error_message = mark.do('craft', target='oak_planks', target_num=30)
    print(result, error_message)
    result, error_message = mark.do('craft', target='stick', target_num=30)
    print(result, error_message)
    result, error_message = mark.do('craft', target='crafting_table', target_num=1)
    print(result, error_message)
    result, error_message = mark.do('craft', target='furnace', target_num=1)
    print(result, error_message)
    result, error_message = mark.do('craft', target='wooden_pickaxe', target_num=1)
    print(result, error_message)   

    # smelting
    result, error_message = mark.do('smelt', target='charcoal')
    print(result, error_message)
    result, error_message = mark.do('smelt', target='baked_potato')
    print(result, error_message)

    # crafting
    result, error_message = mark.do('craft', target='oak_planks', target_num=10)
    print(result, error_message)
    result, error_message = mark.do('equip', target_item='crafting_table')
    print(result, error_message)
    result, error_message = mark.do('craft', target='wooden_pickaxe', target_num=1)
    print(result, error_message)
    result, error_message = mark.do('craft', target='stick', target_num=10)
    print(result, error_message)

    env.close()