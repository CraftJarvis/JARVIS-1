import random
from typing import (
    Sequence, List, Mapping, Dict, 
    Callable, Any, Tuple, Optional, Union
)
from jarvis.stark_tech.env_interface import MinecraftWrapper
from jarvis.utils import write_video
from jarvis.assembly.scripts.craft_agent import Worker

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

CAMERA_SCALER = 360.0 / 2400.0
WIDTH, HEIGHT = 640, 360

# compute slot position
KEY_POS_INVENTORY_WO_RECIPE = {
    'resource_slot': {
        'left-top': (329, 114), 
        'right-bottom': (365, 150), 
        'row': 2, 
        'col': 2,
        'prefix': 'resource', 
        'start_id': 0, 
    },
    'result_slot': {
        'left-top': (385, 124), 
        'right-bottom': (403, 142),
        'row': 1, 
        'col': 1,
        'prefix': 'result', 
        'start_id': 0, 
    },
    'hotbar_slot': {
        'left-top': (239, 238), 
        'right-bottom': (401, 256),
        'row': 1, 
        'col': 9, 
        'prefix': 'inventory', 
        'start_id': 0, 
    }, 
    'inventory_slot': {
        'left-top': (239, 180), 
        'right-bottom': (401, 234), 
        'row': 3, 
        'col': 9,
        'prefix': 'inventory',
        'start_id': 9,
    }, 
    'recipe_slot': {
        'left-top': (336, 158),
        'right-bottom': (356, 176),
        'row': 1, 
        'col': 1,
        'prefix': 'recipe', 
        'start_id': 0,
    }
}
KEY_POS_TABLE_WO_RECIPE = {
    'resource_slot': {
        'left-top': (261, 113), 
        'right-bottom': (315, 167), 
        'row': 3, 
        'col': 3,
        'prefix': 'resource', 
        'start_id': 0, 
    },
    'result_slot': {
        'left-top': (351, 127), 
        'right-bottom': (377, 153),
        'row': 1, 
        'col': 1,
        'prefix': 'result', 
        'start_id': 0, 
    },
    'hotbar_slot': {
        'left-top': (239, 238), 
        'right-bottom': (401, 256),
        'row': 1, 
        'col': 9, 
        'prefix': 'inventory', 
        'start_id': 0, 
    }, 
    'inventory_slot': {
        'left-top': (239, 180), 
        'right-bottom': (401, 234), 
        'row': 3, 
        'col': 9,
        'prefix': 'inventory',
        'start_id': 9,
    }, 
    'recipe_slot': {
        'left-top': (237, 131),
        'right-bottom': (257, 149),
        'row': 1, 
        'col': 1,
        'prefix': 'recipe', 
        'start_id': 0,
    }
}
def COMPUTE_SLOT_POS(KEY_POS):
    result = {}
    for k, v in KEY_POS.items():
        left_top = v['left-top']
        right_bottom = v['right-bottom']
        row = v['row']
        col = v['col']
        prefix = v['prefix']
        start_id = v['start_id']
        width = right_bottom[0] - left_top[0]
        height = right_bottom[1] - left_top[1]
        slot_width = width // col
        slot_height = height // row
        slot_id = 0
        for i in range(row):
            for j in range(col):
                result[f'{prefix}_{slot_id + start_id}'] = (
                    left_top[0] + j * slot_width + (slot_width // 2), 
                    left_top[1] + i * slot_height + (slot_height // 2),
                )
                slot_id += 1
    return result
SLOT_POS_INVENTORY_WO_RECIPE = COMPUTE_SLOT_POS(KEY_POS_INVENTORY_WO_RECIPE)
SLOT_POS_TABLE_WO_RECIPE = COMPUTE_SLOT_POS(KEY_POS_TABLE_WO_RECIPE)

SLOT_POS_MAPPING = {
    'inventory_w_recipe': None,  # SLOT_POS_INVENTORY_W_RECIPE
    'inventory_wo_recipe': SLOT_POS_INVENTORY_WO_RECIPE,
    'crating_table_w_recipe': None,  # SLOT_POS_TABLE_W_RECIPE
    'crating_table_wo_recipe': SLOT_POS_TABLE_WO_RECIPE,
}

class WorkerPlus(Worker):
    
    def __init__(
        self, 
        env: Union[str, MinecraftWrapper],
        sample_ratio: float = 0.5,
        inventory_slot_range: Tuple[int, int] = (0, 36), 
        **kwargs, 
    )-> None:
        super().__init__(env, sample_ratio, inventory_slot_range, **kwargs)

    ''' find empty slot (there is no object) ids of inventory, num from 0-35, 
        if is the bottom bar, add it to empty_ids_bar number from 0-8 '''
    def find_empty_box(self, inventory):
        empty_ids, empty_ids_bar = [], []
        for k, v in inventory.items():
            if v['type'] == 'none':
                empty_ids.append(k)
                if k < 9:
                    empty_ids_bar.append(k)
        return empty_ids, empty_ids_bar

    ''' equip item such as wooden_pickaxe '''
    def equip_item(self, target_item):

        try:
            # check target_item is equippable before equip_item()
            # check is gui open and open gui
            if not self.info['isGuiOpen']:
                self.open_inventory_wo_recipe()
            # check and get target_item's pos in inventory
            pos_id = self.find_in_inventory(self.get_labels(), target_item)
            self._assert(pos_id, f'Can not find {target_item} in inventory')

            ''' whether pickaxe in inventory or bar, move anyhow '''
            _, empty_bar = self.find_empty_box(self.info['inventory'])
            if len(empty_bar) == 0:
                result = [f'inventory_{i}' for i in range(9)]
                result = random.choice(result)
            else:
                result = 'inventory_{}'.format(random.choice(empty_bar))
                
            pos_bottom = result
            slot_pos = SLOT_POS_MAPPING[self.current_gui_type]
            self.pull_item(slot_pos, pos_id, pos_bottom, 1)

            # if bottom is fully, the item will be substitued with target_item

            self._call_func('inventory')  # close inventory
            hotbar_id = int(pos_bottom.split('_')[-1])
            self._call_func('hotbar.{}'.format(hotbar_id + 1))
            self._attack_continue()
    
        except AssertionError as e:
            return False, str(e) 
        
        return True, None

if __name__ == '__main__':
    worker = WorkerPlus('test')
    
    # done, info = worker.crafting('crafting_table', 1)
    # print(done,info)

    # done, info = worker.crafting('wooden_pickaxe', 1)
    # print(done,info)

    # done, info = worker.crafting('wooden_pickaxe', 1)
    # print(done,info)

    worker.equip_item('wooden_pickaxe')
    write_video('equip_full_bottom_bar.mp4', worker.outframes)

    # write_video('crafting.mp4', worker.outframes)