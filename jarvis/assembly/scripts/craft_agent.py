import random
import math
import os 
import re
import json
import numpy as np
from typing import (
    Sequence, List, Mapping, Dict, 
    Callable, Any, Tuple, Optional, Union
)
from jarvis.stark_tech.env_interface import MinecraftWrapper
from jarvis.utils import write_video

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

class Worker(object):
    
    def __init__(
        self, 
        env: Union[MinecraftWrapper, str],
        sample_ratio: float = 0.5,
        inventory_slot_range: Tuple[int, int] = (0, 36), 
        **kwargs, 
    )-> None:
        # print("Initializing worker...")
        self.sample_ratio = sample_ratio
        self.inventory_slot_range = inventory_slot_range
        self.outframes, self.outactions, self.outinfos = [], [], []

        if isinstance(env, str):
            self.env = MinecraftWrapper(env)
            self.obs, self.info = self.env.reset()
        else:
            self.env = env
        self.reset(fake_reset=False)

    def reset(self, fake_reset=True):
        if not fake_reset:
            self.current_gui_type = None
            self.crafting_slotpos = 'none'
            self.resource_record = {f'resource_{x}': {'type': 'none', 'quantity': 0} for x in range(9)}
            self._null_action(1)
        else:
            self.outframes, self.outactions, self.outinfos = [], [], []

    def _assert(self, condition, message=None):
        if not condition:
            if self.info['isGuiOpen']:
                self._call_func('inventory')   
            self.current_gui_type = None
            self.crafting_slotpos = 'none'
            self.resource_record = {f'resource_{x}': {'type': 'none', 'quantity': 0} for x in range(9)}      
                 
            raise AssertionError(message)
        
    def _step(self, action):
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.outframes.append(self.info['pov'].astype(np.uint8))
        self.outinfos.append(self.info)
        self.outactions.append(action)
        self.info['resource'] = self.resource_record
        return self.obs, reward, terminated, truncated, self.info

    # open inventory    
    def open_inventory_wo_recipe(self):
        self._call_func('inventory')
        self.cursor = [WIDTH // 2, HEIGHT // 2]
        # update slot pos
        self.current_gui_type = 'inventory_wo_recipe'
        self.crafting_slotpos = SLOT_POS_INVENTORY_WO_RECIPE

    # before opening crafting_table
    def pre_open_tabel(self, attack_num=20):
        action = self.env.noop_action()
        self.obs, _, _, _, self.info = self._step(action)
        height_1 = self.info['location_stats']['ypos']

        action['jump'] = 1
        self.obs, _, _, _, self.info = self._step(action)
        height_2 = self.info['location_stats']['ypos']

        self._null_action(1)
        if height_2 - height_1 > 0.419:
            pass
        else:
            '''euip pickaxe'''
            self.obs, _, _, _, self.info = self._step(action)
            height = self.info['location_stats']['ypos']
            if height < 50:
                # find pickaxe
                labels = self.get_labels()
                inventory_id_diamond = self.find_in_inventory(labels, 'diamond_pickaxe', 'item')
                inventory_id_iron = self.find_in_inventory(labels, 'iron_pickaxe', 'item')
                inventory_id_stone = self.find_in_inventory(labels, 'stone_pickaxe', 'item')
                inventory_id_wooden = self.find_in_inventory(labels, 'wooden_pickaxe', 'item')

                if inventory_id_wooden:
                    inventory_id = inventory_id_wooden
                if inventory_id_stone:
                    inventory_id = inventory_id_stone
                if inventory_id_iron:
                    inventory_id = inventory_id_iron
                if inventory_id_diamond:
                    inventory_id = inventory_id_diamond
                
                if inventory_id != 'inventory_0':
                    self.open_inventory_wo_recipe()
                    
                    '''clear inventory 0'''
                    labels=self.get_labels()
                    if labels['inventory_0']['type'] != 'none':
                        for i in range(9):
                            del labels["resource_"+str(i)]
                        inventory_id_none = self.find_in_inventory(labels, 'none')
                        self.pull_item_all(self.crafting_slotpos, 'inventory_0', inventory_id_none)
                    
                    self.pull_item(self.crafting_slotpos, inventory_id, 'inventory_0', 1)
                    self._call_func('inventory')
                    self.current_gui_type = None
                    self.crafting_slotpos = 'none'
                    self._call_func('hotbar.1')

            action = self.env.noop_action()
            for i in range(2):
                action['camera'] = np.array([-88, 0])
                self.obs, _, _, _, self.info = self._step(action)

            action['camera'] = np.array([22, 0])
            self.obs, _, _, _, self.info = self._step(action)   

            for i in range(5):
                action['camera'] = np.array([0, 60])
                self.obs, _, _, _, self.info = self._step(action)
                self._attack_continue(attack_num)

    # open crafting_table
    def open_crating_table_wo_recipe(self):
        self.pre_open_tabel()
        self._null_action(1)
        if self.info['isGuiOpen']:
            self._call_func('inventory')      
        self.open_inventory_wo_recipe()
        labels=self.get_labels()
        inventory_id = self.find_in_inventory(labels, 'crafting_table')
        self._assert(inventory_id, f"no crafting_table")

        if inventory_id != 'inventory_0':
            labels=self.get_labels()
            if labels['inventory_0']['type'] != 'none':
                for i in range(9):
                    del labels["resource_"+str(i)]
                inventory_id_none = self.find_in_inventory(labels, 'none')
                self.pull_item_all(self.crafting_slotpos, 'inventory_0', inventory_id_none)
            self.pull_item(self.crafting_slotpos, inventory_id, 'inventory_0', 1)
        
        self._call_func('inventory')
        self.current_gui_type = None
        self.crafting_slotpos = 'none'

        self._call_func('hotbar.1')

        self._place_down()
        for i in range(5):
            self._call_func('use')
            if self.info['isGuiOpen']:
                break
        self.cursor = [WIDTH // 2, HEIGHT // 2]
        self.current_gui_type = 'crating_table_wo_recipe'
        self.crafting_slotpos = SLOT_POS_TABLE_WO_RECIPE

    # action wrapper
    def _call_func(self, func_name: str):
        action = self.env.noop_action()
        action[func_name] = 1
        for i in range(1):
            self.obs, _, _, _, self.info = self._step(action)
        action[func_name] = 0
        for i in range(5):
            self.obs, _, _, _, self.info = self._step(action)

    def _look_down(self):
        action = self.env.noop_action()
        self._null_action()
        for i in range(2):
            action['camera'] = np.array([88, 0])
            self.obs, _, _, _, self.info = self._step(action)
    
    def _jump(self):
        self._call_func('jump')
    
    def _place_down(self):
        self._look_down()
        self._jump()
        self._call_func('use')

    def _use_item(self):
        self._call_func('use')
    
    def _select_item(self):
        self._call_func('attack')

    def _null_action(self, times=1):
        action = self.env.noop_action()
        for i in range(times):
            self.obs, _, _, _, self.info = self._step(action)

    # continue attack (retuen crafting table)
    def _attack_continue(self, times=1):
        action = self.env.noop_action()
        action['attack'] = 1
        for i in range(times):
            self.obs, _, _, _, self.info = self._step(action)
        
    # move 
    def move_to_pos(self, x: float, y: float, speed: float = 20):
        camera_x = x - self.cursor[0]
        camera_y = y - self.cursor[1]
        distance =max(abs(camera_x), abs(camera_y))
        num_steps= int(random.uniform(5, 10) * math.sqrt(distance) / speed)
        if num_steps < 1:
            num_steps = 1
        for _ in range(num_steps):
            d1 = (camera_x / num_steps) 
            d2 = (camera_y / num_steps) 
            self.move_once(d1, d2)

    def random_move_or_stay(self):
        if np.random.uniform(0, 1) > 0.5:
            num_random = random.randint(2, 4)
            if random.uniform(0, 1) > 0.25:
                for i in range(num_random):
                    self.move_once(0, 0)
            else:
                for i in range(num_random):
                    d1 =  random.uniform(-5, 5)
                    d2 =  random.uniform(-5, 5)
                    self.move_once(d1, d2)
        else:
            pass

    def move_once(self, x: float, y: float):
        action = self.env.noop_action() 
        action['camera'] = np.array([y * CAMERA_SCALER, x * CAMERA_SCALER])
        self.obs, _, _, _, self.info = self._step(action) 
        self.cursor[0] += x
        self.cursor[1] += y
    
    def move_to_slot(self, SLOT_POS: Dict, slot: str):
        self._assert(slot in SLOT_POS, f'Error: slot: {slot}')
        x, y = SLOT_POS[slot]
        self.move_to_pos(x, y)
    # pull
    # select item_from, select item_to
    def pull_item_all(self, 
        SLOT_POS: Dict, 
        item_from: str, 
        item_to: str
    ) -> None:
        self.move_to_slot(SLOT_POS, item_from)
        self._null_action(1)
        self._select_item()
        self._null_action(1)
        self.move_to_slot(SLOT_POS, item_to) 
        self._null_action(1)
        self._select_item()
        self._null_action(1)
        self.random_move_or_stay()
    # select item_from, use n item_to    
    def pull_item(self, 
        SLOT_POS: Dict, 
        item_from: str, 
        item_to: str,
        target_number: int
    ) -> None:
        if 'resource' in item_to:
            item = self.info['inventory'][int(item_from.split('_')[-1])]
            self.resource_record[item_to] = item
        self.move_to_slot(SLOT_POS, item_from)
        self._null_action(1)
        self._select_item()
        self.move_to_slot(SLOT_POS, item_to)
        self._null_action(1)
        for i in range(target_number):
            self._use_item()
            self._null_action(1)
        self.random_move_or_stay()
    # use n item_to 
    def pull_item_continue(self, 
        SLOT_POS: Dict, 
        item_to: str,
        item: str,
        target_number: int 
    ) -> None:
        if 'resource' in item_to:
            self.resource_record[item_to] = item
        self.move_to_slot(SLOT_POS, item_to)
        self._null_action(1)
        for i in range(target_number):
            self._use_item()
            self._null_action(1)
        self.random_move_or_stay()
    # select item_to 
    def pull_item_return(self, 
        SLOT_POS: Dict, 
        item_to: str,
    ) -> None: 
        self.move_to_slot(SLOT_POS, item_to)
        self._null_action(1)
        self._select_item()
        self._null_action(1)
        self.random_move_or_stay()
    # use n item_frwm, select item_to
    def pull_item_result(self, 
        SLOT_POS: Dict, 
        item_from: str,
        item_to: str,
        target_number: int
    ) -> None: 
        self.move_to_slot(SLOT_POS, item_from)
        for i in range(target_number):
            self._use_item()
            self._null_action(1)
        self.move_to_slot(SLOT_POS, item_to)
        self._select_item()
        self._null_action(1)
        self.random_move_or_stay()

    # get all labels
    def get_labels(self, noop=True):
        if noop:
            self._null_action(1)
        result = {}
        # generate resource recording item labels
        for i in range(9):
            slot = f'resource_{i}'
            item = self.resource_record[slot]
            result[slot] = item
        
        # generate inventory item labels
        for slot, item in self.info['inventory'].items():
            result[f'inventory_{slot}'] = item
        
        return result
    
    # crafting
    def crafting(self, target: str, target_num: int=1):
        try:

            # is item/tag
            is_tag = False
            cur_path = os.path.abspath(os.path.dirname(__file__))
            root_path = cur_path[:cur_path.find('jarvis')]
            relative_path = os.path.join("jarvis/assets/tag_items.json")
            tag_json_path = os.path.join(root_path, relative_path)
            with open(tag_json_path) as file:
                tag_info = json.load(file)
            for key in tag_info:
                if key[10:] == target:
                    is_tag = True

            # open recipe one by one: only shapeless crafting like oak_planks        
            if is_tag:
                enough_material = False
                enough_material_target = 'none'
                item_list = tag_info['minecraft:'+target]

                for item in item_list:
                    subtarget = item[10:]
                
                    relative_path = os.path.join("jarvis/assets/recipes", subtarget + '.json')
                    recipe_json_path = os.path.join(root_path, relative_path)
                    with open(recipe_json_path) as file:
                        recipe_info = json.load(file)
                    need_table = self.crafting_type(recipe_info)

                    # find materials(shapeless) like oak_planks
                    ingredients = recipe_info.get('ingredients')
                    random.shuffle(ingredients)
                    items = dict()
                    items_type = dict()

                    # clculate the amount needed and store <item, quantity> in items
                    for i in range(len(ingredients)):
                        if ingredients[i].get('item'):
                            item = ingredients[i].get('item')[10:]
                            item_type = 'item'
                        else:
                            item = ingredients[i].get('tag')[10:]
                            item_type = 'tag'
                        items_type[item] = item_type
                        if items.get(item):
                            items[item] += 1
                        else:
                            items[item] = 1

                    if recipe_info.get('result').get('count'):
                        iter_num = math.ceil(target_num / int(recipe_info.get('result').get('count')))
                    else:
                        iter_num = target_num

                    enough_material_subtarget = True
                    for item, num_need in items.items():
                        labels = self.get_labels()
                        inventory_id = self.find_in_inventory(labels, item, items_type[item])
                        if not inventory_id:
                            enough_material_subtarget = False
                            break
                        inventory_num = labels.get(inventory_id).get('quantity')
                        if num_need * iter_num > inventory_num:
                            enough_material_subtarget = False
                            break
                    if enough_material_subtarget:
                        enough_material = True
                        enough_material_target = subtarget

                if enough_material:
                    target = enough_material_target
                else:
                    self._assert(0, f"not enough materials for {target}")

            # if inventory is open by accident, close inventory
            self._null_action(1)
            if self.info['isGuiOpen']:
                self._call_func('inventory')           
            cur_path = os.path.abspath(os.path.dirname(__file__))
            root_path = cur_path[:cur_path.find('jarvis')]
            relative_path = os.path.join("jarvis/assets/recipes", target + '.json')
            recipe_json_path = os.path.join(root_path, relative_path)
            with open(recipe_json_path) as file:
                recipe_info = json.load(file)
            need_table = self.crafting_type(recipe_info)

            if need_table:
                self.open_crating_table_wo_recipe()
            else:
                self.open_inventory_wo_recipe()               
            
            # crafting
            if recipe_info.get('result').get('count'):
                iter_num = math.ceil(target_num / int(recipe_info.get('result').get('count')))
            else:
                iter_num = target_num

            self.crafting_once(target, iter_num, recipe_info, target_num)

            # close inventory
            self._call_func('inventory')
            if need_table:
                self.return_crafting_table()
            self.current_gui_type = None
            self.crafting_slotpos = 'none'  

        except AssertionError as e:
            return False, str(e) 
        
        return True, None

    # return crafting table
    def return_crafting_table(self):
        self._look_down()
        labels = self.get_labels()
        table_info = self.find_in_inventory(labels, 'crafting_table')
        tabel_exist = 0
        if table_info:
            tabel_exist = 1
            tabel_num = labels.get(table_info).get('quantity')

        done = 0
        for i in range(4):
            for i in range(10):
                self._attack_continue(8)
                labels = self.get_labels(noop=False)
                if tabel_exist:
                    table_info = self.find_in_inventory(labels, 'crafting_table')
                    tabel_num_2 = labels.get(table_info).get('quantity')
                    if tabel_num_2 != tabel_num:
                        done = 1
                        break
                else:
                    table_info = self.find_in_inventory(labels, 'crafting_table')
                    if table_info:
                        done = 1
                        break
            self._call_func('forward')
        self._assert(done, f'return crafting_table unsuccessfully')    

    # judge crafting_table / inventory
    def crafting_type(self, target_data: Dict):
        if 'pattern' in target_data:
            pattern = target_data.get('pattern')
            col_len = len(pattern)
            row_len = len(pattern[0])
            if col_len <= 2 and row_len <= 2:
                return False
            else:
                return True
        else:
            ingredients = target_data.get('ingredients')
            item_num = len(ingredients)
            if item_num <= 4:
                return False
            else:
                return True
    
    # search item in agent's inventory 
    def find_in_inventory(self, labels: Dict, item: str, item_type: str='item', path=None):

        if path == None:
            path = []
        for key, value in labels.items():
            current_path = path + [key]
            if item_type == "item":
                if re.match(item, str(value)):
                    return current_path
                elif isinstance(value, dict):
                    result = self.find_in_inventory(value, item, item_type, current_path)
                    if result is not None:
                        return result[0]
            elif item_type == "tag":
                # tag info
                cur_path = os.path.abspath(os.path.dirname(__file__))
                root_path = cur_path[:cur_path.find('jarvis')]
                relative_path = os.path.join("jarvis/assets/tag_items.json")
                tag_json_path = os.path.join(root_path, relative_path)
                with open(tag_json_path) as file:
                    tag_info = json.load(file)

                item_list = tag_info['minecraft:'+item]
                for i in range(len(item_list)):
                    if re.match(item_list[i][10:], str(value)):
                        return current_path
                    elif isinstance(value, dict):
                        result = self.find_in_inventory(value, item, item_type, current_path)
                        if result is not None:
                            return result[0]
        return None

    # crafting once 
    def crafting_once(self, target: str, iter_num: int, recipe_info: Dict, target_num:int):

        # shaped crafting
        if "pattern" in recipe_info:
            self.crafting_shaped(target, iter_num, recipe_info)
        # shapless crafting
        else:
            self.crafting_shapeless(target, iter_num, recipe_info)

        # get result
        # Do not put the result in resource
        labels = self.get_labels()
        for i in range(9):
            del labels["resource_"+str(i)]
        
        result_inventory_id_1 = self.find_in_inventory(labels, target)

        if result_inventory_id_1:
            item_num = labels.get(result_inventory_id_1).get('quantity')
            if item_num + target_num < 60:
                self.pull_item_result(self.crafting_slotpos, 'result_0', result_inventory_id_1, iter_num)
                labels_after = self.get_labels()
                item_num_after = labels_after.get(result_inventory_id_1).get('quantity')

                if item_num == item_num_after:
                    result_inventory_id_2 = self.find_in_inventory(labels, 'none')
                    self._assert(result_inventory_id_2, f"no space to place result")
                    self.pull_item_return(self.crafting_slotpos, result_inventory_id_2)
                    self._assert(self.get_labels().get(result_inventory_id_2).get('type') == target, f"fail for unkown reason")
            else:
                result_inventory_id_2 = self.find_in_inventory(labels, 'none')
                self._assert(result_inventory_id_2, f"no space to place result")
                self.pull_item_result(self.crafting_slotpos, 'result_0', result_inventory_id_2, iter_num)
                self._assert(self.get_labels().get(result_inventory_id_2).get('type') == target, f"fail for unkown reason")
        else:
            result_inventory_id_2 = self.find_in_inventory(labels, 'none')
            self._assert(result_inventory_id_2, f"no space to place result")
            self.pull_item_result(self.crafting_slotpos, 'result_0', result_inventory_id_2, iter_num)
            self._assert(self.get_labels().get(result_inventory_id_2).get('type') == target, f"fail for unkown reason")

        # clear resource          
        self.resource_record =  {f'resource_{x}': {'type': 'none', 'quantity': 0} for x in range(9)}

    # shaped crafting
    def crafting_shaped(self, target:str, iter_num:int, recipe_info: Dict):
        slot_pos = self.crafting_slotpos
        labels = self.get_labels()
        pattern = recipe_info.get('pattern')
        items = recipe_info.get('key')
        items = random_dic(items)
        # place each item in order
        for k, v in items.items():
            signal = k
            if v.get('item'):
                item = v.get('item')[10:]
                item_type= 'item'
            else:
                item = v.get('tag')[10:]
                item_type= 'tag'
            labels = self.get_labels()
            inventory_id = self.find_in_inventory(labels, item, item_type)
            self._assert(inventory_id, f"not enough {item}")
            inventory_num = labels.get(inventory_id).get('quantity')

            # clculate the amount needed
            num_need = 0
            for i in range(len(pattern)):
                for j in range(len(pattern[i])):
                    if pattern[i][j] == signal:
                        num_need += 1
            num_need = num_need * iter_num
            self._assert(num_need <= inventory_num, f"not enough {item}")

            # place
            resource_idx = 0
            first_pull = 1
            if 'table' in self.current_gui_type:
                type = 3
            else:
                type = 2
            for i in range(len(pattern)):
                resource_idx = i * type
                for j in range(len(pattern[i])):
                    if pattern[i][j] == signal:
                        if first_pull:
                            self.pull_item(slot_pos, inventory_id, 'resource_' + str(resource_idx), iter_num)
                            first_pull = 0
                        else:
                            self.pull_item_continue(slot_pos, 'resource_' + str(resource_idx), item, iter_num)
                    resource_idx += 1

            # return the remaining items
            if num_need < inventory_num:
                self.pull_item_return(slot_pos, inventory_id)
    
    # shapeless crafting
    def crafting_shapeless(self, target:str, iter_num:int, recipe_info: Dict):   
        slot_pos = self.crafting_slotpos 
        labels = self.get_labels()
        ingredients = recipe_info.get('ingredients')
        random.shuffle(ingredients)
        items = dict()
        items_type = dict()

        # clculate the amount needed and store <item, quantity> in items
        for i in range(len(ingredients)):
            if ingredients[i].get('item'):
                item = ingredients[i].get('item')[10:]
                item_type = 'item'
            else:
                item = ingredients[i].get('tag')[10:]
                item_type = 'tag'
            items_type[item] = item_type
            if items.get(item):
                items[item] += 1
            else:
                items[item] = 1
        
        # place each item in order
        resource_idx = 0
        for item, num_need in items.items():
            labels = self.get_labels()
            inventory_id = self.find_in_inventory(labels, item, items_type[item])
            self._assert(inventory_id, f"not enough {item}")
            inventory_num = labels.get(inventory_id).get('quantity')
            self._assert(num_need * iter_num <= inventory_num, f"not enough {item}")

            # place 
            num_need -= 1
            self.pull_item(slot_pos, inventory_id, 'resource_' + str(resource_idx), iter_num)

            resource_idx += 1
            if num_need != 0:
                for i in range(num_need):
                    self.pull_item_continue(slot_pos, 'resource_' + str(resource_idx), item, iter_num)
                    resource_idx += 1
            
            # return the remaining items
            num_need = (num_need + 1) * iter_num
            if num_need < inventory_num:
                self.pull_item_return(slot_pos, inventory_id)

if __name__ == '__main__':

    # cur_path = os.path.abspath(os.path.dirname(__file__))
    # root_path = cur_path[:cur_path.find('jarvis')]
    # relative_path = os.path.join("jarvis/assets/tag_items.json")
    # tag_json_path = os.path.join(root_path, relative_path)
    # with open(tag_json_path) as file:
    #     tag_info = json.load(file)
    # for key in tag_info:
    #     print(key[10:])
  
    worker = Worker('test')
    done, info = worker.crafting('wooden_pickaxe', 1)
    print(done, info)
    write_video('crafting.mp4', worker.outframes)