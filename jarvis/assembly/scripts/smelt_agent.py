
import os 
import json
from typing import Sequence, List, Mapping, Dict, Callable, Any, Tuple, Optional
from jarvis.assembly.scripts.craft_agent import *


CAMERA_SCALER = 360.0 / 2400.0
WIDTH, HEIGHT = 640, 360
KEY_POS_FURNACE_WO_RECIPE = {
    'resource_slot': {
        'left-top': (287, 113), 
        'right-bottom': (303, 164), 
        'row': 2, 
        'col': 1,
        'prefix': 'resource', 
        'start_id': 0, 
    },
    'result_slot': {
        'left-top': (345, 127), 
        'right-bottom': (368, 152),
        'row': 1, 
        'col': 1,
        'prefix': 'result', 
        'start_id': 0, 
    },
    'hotbar_slot': {
        'left-top': (242, 236), 
        'right-bottom': (401, 256),
        'row': 1, 
        'col': 9, 
        'prefix': 'inventory', 
        'start_id': 0, 
    }, 
    'inventory_slot': {
        'left-top': (242, 178), 
        'right-bottom': (401, 234), 
        'row': 3, 
        'col': 9,
        'prefix': 'inventory',
        'start_id': 9,
    }, 
    'recipe_slot': {
        'left-top': (254, 132),
        'right-bottom': (272, 147),
        'row': 1, 
        'col': 1,
        'prefix': 'recipe', 
        'start_id': 0,
    }
}
SLOT_POS_FURNACE_WO_RECIPE = COMPUTE_SLOT_POS(KEY_POS_FURNACE_WO_RECIPE)

class Worker_smelting(Worker):
    def reset(self, fake_reset=True):
        if not fake_reset:
            self.current_gui_type = None
            self.resource_record = {f'resource_{x}': {'type': 'none', 'quantity': 0} for x in range(2)}
            self.crafting_slotpos = 'none'
            self._null_action(1)
        else:
            self.outframes, self.outactions, self.outinfos = [], [], []

    def _assert(self, condition, message=None):
        if not condition:
            if self.info['isGuiOpen']: 
                self._call_func('inventory')
            self.current_gui_type = None
            self.crafting_slotpos = 'none'   
            self.resource_record = {f'resource_{x}': {'type': 'none', 'quantity': 0} for x in range(2)}     
            write_video('error.mp4', self.outframes)
            raise AssertionError(message)     
            
    def open_furnace_wo_recipe(self):
        self.pre_open_tabel(attack_num=40)
        self._null_action(1)
        if self.info['isGuiOpen']:
            self._call_func('inventory')      
        self.open_inventory_wo_recipe()
        labels=self.get_labels()
        inventory_id = self.find_in_inventory(labels, 'furnace')
        self._assert(inventory_id, f'furnace')

        if inventory_id != 'inventory_0':
            labels=self.get_labels()
            if labels['inventory_0']['type'] != 'none':
                for i in range(2):
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
        self.current_gui_type = 'furnace_wo_recipe'
        self.crafting_slotpos = SLOT_POS_FURNACE_WO_RECIPE

    def get_labels(self, noop=True):
        if noop:
            self._null_action(1)
        result = {}
        # generate resource recording item labels
        for i in range(2):
            slot = f'resource_{i}'
            item = self.resource_record[slot]
            result[slot] = item
        
        # generate inventory item labels
        for slot, item in self.info['inventory'].items():
            result[f'inventory_{slot}'] = item
        
        return result

    def smelting(self, target: str, target_num: int=1):
        try:
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
            
            
            self.open_furnace_wo_recipe()

            # find coals
            fuels_type = 'none'
            labels = self.get_labels()
            inventory_id = self.find_in_inventory(labels, 'coals', 'tag')
            # find logs
            if inventory_id:
                fuels_type = 'coals'
            else:
                inventory_id_logs = self.find_in_inventory(labels, 'logs', 'tag')
                inventory_id_planks = self.find_in_inventory(labels, 'planks', 'tag')
                if inventory_id_logs and inventory_id_planks:
                    fuels_type = 'coalstodo'
                else:
                    if inventory_id_planks:
                        fuels_type = 'planks'
                    else:
                        if inventory_id_logs:
                            fuels_type = 'logs'
            
            if fuels_type == 'none':
                self._assert(inventory_id, f"not enough fuels")
            
            if fuels_type == 'coalstodo':

                relative_path_fuels = os.path.join("jarvis/assets/recipes", 'charcoal' + '.json')
                recipe_json_path_fuels = os.path.join(root_path, relative_path_fuels)
                with open(recipe_json_path_fuels) as file:
                    recipe_info_fuels = json.load(file)

                self.smelting_once('charcoal', recipe_info_fuels, target_num=1, fuels='planks')
                fuels_type = 'coals'
        
            self.smelting_once(target, recipe_info, target_num=target_num, fuels=fuels_type)

            # close inventory
            labels = self.get_labels()
            inventory_id = self.find_in_inventory(labels, 'wooden_pickaxe')
            self._assert(inventory_id, f"no wooden_pickaxe to return furnace")
            if inventory_id != 'inventory_0':
                if labels['inventory_0']['type'] != 'none':
                    for i in range(2):
                        del labels["resource_"+str(i)]
                    inventory_id_none = self.find_in_inventory(labels, 'none')
                    self.pull_item_all(self.crafting_slotpos, 'inventory_0', inventory_id_none)
                self.pull_item(self.crafting_slotpos, inventory_id, 'inventory_0', 1)

                self._call_func('inventory')
                self.return_furnace()

                self.current_gui_type = None
                self.crafting_slotpos = 'none'
                self.resource_record = {f'resource_{x}': {'type': 'none', 'quantity': 0} for x in range(2)}   

        except AssertionError as e:
            return False, str(e) 
        return True, None
    
    def return_furnace(self):
        self._look_down()
        labels = self.get_labels()
        table_info = self.find_in_inventory(labels, 'furnace')
        tabel_exist = 0
        if table_info:
            tabel_exist = 1
            tabel_num = labels.get(table_info).get('quantity')
        
        self._call_func('hotbar.1')  

        done = 0
        for i in range(4):
            for i in range(10):
                self._attack_continue(8)
                labels = self.get_labels(noop=False)
                if tabel_exist:
                    table_info = self.find_in_inventory(labels, 'furnace')
                    tabel_num_2 = labels.get(table_info).get('quantity')
                    if tabel_num_2 != tabel_num:
                        done = 1
                        break
                else:
                    table_info = self.find_in_inventory(labels, 'furnace')
                    if table_info:
                        done = 1
                        break
            self._call_func('forward') 
        self._assert(done, f'return furnace unsuccessfully')    

    def smelting_once(self, target: str,  recipe_info: Dict, target_num, fuels):
        slot_pos = self.crafting_slotpos 
        ingredient = recipe_info.get('ingredient')
        cook_time = recipe_info.get('cookingtime')
        items = dict()
        items_type = dict()
        # clculate the amount needed and store <item, quantity> in items
        if ingredient.get('item'):
            item = ingredient.get('item')[10:]
            item_type = 'item'
        else:
            item = ingredient.get('tag')[10:]
            item_type = 'tag'
        items_type[item] = item_type
        if items.get(item):
            items[item] += 1
        else:
            items[item] = 1
                
        # place each item in order
        resource_idx = 0
        first_pull = 1
        for item, _ in items.items():
            labels = self.get_labels()
            for i in range(2):
                del labels["resource_"+str(i)]
            item_type = items_type[item]
            inventory_id = self.find_in_inventory(labels, item, item_type)
            self._assert(inventory_id, f"not enough {item}")
            inventory_num = labels.get(inventory_id).get('quantity')

            # place 
            if first_pull:
                self.pull_item(slot_pos, inventory_id, 'resource_' + str(resource_idx), target_num)
                first_pull = 0
            resource_idx += 1

            # return the remaining items
            if inventory_num > 1:
                self.pull_item_return(slot_pos, inventory_id)
        
        if fuels!='coals':
            for i in range(target_num):
                inventory_id = self.find_in_inventory(labels, fuels, 'tag')
                if not inventory_id:
                    if fuels == 'planks':
                        inventory_id = self.find_in_inventory(labels, 'logs', 'tag')
                self._assert(inventory_id, f"not enough fuels")
                inventory_num = labels.get(inventory_id).get('quantity')
                self.pull_item(slot_pos, inventory_id, 'resource_' + str(resource_idx), 1)
                if inventory_num > 1:
                    self.pull_item_return(slot_pos, inventory_id)
                self._null_action(int(cook_time))
        else:
            pass
            inventory_id = self.find_in_inventory(labels, fuels, 'tag')
            inventory_num = labels.get(inventory_id).get('quantity')
            self.pull_item(slot_pos, inventory_id, 'resource_' + str(resource_idx), 1)
            if inventory_num > 1:
                self.pull_item_return(slot_pos, inventory_id)
            self._null_action(int(cook_time*target_num))

        # get result
        # Do not put the result in resource
        labels = self.get_labels()
        for i in range(2):
            del labels["resource_"+str(i)]

        result_inventory_id_1 = self.find_in_inventory(labels, target)

        if result_inventory_id_1:
            item_num = labels.get(result_inventory_id_1).get('quantity')
            if item_num + target_num < 60:
                self.pull_item_result(self.crafting_slotpos, 'result_0', result_inventory_id_1, target_num)
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
                self.pull_item_result(self.crafting_slotpos, 'result_0', result_inventory_id_2, target_num)
                self._assert(self.get_labels().get(result_inventory_id_2).get('type') == target, f"fail for unkown reason")
        else:
            result_inventory_id_2 = self.find_in_inventory(labels, 'none')
            self._assert(result_inventory_id_2, f"no space to place result")
            self.pull_item_result(self.crafting_slotpos, 'result_0', result_inventory_id_2, target_num)
            self._assert(self.get_labels().get(result_inventory_id_2).get('type') == target, f"fail for unkown reason")

        # clear resource          
        self.resource_record =  {f'resource_{x}': {'type': 'none', 'quantity': 0} for x in range(2)}

if __name__ == '__main__':
    worker = Worker_smelting('test')
    done, info = worker.smelting('iron_ingot', 8)
    print(done,info)
    write_video('crafting.mp4', worker.outframes)