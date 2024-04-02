from jarvis.stark_tech.env_interface import MinecraftWrapper
from jarvis.assembly.base import tag_items, skills

import random
import json
import os

from datetime import datetime
from functools import partial
from rich import print as rprint

# DONE: Fix tag bugs
def monitor_function(obj:dict, info:dict):
    if 'inventory' in info.keys():
        # print(info.keys())
        for item, num in obj.items():
            item_inv_num = 0
            if 'minecraft:'+item in tag_items.keys():
                item_inv_num_dict = {}
                for i in range(36):
                    if 'minecraft:'+info['inventory'][i]['type'] in tag_items['minecraft:'+item]:
                        # item_inv_num += info['inventory'][i]['quantity']
                        if 'minecraft:'+info['inventory'][i]['type'] in item_inv_num_dict.keys():
                            item_inv_num_dict['minecraft:'+info['inventory'][i]['type']] += info['inventory'][i]['quantity']
                        else:
                            item_inv_num_dict['minecraft:'+info['inventory'][i]['type']] = info['inventory'][i]['quantity']
                for k, v in item_inv_num_dict.items():
                    if v > item_inv_num:
                        item_inv_num = v
            else:
                for i in range(36):
                    if item == info['inventory'][i]['type']:
                        item_inv_num += info['inventory'][i]['quantity']
            if item_inv_num < num:
                return False, {"success": False, "reason": f"{item} is not enough"}
        return True, {"success": True}
    else:
        return False, {"success": False, "reason": f"{item} is not enough"}
    
def get_inventory_from_info(info:dict):
    inventory = {}
    for i in range(36):
        if info['inventory'][i]['type'] == 'none':
            continue
        
        if info['inventory'][i]['type'] in inventory.keys():
            inventory[info['inventory'][i]['type']] += info['inventory'][i]['quantity']
        else:
            inventory[info['inventory'][i]['type']] = info['inventory'][i]['quantity']
    return inventory

def print_info(infos:dict):
    if len(infos) == 0:
        return
    print(f"step: {len(infos)}, inventory: {get_inventory_from_info(infos[-1])}")


