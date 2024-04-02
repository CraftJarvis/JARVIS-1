import numpy as np


class ProgrammaticEvaluator:
    """Class for keeping track of: travel distance, seed count, dirt count, and log count."""
    def __init__(self, initial_info) -> None:
        # Store the max inventory counts for each block type and travel distance (these are lower bound measures)
        self.prog_values = {}
        self.initial_info = initial_info

    def update(self, info):
        """Update the programmatic evaluation metrics."""
        self.prog_values = compute_programmatic_rewards(self.initial_info, info, self.prog_values)

    def print_results(self):
        """Print the results of the programmatic evaluation."""
        print("Programmatic Evaluation Results:")
        for prog_task in self.prog_values.keys():
            print(f"{prog_task}: {self.prog_values[prog_task]}")
        print()


def update_max_inventory_counts(current_inventory, inventory_counts, block_type, block_key):
    """ Update the inventory counts for the block type

    Args:
        current_inventory (dict): Dictionary containing the agent's current inventory counts for each block type
        inventory_counts (dict): Dictionary containing the max inventory counts for each block type
        block_type (str): The string filter for the block type to update the inventory count for
        block_key (str): The key for the block type in the inventory dictionary
    """
    block_count = 0
    for x in current_inventory.values():
        if block_type in x['type']:
            block_count += x['quantity']

    # Update the dirt count in inventory_counts
    if block_count > inventory_counts.get(block_key, 0):
        print(f"Updating count for {block_key} from {inventory_counts.get(block_key, 0)} to {block_count}")
        inventory_counts[block_key] = block_count

    return inventory_counts


def compute_programmatic_rewards(info_init, info_current, prog_values):
    """Compute the inventory count across various types of blocks."""
    current_inventory = info_current['inventory']

    block_filter_types = ["_log", "dirt", "seed"]
    block_names = ["log", "dirt", "seed"]

    # Update the inventory counts for the block types
    for block_name in block_names:
        if block_name not in prog_values:
            prog_values[block_name] = 0

    for block_filter_type, block_name in zip(block_filter_types, block_names):
        prog_values = update_max_inventory_counts(current_inventory, prog_values, block_filter_type,
                                                  block_name)

    # Keep track of the travel distance. The travel distance is the Euclidean distance from the spawn point to the
    # farthest point the agent reached during the episode on the horizontal (x-z) plane
    curr_x, curr_z = info_current['location_stats']['xpos'], info_current['location_stats']['zpos']

    # Compute the Euclidean distance from the spawn point to the current location
    dist = np.sqrt(
        (curr_x - info_init['location_stats']['xpos']) ** 2 + (curr_z - info_init['location_stats']['zpos']) ** 2)

    if dist > prog_values.get("travel_dist", 0):
        prog_values["travel_dist"] = dist

    return prog_values
