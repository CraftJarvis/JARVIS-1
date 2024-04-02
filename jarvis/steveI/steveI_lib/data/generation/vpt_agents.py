import itertools

model_2x_path = 'data/weights/vpt/2x.model'
model_3x_path = 'data/weights/vpt/3x.model'

rl_from_foundation_2x = {
    'name': 'rl_from_foundation_2x',
    'in_model': model_2x_path,
    'in_weights': 'data/weights/vpt/rl-from-foundation-2x.weights'
}
VPT_AGENT_PAIRS = [(rl_from_foundation_2x, rl_from_foundation_2x)]

# Download the following weights from download_weights.sh and uncomment the following lines
# to use these models for dataset generation (in addition to rl_from_foundation_2x above).
# foundation_model_2x = {
#     'name': 'foundation_model_2x',
#     'in_model': model_2x_path,
#     'in_weights': 'data/weights/vpt/foundation-model-2x.weights'
# }
# bc_early_game_2x = {
#     'name': 'bc_early_game_2x',
#     'in_model': model_2x_path,
#     'in_weights': 'data/weights/vpt/bc-early-game-2x.weights'
# }
# bc_house_3x = {
#     'name': 'bc_house_3x',
#     'in_model': model_3x_path,
#     'in_weights': 'data/weights/vpt/bc-house-3x.weights'
# }
# rl_from_house_2x = {
#     'name': 'rl_from_house_2x',
#     'in_model': model_2x_path,
#     'in_weights': 'data/weights/vpt/rl-from-house-2x.weights'
# }
# VPT_AGENTS_ALL = [
#     foundation_model_2x,
#     bc_early_game_2x,
#     bc_house_3x,
#     rl_from_foundation_2x,
#     rl_from_house_2x,
# ]
# VPT_AGENT_PAIRS = list(itertools.combinations(VPT_AGENTS_ALL, 2))
# VPT_AGENT_PAIRS = [pair for pair in VPT_AGENT_PAIRS
#                    if not ('rl_from' in pair[0]['name'] and 'rl_from' in pair[1]['name'])]
