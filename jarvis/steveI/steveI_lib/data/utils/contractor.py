import requests
import os
import jsonlines
import shutil
import cv2
import numpy as np

from jarvis.steveI.steveI_lib.VPT.agent import resize_image, AGENT_RESOLUTION

CURSOR_FILE = 'steve1/data/generation/assets/mouse_cursor_white_16x16.png'

cursor_image = cv2.imread(CURSOR_FILE, cv2.IMREAD_UNCHANGED)
cursor_image = cursor_image[:16, :16, :]
cursor_alpha = cursor_image[:, :, 3:] / 255.0
cursor_image = cursor_image[:, :, :3]

KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
}

NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

CAMERA_SCALER = 360.0 / 2400.0

MINEREC_ORIGINAL_HEIGHT_PX = 720


def process_frame_mineclip(frame: np.ndarray, height: int = 160, width: int = 256):
    """Processes frame to format that mineclip expects (160x256) and (C, H, W)."""
    assert frame.shape[2] == 3, f'Expected channel dim to be at axis 2, got shape {frame.shape}'

    if frame.shape != (160, 256, 3):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    return np.moveaxis(frame, -1, 0)


index_files = {
    '6.x': 'https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_6xx_Jun_29.json',
    '7.x': 'https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_7xx_Apr_6.json',
    '8.x': 'https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_8xx_Jun_29.json',
    '9.x': 'https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_9xx_Jun_29.json',
    '10.x': 'https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_10xx_Jun_29.json'
}


def get_index(version):
    index = requests.get(index_files[version]).json()
    return index


class ContractorData:

    def __init__(self, version, cache_dir):
        self.index = get_index(version)
        self.version = version
        self.cache_dir = cache_dir

    @property
    def basedir(self):
        return self.index['basedir'][:-1]

    def get_video_url(self, idx):
        relpath = self.index['relpaths'][idx]
        relpath = relpath.replace('.mp4', '')
        path = f'{self.basedir}/{relpath}.mp4'
        return path

    def get_action_url(self, idx):
        relpath = self.index['relpaths'][idx]
        relpath = relpath.replace('.mp4', '')
        path = f'{self.basedir}/{relpath}.jsonl'
        return path

    def download(self, idx):
        """Returns the location of the locally downloaded video and also the
        action object."""
        video_url = self.get_video_url(idx)
        action_url = self.get_action_url(idx)
        cache_dir = os.path.join(self.cache_dir, f'{self.version}_{idx}')

        # Download video
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        try:
            video_path = os.path.join(cache_dir, 'video.mp4')
            if not os.path.exists(video_path):
                print(f'Downloading {video_url} to {video_path}...')
                r = requests.get(video_url)
                with open(video_path, 'wb') as f:
                    f.write(r.content)

            # Download action
            action_path = os.path.join(cache_dir, 'action.jsonl')
            if not os.path.exists(action_path):
                print(f'Downloading {action_url} to {action_path}...')
                r = requests.get(action_url)
                with open(action_path, 'wb') as f:
                    f.write(r.content)

            # Open the action with jsonlines
            with jsonlines.open(action_path) as reader:
                json_data = [action for action in reader]

            print(f'Converting data to frames and actions...')

            frames, frames_mineclip, actions = load_episode(video_path, json_data)
            self.clean_cache(cache_dir)
        except Exception as e:
            print(f'Failed to download {video_url} or {action_url}: {e}')
            self.clean_cache(cache_dir)
            return None, None, None

        return frames, frames_mineclip, actions

    def __len__(self):
        return len(self.index['relpaths'])

    def clean_cache(self, cache_dir):
        """Removes all cached videos and actions."""
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


def composite_images_with_alpha(image1, image2, alpha, x, y):
    """
    Draw image2 over image1 at location x,y, using alpha as the opacity for image2.

    Modifies image1 in-place
    """
    ch = max(0, min(image1.shape[0] - y, image2.shape[0]))
    cw = max(0, min(image1.shape[1] - x, image2.shape[1]))
    if ch == 0 or cw == 0:
        return
    alpha = alpha[:ch, :cw]
    image1[y:y + ch, x:x + cw, :] = (image1[y:y + ch, x:x + cw, :] * (1 - alpha) + image2[:ch, :cw, :] * alpha).astype(np.uint8)


def json_action_to_env_action(json_action):
    """
    Converts a json action into a MineRL action.
    Returns (minerl_action, is_null_action)
    """
    # This might be slow...
    env_action = NOOP_ACTION.copy()
    # As a safeguard, make camera action again so we do not override anything
    env_action["camera"] = np.array([0, 0])

    is_null_action = True
    keyboard_keys = json_action["keyboard"]["keys"]
    for key in keyboard_keys:
        # You can have keys that we do not use, so just skip them
        # NOTE in original training code, ESC was removed and replaced with
        #      "inventory" action if GUI was open.
        #      Not doing it here, as BASALT uses ESC to quit the game.
        if key in KEYBOARD_BUTTON_MAPPING:
            env_action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    mouse = json_action["mouse"]
    camera_action = env_action["camera"]
    camera_action[0] = mouse["dy"] * CAMERA_SCALER
    camera_action[1] = mouse["dx"] * CAMERA_SCALER

    if mouse["dx"] != 0 or mouse["dy"] != 0:
        is_null_action = False
    else:
        if abs(camera_action[0]) > 180:
            camera_action[0] = 0
        if abs(camera_action[1]) > 180:
            camera_action[1] = 0

    mouse_buttons = mouse["buttons"]
    if 0 in mouse_buttons:
        env_action["attack"] = 1
        is_null_action = False
    if 1 in mouse_buttons:
        env_action["use"] = 1
        is_null_action = False
    if 2 in mouse_buttons:
        env_action["pickItem"] = 1
        is_null_action = False

    return env_action, is_null_action


def load_episode(video_path, json_data):
    video = cv2.VideoCapture(video_path)

    attack_is_stuck = False
    last_hotbar = 0

    frames, frames_mineclip, actions = [], [], []

    for i in range(len(json_data)):
        step_data = json_data[i]
        if i == 0:
            # Check if attack will be stuck down
            if step_data['mouse']['newButtons'] == [0]:
                attack_is_stuck = True
        elif attack_is_stuck:
            if 0 in step_data['mouse']['newButtons']:
                attack_is_stuck = False
        # If still stuck, remove the action
        if attack_is_stuck:
            step_data['mouse']['buttons'] = [button for button in step_data['mouse']['buttons'] if button != 0]

        action, is_null_action = json_action_to_env_action(step_data)

        # Update hotbar selection
        current_hotbar = step_data['hotbar']
        if current_hotbar != last_hotbar:
            action['hotbar.{}'.format(current_hotbar + 1)] = 1
        last_hotbar = current_hotbar

        # Read frame even if this is null so we progress forward
        ret, frame = video.read()
        if ret:
            # Skip null actions as done in the VPT paper
            # NOTE: in VPT paper, this was checked _after_ transforming into agent's action-space.
            #       We do this here as well to reduce amount of data sent over.
            if is_null_action:
                continue
            if step_data["isGuiOpen"]:
                camera_scaling_factor = frame.shape[0] / MINEREC_ORIGINAL_HEIGHT_PX
                cursor_x = int(step_data["mouse"]["x"] * camera_scaling_factor)
                cursor_y = int(step_data["mouse"]["y"] * camera_scaling_factor)
                composite_images_with_alpha(frame, cursor_image, cursor_alpha, cursor_x, cursor_y)
            cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
            frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
            mineclip_frame = process_frame_mineclip(frame)
            frame = resize_image(frame, AGENT_RESOLUTION)
            frames.append(frame)
            frames_mineclip.append(mineclip_frame)
            actions.append(action)

    video.release()
    return frames, frames_mineclip, actions
