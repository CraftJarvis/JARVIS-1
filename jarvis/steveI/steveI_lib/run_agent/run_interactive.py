import argparse
import json
import os

import cv2
import numpy as np
import torch
from jarvis.steveI.steveI_lib.data.text_alignment.vae import load_vae_model
from jarvis.steveI.steveI_lib.utils.mineclip_agent_env_utils import load_mineclip_agent_env
from jarvis.steveI.steveI_lib.utils.text_overlay_utils import created_fitted_text_image
from jarvis.steveI.steveI_lib.utils.video_utils import save_frames_as_video

from jarvis.steveI.steveI_lib.config import PRIOR_INFO, DEVICE
from jarvis.steveI.steveI_lib.utils.embed_utils import get_prior_embed

FPS = 20

def create_video_frame(gameplay_pov, prompt):
    """Creates a frame for the generated video with the gameplay POV and the prompt text on the right side."""
    frame = cv2.cvtColor(gameplay_pov, cv2.COLOR_RGB2BGR)
    prompt_section = created_fitted_text_image(frame.shape[1] // 2, prompt,
                                               background_color=(0, 0, 0),
                                               text_color=(255, 255, 255))
    pad_top_height = (frame.shape[0] - prompt_section.shape[0]) // 2
    pad_top = np.zeros((pad_top_height, prompt_section.shape[1], 3), dtype=np.uint8)
    pad_bottom_height = frame.shape[0] - pad_top_height - prompt_section.shape[0]
    pad_bottom = np.zeros((pad_bottom_height, prompt_section.shape[1], 3), dtype=np.uint8)
    prompt_section = np.vstack((pad_top, prompt_section, pad_bottom))
    frame = np.hstack((frame, prompt_section))
    return frame


def run_interactive(in_model, in_weights, cond_scale, prior_info, output_video_dirpath, env_config):
    """Runs the agent in the MineRL env and allows the user to enter prompts to control the agent.
    Clicking on the gameplay window will pause the gameplay and allow the user to enter a new prompt.

    Typing 'reset agent' will reset the agent's state.
    Typing 'reset env' will reset the environment.
    Typing 'save video' will save the video so far (and ask for a video name). It will also save a json storing
        the active prompt at each frame of the video.
    """
    agent, mineclip, env = load_mineclip_agent_env(in_model, in_weights, cond_scale, env_config)
    prior = load_vae_model(prior_info)
    window_name = 'STEVE-1 Gameplay (Click to Enter Prompt)'

    state = {'obs': None}
    os.makedirs(output_video_dirpath, exist_ok=True)
    video_frames = []
    frame_prompts = []

    def handle_prompt():
        # Pause the gameplay and ask for a new prompt
        prompt = input('\n\nEnter a prompt:\n>').strip().lower()

        # Reset the agent or env if prompted
        if prompt == 'reset agent':
            print('\n\nResetting agent...')
            agent.reset(cond_scale)
            print(f'Done. Continuing gameplay with previous prompt...')
            return
        elif prompt == 'reset env':
            reset_env()
            print(f'Done. Continuing gameplay with previous prompt...')
            return

        # Save the video so far if prompted
        if prompt == 'save video':
            # Ask for a video name
            video_name = input('Enter a video name:\n>').strip().lower()

            # Save both the video and the prompts for each frame
            output_video_filepath = os.path.join(output_video_dirpath, f'{video_name}.mp4')
            prompts_for_frames_filepath = os.path.join(output_video_dirpath, f'{video_name}.json')
            print(f'Saving video to {output_video_filepath}...')
            save_frames_as_video(video_frames, output_video_filepath, fps=FPS)
            print(f'Saving prompts for frames to {prompts_for_frames_filepath}...')
            with open(prompts_for_frames_filepath, 'w') as f:
                json.dump(frame_prompts, f)
            print(f'Done. Continuing gameplay with previous prompt...')
            return

        # Use prior to get the prompt embed
        prompt_embed = get_prior_embed(prompt, mineclip, prior, DEVICE)

        with torch.cuda.amp.autocast():
            while True:
                minerl_action = agent.get_action(state['obs'], prompt_embed)
                state['obs'], _, _, _, info = env.step(minerl_action)

                frame = create_video_frame(info['pov'], prompt)
                video_frames.append(frame)
                frame_prompts.append(prompt)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)

    def reset_env():
        print('\nResetting environment...')
        state['obs'], _ = env.reset()
    reset_env()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            handle_prompt()

    initial_frame = create_video_frame(state['obs']['img'], 'Click to Enter a Prompt')
    cv2.imshow(window_name, initial_frame)
    cv2.setMouseCallback(window_name, on_click)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Close the window when 'q' is pressed
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model', type=str, default='data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default='data/weights/steve1/steve1.weights')
    parser.add_argument('--prior_weights', type=str, default='data/weights/steve1/steve1_prior.pt')
    parser.add_argument('--output_video_dirpath', type=str, default='data/generated_videos/interactive_videos')
    parser.add_argument('--cond_scale', type=float, default=6.0)
    parser.add_argument('--env_config', type=str, default='plains')
    args = parser.parse_args()

    run_interactive(
        in_model=args.in_model,
        in_weights=args.in_weights,
        cond_scale=args.cond_scale,
        prior_info=PRIOR_INFO,
        output_video_dirpath=args.output_video_dirpath,
        env_config=args.env_config
    )
