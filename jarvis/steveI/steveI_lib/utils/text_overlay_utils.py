import numpy as np
import cv2

from jarvis.steveI.steveI_lib.config import FONT


def created_fitted_text_image(desired_width, text, thickness=2,
                              background_color=(255, 255, 255), text_color=(0, 0, 0), height_padding=20):
    """Create an image with text fitted to the desired width."""
    font_scale = 0.1
    text_size, _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    text_width, _ = text_size
    pad = desired_width // 5
    while text_width < desired_width - pad:
        font_scale += 0.1
        text_size, _ = cv2.getTextSize(text, FONT, font_scale, thickness)
        text_width, _ = text_size
    image = np.zeros((text_size[1] + 2 * height_padding, desired_width, 3), dtype=np.uint8)
    image[:] = background_color
    org = ((image.shape[1] - text_width) // 2, image.shape[0] - height_padding)
    return cv2.putText(image, text, org, FONT, font_scale, text_color, thickness)

