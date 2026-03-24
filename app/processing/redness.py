"""
Redness Detection Module
Analyses skin redness by measuring the proportion of skin pixels in the
red HSV hue range, following the supervisor's recommended approach:
  - Convert to HSV
  - Define red hue range (H: 0-10, 160-180)
  - Calculate percentage of skin pixels in red range
  - Track relative changes vs. baseline
"""

import cv2
import numpy as np


def analyse_redness(face_array):
    """
    Measure redness across the facial image.

    Args:
        face_array: numpy array (H, W, 3) RGB, preprocessed

    Returns:
        dict:
            redness_score  (float) 0-100, where 100 = no redness (best)
            redness_pct    (float) raw percentage of red pixels
            redness_map    (numpy array) heatmap of redness distribution
    """
    if face_array is None or face_array.size == 0:
        return {'redness_score': 50.0, 'redness_pct': 0.0, 'redness_map': None}

    bgr = cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Red hue masks (wraps at 180)
    lower_red1 = np.array([0,   40,  40])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([155, 40,  40])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Exclude very dark pixels (shadow areas)
    v_channel = hsv[:, :, 2]
    bright_mask = (v_channel > 40).astype(np.uint8) * 255
    red_mask    = cv2.bitwise_and(red_mask, bright_mask)

    total_bright = np.sum(bright_mask > 0)
    red_pixels   = np.sum(red_mask > 0)
    redness_pct  = (red_pixels / (total_bright + 1e-6)) * 100.0

    # Score: 0 % redness → 100; 30 %+ redness → 0
    redness_score = max(0.0, 100.0 - (redness_pct / 30.0) * 100.0)
    redness_score = round(redness_score, 1)

    # Build a smooth redness heatmap
    red_float   = red_mask.astype(np.float32) / 255.0
    red_blurred = cv2.GaussianBlur(red_float, (21, 21), 0)
    redness_map = (red_blurred * 255).astype(np.uint8)
    redness_map = cv2.applyColorMap(redness_map, cv2.COLORMAP_HOT)
    redness_map = cv2.cvtColor(redness_map, cv2.COLOR_BGR2RGB)

    return {
        'redness_score': redness_score,
        'redness_pct':   round(redness_pct, 2),
        'redness_map':   redness_map,
    }


def redness_by_region(face_array, regions):
    """
    Returns per-region redness scores for forehead, cheeks, nose, chin.
    """
    if not regions:
        return {}
    scores = {}
    for region_name, region_arr in regions.items():
        if region_arr.size > 0:
            res = analyse_redness(region_arr)
            scores[region_name] = res['redness_score']
    return scores
