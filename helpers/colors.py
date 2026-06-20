import inspect
from skimage.exposure import match_histograms
import cv2

# skimage renamed `multichannel=True` -> `channel_axis=-1` (deprecated in 0.19,
# removed in 0.21+). Pick whichever the installed version supports so color
# coherence works across old and new scikit-image.
_MH_USES_CHANNEL_AXIS = "channel_axis" in inspect.signature(match_histograms).parameters

def _match_histograms(src, ref):
    if _MH_USES_CHANNEL_AXIS:
        return match_histograms(src, ref, channel_axis=-1)
    return match_histograms(src, ref, multichannel=True)

def maintain_colors(prev_img, color_match_sample, mode):
    if mode == 'Match Frame 0 RGB':
        return _match_histograms(prev_img, color_match_sample)
    elif mode == 'Match Frame 0 HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = _match_histograms(prev_img_hsv, color_match_hsv)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else: # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = _match_histograms(prev_img_lab, color_match_lab)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
