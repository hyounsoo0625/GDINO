import cv2
import numpy as np
if not hasattr(np, 'float_'): np.float_ = np.float64
import skimage.filters
original_gaussian = skimage.filters.gaussian
def patched_gaussian(*args, **kwargs):
    if 'multichannel' in kwargs: kwargs['channel_axis'] = -1 if kwargs.pop('multichannel') else None
    return original_gaussian(*args, **kwargs)
skimage.filters.gaussian = patched_gaussian
from imagecorruptions import corrupt

def apply_corruption(img_bgr, corruption_name, severity=3):
    """
    Returns the original image (in BGR format) with the specified corruption applied.

    Args:
        img_bgr (numpy.ndarray): The original image read by OpenCV (BGR format)
        corruption_name (str): The name of the corruption to apply (returns the original image if None)
        severity (int): The intensity of the corruption (can be set from 1 to 5; default is 3)
    Returns:
        numpy.ndarray: Image with corruption applied (BGR format)
    """
    if corruption_name is None: 
        return img_bgr

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    corrupted_rgb = corrupt(img_rgb, corruption_name=corruption_name, severity=severity)

    return cv2.cvtColor(corrupted_rgb, cv2.COLOR_RGB2BGR)