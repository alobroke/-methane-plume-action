import rasterio
import numpy as np


# Image Loader


def load_image(file_path):
    """
    Load satellite image from TIFF file
    Output shape: (H, W, C)
    """
    with rasterio.open(file_path) as src:
        img = src.read().astype(np.float32)  
        img = np.transpose(img, (1, 2, 0))   
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        return img



# Fusion Engine


def fuse(rx_map, unet_map):
    """
    Fuse physics (RX) and AI (U-Net) maps
    """
    return 0.7 * unet_map + 0.3 * rx_map



# Plume Area Calculation


PIXEL_AREA = 25   

def compute_plume_area(mask):
    """
    Compute methane plume area in square meters
    """
    return int(mask.sum() * PIXEL_AREA)



# Confidence Estimation


def compute_confidence(fused_map, mask):
    """
    Confidence score from probability + spatial coverage
    """
    max_prob = fused_map.max()
    coverage = mask.sum() / mask.size
    confidence = 0.6 * max_prob + 0.4 * coverage
    return float(confidence)



# Alert Engine


def alert_level(area, confidence):
    """
    Risk classification based on plume size & confidence
    """
    if area > 3000 and confidence > 0.85:
        return "CRITICAL"
    elif area > 1500 and confidence > 0.7:
        return "HIGH"
    elif area > 500:
        return "MEDIUM"
    else:
        return "LOW"
