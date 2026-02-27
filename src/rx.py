import numpy as np

def rx_detector(image):
    """
    Reed-Xiaoli (RX) Anomaly Detector

    Parameters:
        image : numpy array of shape (H, W, C)
                Satellite image with C spectral bands

    Returns:
        rx_map : numpy array of shape (H, W)
                 Normalized anomaly score map (0 → 1)
    """

    H, W, C = image.shape

   
    X = image.reshape(-1, C)

  
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    
    cov_inv = np.linalg.pinv(cov)

   
    diff = X - mean
    rx_scores = np.sum(diff @ cov_inv * diff, axis=1)

   
    rx_map = rx_scores.reshape(H, W)

 
    rx_map = (rx_map - rx_map.min()) / (rx_map.max() - rx_map.min() + 1e-6)

    return rx_map
