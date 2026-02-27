import sys
import os

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np

from model import UNetPro

MODEL_PATH = "model/unet_rx_3band.pth"


def load_model():

    model = UNetPro()

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    # Case 1: checkpoint is state_dict directly
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])

        else:
            # assume checkpoint itself is state_dict
            model.load_state_dict(checkpoint)

    else:
        raise ValueError("Unsupported checkpoint format")

    model.eval()

    return model


def preprocess(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found")

    img = cv2.resize(img, (256, 256))

    img = img / 255.0

    img = np.transpose(img, (2, 0, 1))

    img = torch.tensor(img).float().unsqueeze(0)

    return img


def detect(image_path):

    model = load_model()

    img_tensor = preprocess(image_path)

    with torch.no_grad():
        output = model(img_tensor)

    mask = output.squeeze().numpy()

    plume_pixels = np.sum(mask > 0.5)

    confidence = float(np.mean(mask))

    print(f"Plume pixels: {plume_pixels}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/detect.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    detect(image_path)