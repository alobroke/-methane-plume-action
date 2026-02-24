from fastapi import FastAPI, UploadFile, File
import torch
import shutil
import numpy as np
import os

from model import UNetPro
from rx import rx_detector
from utils import load_image, fuse, compute_plume_area, compute_confidence, alert_level

# -------------------------------
# Create FastAPI app instance
# -------------------------------

app = FastAPI(
    title="Methane Detection API",
    description="Hybrid RX + U-Net Methane Plume Detection System",
    version="1.0"
)

# -------------------------------
# Device
# -------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load trained model
# -------------------------------

MODEL_PATH = "unet_rx_3band.pth"

model = UNetPro(in_channels=3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------------------------------
# Routes
# -------------------------------

@app.post("/detect")
async def detect_methane(file: UploadFile = File(...)):

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print("Loading image...")
        image = load_image(temp_path)
        print("Image shape:", image.shape)

        print("Running RX detector...")
        rx_map = rx_detector(image)

        print("Running U-Net...")
        img_tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            unet_prob = model(img_tensor).cpu().squeeze().numpy()

        print("Fusing outputs...")
        fused = fuse(rx_map, unet_prob)

        threshold = np.percentile(fused, 98)
        mask = (fused > threshold).astype(np.uint8)

        area = compute_plume_area(mask)
        confidence = compute_confidence(fused, mask)
        alert = alert_level(area, confidence)

        return {
            "status": "success",
            "plume_area_m2": area,
            "confidence": round(confidence, 3),
            "alert_level": alert
        }

    except Exception as e:
        print("ERROR:", e)
        return {"error": str(e)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


