from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import torch
import time
import base64
from io import BytesIO
from typing import List
import numpy as np
import cv2
from PIL import Image

app = FastAPI()

# Cargar modelo al iniciar
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./cache')

@app.post("/embeddings")
async def get_embeddings(request: Request):
    try:
        data = await request.json()
        tags = data.get("tags", [])

        if not tags or not isinstance(tags, list):
            return JSONResponse(status_code=400, content={"error": "Field 'tags' must be a list."})

        start_time = time.perf_counter()
        with torch.inference_mode():
            embeddings_tensor = model.encode(tags, batch_size=16, convert_to_tensor=True)
        embeddings = embeddings_tensor.cpu().tolist()
        print(f"⏳ [Get Embeddings] Tiempo de ejecución: {time.perf_counter() - start_time:.4f} segundos")

        return {"tags": tags, "embeddings": embeddings}

    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

@app.post("/color-histograms")
async def get_color_histograms(request: Request):
    try:
        data = await request.json()
        images = data.get("images", [])

        if not images or not isinstance(images, list):
            return JSONResponse(status_code=400, content={"error": "Field 'images' must be a list."})

        hue_bins = 16
        sat_bins = 4

        start_time = time.perf_counter()
        results = []

        for item in images:
            idx = item.get("id")
            base64_image = item.get("base64")

            if not idx or not base64_image:
                continue

            image_data = base64.b64decode(base64_image)
            img = Image.open(BytesIO(image_data)).convert("RGB")

            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            hsv_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

            hist = cv2.calcHist([hsv_image], [0, 1], None, [hue_bins, sat_bins], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histogram_vector = hist.tolist()

            results.append({"id": idx, "embedding": histogram_vector})

        print(f"⏳ [Get Color Histograms] Tiempo de ejecución: {time.perf_counter() - start_time:.4f} segundos")

        return {"embeddings": results}

    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
