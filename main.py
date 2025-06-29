from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import torch
import time
import os

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
