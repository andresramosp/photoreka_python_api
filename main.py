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
from dimensionality_reduction import reduce_embeddings_to_3d

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
        dominant_n = 4

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

            # Histograma completo
            hist = cv2.calcHist([hsv_image], [0, 1], None, [hue_bins, sat_bins], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histogram_vector = hist.tolist()

            # Calcular histograma reducido
            hue_sums = []
            for h in range(hue_bins):
                start_idx = h * sat_bins
                hue_sum = float(np.sum(hist[start_idx:start_idx + sat_bins]))  # Forzar tipo float
                hue_sums.append((h, hue_sum))

            # Obtener los índices de los hues más dominantes
            top_hues = sorted(hue_sums, key=lambda x: x[1], reverse=True)[:dominant_n]

            reduced_vector = [0.0] * len(hist)
            for hue_index, _ in top_hues:
                start_idx = hue_index * sat_bins
                for s in range(sat_bins):
                    reduced_vector[start_idx + s] = float(hist[start_idx + s])  # Forzar float

            results.append({
                "id": idx,
                "embedding_full": histogram_vector,
                "embedding_dominant": reduced_vector
            })

        print(f"⏳ [Get Color Histograms] Tiempo de ejecución: {time.perf_counter() - start_time:.4f} segundos")

        return {"embeddings": results}

    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

@app.post("/reduce_embeddings_to_3d")
async def reduce_embeddings_to_3d_endpoint(request: Request):
    try:
        data = await request.json()

        items = data.get("items")
        if items is None or not isinstance(items, list):
            return JSONResponse(status_code=400, content={"error": "Missing or invalid 'items' list"})

        method = data.get("method", "pca_tsne")
        tsne_perplexity = data.get("tsne_perplexity", 30.0)
        tsne_metric = data.get("tsne_metric", "cosine")
        tsne_learning_rate = data.get("tsne_learning_rate", 200.0)  # default típico para t-SNE
        random_state = data.get("random_state", 42)

        # Nuevos parámetros con defaults
        output_dims = data.get("output_dims", 3)
        normalize = data.get("normalize", "none")
        pca_whiten = data.get("pca_whiten", False)
        pca_max_components = data.get("pca_max_components")
        pca_pre_tsne = data.get("pca_pre_tsne", True)
        tsne_perplexity_strategy = data.get("tsne_perplexity_strategy", "fixed")

        # UMAP params
        umap_n_neighbors = data.get("umap_n_neighbors", 15)
        umap_min_dist = data.get("umap_min_dist", 0.1)
        umap_metric = data.get("umap_metric", "cosine")  # UMAP metric
        umap_n_components_pca = data.get("umap_n_components_pca", 50)
        umap_spread = data.get("umap_spread", 1.0)

        # Validación de method
        valid_methods = {"pca", "pca_tsne", "umap"}
        if method not in valid_methods:
            return JSONResponse(status_code=400, content={
                "error": f"Invalid method '{method}'. Must be one of: {', '.join(valid_methods)}"
            })

        try:
            tsne_perplexity_value = float(tsne_perplexity)
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={"error": "'tsne_perplexity' must be a number"})

        try:
            tsne_learning_rate_value = float(tsne_learning_rate)
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={"error": "'tsne_learning_rate' must be a number"})

        try:
            random_state_value = None if random_state in (None, "none", "None") else int(random_state)
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={"error": "'random_state' must be null or an integer"})
        
        try:
            output_dims_value = int(output_dims)
            if output_dims_value < 1:
                raise ValueError("output_dims must be >= 1")
        except (TypeError, ValueError):
            return JSONResponse(status_code=400, content={"error": "'output_dims' must be a positive integer"})

        embeddings = []
        ids = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                return JSONResponse(status_code=400, content={"error": f"Invalid item at index {idx}: expected an object"})

            embedding = item.get("embedding")
            if embedding is None:
                return JSONResponse(status_code=400, content={"error": f"Missing 'embedding' for item at index {idx}"})

            try:
                vector = [float(value) for value in embedding]
            except (TypeError, ValueError):
                return JSONResponse(status_code=400, content={"error": f"Embedding at index {idx} must be an iterable of numbers"})

            embeddings.append(vector)
            ids.append(item.get("id", str(idx)))

        start_time = time.perf_counter()
        try:
            reduced, metadata = reduce_embeddings_to_3d(
                embeddings,
                method=method,
                output_dims=output_dims_value,
                normalize=normalize,
                tsne_perplexity=tsne_perplexity_value,
                tsne_metric=tsne_metric,
                tsne_learning_rate=tsne_learning_rate_value,
                tsne_perplexity_strategy=tsne_perplexity_strategy,
                pca_whiten=pca_whiten,
                pca_max_components=pca_max_components,
                pca_pre_tsne=pca_pre_tsne,
                umap_n_neighbors=umap_n_neighbors,
                umap_min_dist=umap_min_dist,
                umap_spread=umap_spread,
                umap_metric=umap_metric,
                umap_n_components_pca=umap_n_components_pca,
                random_state=random_state_value,
                return_metadata=True,
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        except RuntimeError as exc:
            # Para el caso de UMAP no instalado
            return JSONResponse(status_code=400, content={"error": str(exc)})

        print(f"⏳ [Reduce Embeddings to 3D] Tiempo de ejecución: {time.perf_counter() - start_time:.4f} segundos")

        response_items = [
            {
                "id": id_,
                "embedding_3d": coords
            }
            for id_, coords in zip(ids, reduced)
        ]

        meta = {
            "method": metadata.method,
            "original_dimension": metadata.original_dimension,
            "effective_dimension": metadata.effective_dimension,
            "n_samples": metadata.n_samples,
        }
        if metadata.tsne_perplexity is not None:
            meta["tsne_perplexity"] = metadata.tsne_perplexity
        if metadata.random_state is not None:
            meta["random_state"] = metadata.random_state
        if metadata.method == "umap":
            # expose umap_spread if present in extra
            spread = metadata.extra.get("umap_spread") if hasattr(metadata, "extra") else None
            if spread is not None:
                meta["umap_spread"] = spread

        return JSONResponse(
            content={
                "items": response_items,
                "meta": meta,
            }
        )

    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
