"# Photoreka Python CPU API

This is a lightweight CPU-based API for handling embeddings, color histograms, and dimensionality reduction.

## Endpoints

- `POST /embeddings` - Generate embeddings from text tags
- `POST /color-histograms` - Extract color histograms from images
- `POST /reduce_embeddings_to_3d` - Reduce high-dimensional embeddings to 3D space using PCA, t-SNE, or UMAP

## Dimensionality Reduction API

El endpoint `/reduce_embeddings_to_3d` soporta un nuevo parámetro para UMAP:

- `umap_spread` (float, default 1.0): controla la expansión global de la nube de puntos. Úsalo junto con `umap_min_dist`:
  - `umap_min_dist` define cuán pegados están los puntos dentro de cada clúster (densidad local)
  - `umap_spread` define el tamaño general que pueden ocupar los clústeres en el espacio embebido (escala global)

Ejemplo de invocación para obtener clústeres más expansivos:

```json
{
  "method": "umap",
  "items": [
    { "id": "a", "embedding": [0.1, 0.2, 0.3, 0.4] },
    { "id": "b", "embedding": [0.11, 0.18, 0.29, 0.41] }
  ],
  "umap_n_neighbors": 30,
  "umap_min_dist": 0.2,
  "umap_spread": 3.0,
  "output_dims": 3,
  "umap_metric": "cosine",
  "random_state": 42
}
```

Si no se especifica, `umap_spread` se mantiene en 1.0 (comportamiento por defecto de UMAP)." 
