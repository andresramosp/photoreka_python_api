from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Optional, Sequence, Dict, Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap  # type: ignore
    _UMAP_AVAILABLE = True
except Exception:  # pragma: no cover
    _UMAP_AVAILABLE = False


ReductionMethod = Literal["pca", "pca_tsne", "umap"]
NormalizationMethod = Literal["none", "l2", "standard", "mean_center", "auto"]


@dataclass
class ReductionMetadata:
    method: ReductionMethod
    original_dimension: int
    effective_dimension: int
    n_samples: int
    tsne_perplexity: Optional[float] = None
    random_state: Optional[int] = None
    tsne_metric: Optional[str] = None
    normalize: Optional[NormalizationMethod] = None
    pca_whiten: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _pad_to_n_dimensions(values: np.ndarray, n_dims: int) -> np.ndarray:
    """Pad (or trim) an array to have exactly ``n_dims`` columns.

    Backwards compatible replacement of the previous _pad_to_three_dimensions.
    """
    if values.shape[1] == n_dims:
        return values
    if values.shape[1] > n_dims:
        return values[:, :n_dims]
    if n_dims <= 0:
        raise ValueError("n_dims must be positive")
    padding = np.zeros((values.shape[0], n_dims - values.shape[1]), dtype=values.dtype)
    return np.concatenate([values, padding], axis=1)


def reduce_embeddings_to_3d(
    embeddings: Sequence[Sequence[float]],
    method: ReductionMethod = "pca_tsne",
    *,
    tsne_perplexity: float = 30.0,
    random_state: Optional[int] = 42,
    return_metadata: bool = False,
    normalize: NormalizationMethod = "none",
    tsne_metric: str = "cosine",
    pca_whiten: bool = False,
    # UMAP params
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "cosine",
    umap_n_components_pca: int = 50,
    umap_spread: float = 1.0,
    # New generic projection params
    output_dims: int = 3,
    pca_max_components: Optional[int] = 50,
    pca_pre_tsne: bool = True,
    tsne_learning_rate: float = 200.0,
    tsne_perplexity_strategy: Literal["fixed", "auto"] = "fixed",
    tsne_auto_min_perplexity: float = 30.0,
    tsne_auto_max_perplexity: float = 120.0,
    tsne_auto_scale: float = 30.0,
    # If True and strategy auto, perplexity = clamp((n/100)*scale, min, max)
) -> List[List[float]] | tuple[List[List[float]], ReductionMetadata]:
    """Project high-dimensional embeddings into a low-dimensional space (default 3D).

    Parameters
    ----------
    embeddings:
        Iterable of equal-length embedding vectors (OpenAI, CLIP, etc.).
    method:
        "pca": proyección lineal directa.
        "pca_tsne": PCA previa (hasta 50 comps) + t-SNE 3D.
        "umap": (opcional PCA previa a umap_n_components_pca) + UMAP 3D.
    tsne_perplexity:
        Only used for ``"pca_tsne"``. Automatically adjusted when there are
        few samples to keep the configuration valid.
    normalize:
        Preprocesado opcional:
            - "none" (default) sin cambios
            - "l2" normaliza cada vector a norma 1 (recomendado p/ coseno)
            - "standard" z-score
            - "mean_center" solo centra
            - "auto" heurística: si dimensión >=256 asume embedding tipo CLIP/sentence y aplica L2 si no está ya normalizado.
    tsne_metric:
        Métrica para t-SNE. Valores comunes: "cosine" (recomendado para embeddings CLIP / sentence) o "euclidean".
        Si se selecciona "cosine" y normalize="none" se aplicará l2 internamente.
    pca_whiten:
        Si True, usa PCA(con whitening) antes de t-SNE para uniformar varianzas. Puede ayudar a formar grupos más compactos.
    umap_n_neighbors / umap_min_dist / umap_metric:
        Parámetros estándar de UMAP. Métrica "cosine" recomendada para embeddings CLIP.
    umap_n_components_pca:
        Número máximo de componentes PCA previas antes de UMAP (para acelerar y reducir ruido). Se limita por dims y muestras.
    random_state:
        Seed used for the stochastic parts of PCA/t-SNE to ensure reproducible
        layouts. Set to ``None`` for fully stochastic behaviour.
    return_metadata:
        When ``True`` the function returns ``(vectors, metadata)``.

    Parameters added (advanced)
    ---------------------------
    output_dims: número de dimensiones objetivo (por defecto 3).
    pca_max_components: máximo de componentes PCA previas (None para usar output_dims cuando method==pca, o heurística  min( max, dim, n-1)).
    pca_pre_tsne: si False evita PCA previa antes de t-SNE (puede ser más lento pero preserva más estructura fina).
    tsne_perplexity_strategy: 'fixed' usa tsne_perplexity; 'auto' calcula perplexity = clamp( (n/100)*tsne_auto_scale, tsne_auto_min_perplexity, tsne_auto_max_perplexity ).
    tsne_auto_*: parámetros para la estrategia 'auto'.

    Returns
    -------
    list[list[float]] or tuple[list[list[float]], ReductionMetadata]
        Coordenadas de salida (output_dims) y opcional metadata.
    """

    if embeddings is None:
        raise ValueError("Embeddings input cannot be None")

    if isinstance(embeddings, np.ndarray):
        matrix = embeddings
    else:
        matrix = np.asarray(list(embeddings), dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError("Embeddings must be provided as a 2D array-like structure")

    # Preprocesado / normalización
    if normalize not in {"none", "l2", "standard", "mean_center", "auto"}:
        raise ValueError(f"Unsupported normalization method: {normalize}")

    applied_auto = False
    if normalize == "auto":
        # Heurística simple: si la media de normas difiere mucho de 1 y la dimensión es alta
        # aplicar L2. Si ya están casi normalizados (norma media entre 0.95 y 1.05) no tocar.
        norms_probe = np.linalg.norm(matrix, axis=1)
        mean_norm = norms_probe.mean()
        if matrix.shape[1] >= 256 and not (0.95 <= mean_norm <= 1.05):
            norms = norms_probe.reshape(-1, 1)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
            applied_auto = True
            normalize_effective = "l2"
        else:
            normalize_effective = "none"
    elif normalize == "l2":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        normalize_effective = "l2"
    elif normalize == "standard":
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        matrix = (matrix - mean) / std
        normalize_effective = "standard"
    elif normalize == "mean_center":
        mean = matrix.mean(axis=0, keepdims=True)
        matrix = matrix - mean
        normalize_effective = "mean_center"
    else:  # none
        normalize_effective = "none"

    # Forzar l2 si se usa métrica coseno y el usuario no normalizó
    forced_l2 = False
    if tsne_metric == "cosine" and normalize_effective == "none":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        forced_l2 = True

    n_samples, original_dim = matrix.shape

    if n_samples == 0:
        result = np.zeros((0, output_dims), dtype=np.float32)
        return _finalise_output(result, method, original_dim, 0, None, random_state, return_metadata)

    if n_samples == 1:
        # Con un solo embedding no se puede aplicar ni PCA (varianza = 0 tras centrar)
        # ni t-SNE (requiere varias muestras). En lugar de devolver [0,0,0],
        # exponemos una proyección directa: tomamos las primeras 3 componentes
        # (o las que haya) y las rellenamos a 3 dimensiones para preservar
        # información útil al cliente.
        single_vec = matrix[0]
        if original_dim >= output_dims:
            projected = single_vec[:output_dims]
        else:
            projected = np.pad(single_vec, (0, output_dims - original_dim))
        result = projected.reshape(1, output_dims).astype(np.float32)
        return _finalise_output(result, method, original_dim, 1, None, random_state, return_metadata)

    if method not in {"pca", "pca_tsne", "umap"}:
        raise ValueError(f"Unsupported reduction method: {method}")

    # When we only have a handful of samples or dimensions, PCA alone is stable.
    if method == "pca" or (n_samples <= 5 and method != "umap"):
        # For pure PCA mode we ignore pca_pre_tsne flag.
        if pca_max_components is None:
            n_components = min(output_dims, original_dim, n_samples)
        else:
            n_components = min(pca_max_components, output_dims, original_dim, n_samples)
        pca = PCA(n_components=n_components, random_state=random_state, whiten=pca_whiten)
        reduced = pca.fit_transform(matrix)
        reduced = _pad_to_n_dimensions(reduced, output_dims)
        return _finalise_output(
            reduced,
            "pca",
            original_dim,
            n_samples,
            None,
            random_state,
            return_metadata,
            tsne_metric=tsne_metric if method == "pca_tsne" else None,
            normalize=("auto(l2)" if applied_auto else normalize_effective if not forced_l2 else "l2"),
            pca_whiten=pca_whiten,
            extra={"forced_l2": forced_l2, "applied_auto": applied_auto, "normalize_effective": normalize_effective},
        )

    # UMAP branch
    if method == "umap":
        if not _UMAP_AVAILABLE:
            raise RuntimeError("UMAP not installed. Please add 'umap-learn' to requirements.")
        
        # UMAP needs sufficient samples to work properly
        if n_samples < 4:
            raise ValueError(f"UMAP requires at least 4 samples, but only {n_samples} provided. Use 'pca' instead for small datasets.")

        # PCA previa opcional para acelerar si dimensión es alta
        if original_dim > umap_n_components_pca:
            n_comps = min(umap_n_components_pca, original_dim, n_samples - 1)
            if n_comps < 3:
                n_comps = min(3, original_dim, n_samples)
            pca = PCA(n_components=n_comps, random_state=random_state, whiten=pca_whiten)
            matrix_umap = pca.fit_transform(matrix)
            pca_used = True
            pca_components_used = n_comps
        else:
            matrix_umap = matrix
            pca_used = False
            pca_components_used = original_dim

        reducer = umap.UMAP(
            n_neighbors=min(max(2, umap_n_neighbors), n_samples - 1),
            min_dist=max(0.0, min(umap_min_dist, 0.99)),
            spread=float(umap_spread),
            metric=umap_metric,
            n_components=output_dims,
            random_state=random_state,
        )
        
        # Debug info for troubleshooting
        actual_neighbors = min(max(2, umap_n_neighbors), n_samples - 1)
        if actual_neighbors < 2:
            raise ValueError(f"Cannot run UMAP with n_neighbors={actual_neighbors}. Need at least 2 neighbors but only have {n_samples} samples.")
        reduced = reducer.fit_transform(matrix_umap)
        reduced = _pad_to_n_dimensions(reduced, output_dims)
        return _finalise_output(
            reduced,
            "umap",
            original_dim,
            n_samples,
            None,
            random_state,
            return_metadata,
            tsne_metric=None,
            normalize=("auto(l2)" if applied_auto else normalize_effective if not forced_l2 else "l2"),
            pca_whiten=pca_whiten if pca_used else False,
            extra={
                "forced_l2": forced_l2,
                "applied_auto": applied_auto,
                "normalize_effective": normalize_effective,
                "umap_n_neighbors": umap_n_neighbors,
                "umap_min_dist": umap_min_dist,
                "umap_spread": umap_spread,
                "umap_metric": umap_metric,
                "pca_pre_used": pca_used,
                "pca_components_used": pca_components_used,
                "output_dims": output_dims,
            },
        )

    # PCA pre-reduction before t-SNE keeps things faster and more stable.
    # Decide PCA dimensionality for t-SNE pipeline
    if pca_pre_tsne:
        if pca_max_components is None:
            # Heuristic: keep up to min( max( output_dims, 50 ), dim, n-1 )
            target_max = max(output_dims, 50)
        else:
            target_max = pca_max_components
        max_pca_components_eff = min(target_max, original_dim, n_samples - 1)
        if max_pca_components_eff < output_dims:
            max_pca_components_eff = min(output_dims, original_dim, n_samples)
        pca = PCA(n_components=max_pca_components_eff, random_state=random_state, whiten=pca_whiten)
        pca_result = pca.fit_transform(matrix)
    else:
        pca_result = matrix
        max_pca_components_eff = original_dim

    # Adapt perplexity to the sample count so that t-SNE requirements are met.
    if tsne_perplexity_strategy == "auto":
        # auto formula: (n/100)*scale clamped
        proposed = (n_samples / 100.0) * tsne_auto_scale
        max_valid_perplexity = max(tsne_auto_min_perplexity, min(proposed, tsne_auto_max_perplexity, (n_samples - 1) / 3.0))
    else:
        max_valid_perplexity = max(2.0, min(tsne_perplexity, float(n_samples - 1) / 3.0, 50.0))
        if max_valid_perplexity >= n_samples:
            max_valid_perplexity = max(2.0, float(n_samples - 1) / 3.0)
        if max_valid_perplexity < 2.0:
            max_valid_perplexity = 2.0

    # Create TSNE with backward compatibility across scikit-learn versions.
    # Older versions (<1.2) don't accept 'n_iter' during __init__ (it existed but
    # signature differences may raise errors depending on build). We'll attempt
    # to set n_iter via kwargs, and if that fails retry without it.
    tsne_kwargs = dict(
        n_components=output_dims,
        learning_rate=tsne_learning_rate,
        perplexity=max_valid_perplexity,
        init="pca",
        random_state=random_state,
        metric=tsne_metric,
    )

    tsne = None
    try:
        tsne = TSNE(**tsne_kwargs, n_iter=1000)
    except TypeError:
        # Retry without n_iter; default iterations (usually 1000) will apply.
        tsne = TSNE(**tsne_kwargs)
    reduced = tsne.fit_transform(pca_result)

    return _finalise_output(
        _pad_to_n_dimensions(reduced, output_dims),
        "pca_tsne",
        original_dim,
        n_samples,
        max_valid_perplexity,
        random_state,
        return_metadata,
        tsne_metric=tsne_metric,
        normalize=("auto(l2)" if applied_auto else normalize_effective if not forced_l2 else "l2"),
        pca_whiten=pca_whiten,
        extra={
            "forced_l2": forced_l2,
            "applied_auto": applied_auto,
            "normalize_effective": normalize_effective,
            "pca_pre_tsne": pca_pre_tsne,
            "pca_components_used": max_pca_components_eff,
            "output_dims": output_dims,
            "tsne_perplexity_strategy": tsne_perplexity_strategy,
        },
    )


def _finalise_output(
    values: np.ndarray,
    method: ReductionMethod,
    original_dim: int,
    n_samples: int,
    tsne_perplexity: Optional[float],
    random_state: Optional[int],
    return_metadata: bool,
    *,
    tsne_metric: Optional[str] = None,
    normalize: Optional[NormalizationMethod] = None,
    pca_whiten: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> List[List[float]] | tuple[List[List[float]], ReductionMetadata]:
    values = values.astype(np.float32, copy=False)
    coords = values.tolist()

    if not return_metadata:
        return coords

    metadata = ReductionMetadata(
        method=method,
        original_dimension=original_dim,
        effective_dimension=values.shape[1],
        n_samples=n_samples,
        tsne_perplexity=tsne_perplexity if method == "pca_tsne" else None,
        random_state=random_state,
        tsne_metric=tsne_metric if method == "pca_tsne" else None,
        normalize=normalize,
        pca_whiten=pca_whiten,
        extra=extra or {},
    )
    return coords, metadata


__all__ = [
    "ReductionMetadata",
    "ReductionMethod",
    "reduce_embeddings_to_3d",
    "project_clip_embeddings",
    "evaluate_projection_quality",
]


def diagnose_embeddings(
    embeddings: Sequence[Sequence[float]] | np.ndarray,
    *,
    sample_limit: Optional[int] = 5000,
    compute_pairwise_cosine_stats: bool = True,
    pca_components: int = 50,
) -> Dict[str, Any]:
    """Devuelve métricas rápidas para entender la estructura antes de reducir.

    Parámetros
    ----------
    embeddings: colección 2D de embeddings.
    sample_limit: si hay más, se toma una muestra aleatoria (para estadísticas rápidas).
    compute_pairwise_cosine_stats: calcula media/desv de similitudes coseno (costo O(n^2)).
    pca_components: cuántos componentes intentar para varianza explicada.

    Retorna
    -------
    dict con claves:
        n_samples, dim, norm_mean, norm_std, cosine_mean, cosine_std,
        pca_explained_first_10, pca_cumulative_10
    """
    if isinstance(embeddings, np.ndarray):
        M = embeddings.astype(np.float32, copy=False)
    else:
        M = np.asarray(list(embeddings), dtype=np.float32)

    if M.ndim != 2:
        raise ValueError("Embeddings must be 2D")

    n, d = M.shape
    if n == 0:
        return {"n_samples": 0, "dim": d}

    # Subsample if needed
    if sample_limit and n > sample_limit:
        idx = np.random.choice(n, sample_limit, replace=False)
        M_sub = M[idx]
    else:
        M_sub = M

    norms = np.linalg.norm(M_sub, axis=1)
    norm_mean = float(norms.mean())
    norm_std = float(norms.std())

    cosine_mean = cosine_std = None
    if compute_pairwise_cosine_stats and len(M_sub) > 1:
        # Normalizar para coseno
        V = M_sub / np.clip(np.linalg.norm(M_sub, axis=1, keepdims=True), 1e-12, None)
        sims = V @ V.T
        # Ignorar diagonal
        mask = ~np.eye(len(M_sub), dtype=bool)
        vals = sims[mask]
        cosine_mean = float(vals.mean())
        cosine_std = float(vals.std())

    # PCA explained variance
    comps = min(pca_components, d, max(1, len(M_sub) - 1))
    if comps < 2:
        pca_explained_first_10 = []
        pca_cumulative_10 = 0.0
    else:
        pca_local = PCA(n_components=comps, random_state=0)
        try:
            pca_local.fit(M_sub)
            exp = pca_local.explained_variance_ratio_.tolist()
        except Exception:
            exp = []
        pca_explained_first_10 = [round(x, 6) for x in exp[:10]]
        pca_cumulative_10 = float(sum(exp[:10]))

    return {
        "n_samples": int(n),
        "dim": int(d),
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "cosine_mean": cosine_mean,
        "cosine_std": cosine_std,
        "pca_explained_first_10": pca_explained_first_10,
        "pca_cumulative_10": pca_cumulative_10,
    }


def project_clip_embeddings(
    embeddings: Sequence[Sequence[float]] | np.ndarray,
    *,
    prefer: Literal["pca", "umap", "tsne", "auto"] = "umap",
    random_state: int = 42,
    return_metadata: bool = False,
    target_method_fallback: Literal["tsne", "umap"] = "tsne",
    max_umap_neighbors: int = 40,
    min_umap_neighbors: int = 10,
    min_perplexity: int = 10,
    max_perplexity: int = 35,
    pca_whiten: bool = False,
) -> List[List[float]] | tuple[List[List[float]], ReductionMetadata]:
    """Convenience wrapper para embeddings tipo CLIP / sentence.

    Heurísticas:
      - Normalización: usa normalize="auto".
      - Método: si prefer="auto": usa UMAP si está disponible y n>=15; si no, t-SNE.
      - UMAP vecinos: clamp(log2(n)*2, min_umap_neighbors, max_umap_neighbors).
      - UMAP min_dist: 0.05 si n grande (>500), 0.08 intermedio, 0.12 si n<150.
      - t-SNE perplexity: clamp(n/5, min_perplexity, max_perplexity).

    Parámetros
    ----------
    prefer: forzar "umap", "tsne" o dejar "auto".
    target_method_fallback: a qué caer si prefer="umap" pero no hay UMAP.
    pca_whiten: se pasa al backend.
    """
    if isinstance(embeddings, np.ndarray):
        M = embeddings
    else:
        M = np.asarray(list(embeddings), dtype=np.float32)
    if M.ndim != 2:
        raise ValueError("Embeddings must be 2D")
    n, d = M.shape

    # Decide método
    method: ReductionMethod
    if prefer == "pca":
        method = "pca"  # type: ignore
    elif prefer == "umap" or (prefer == "auto" and n >= 15):
        if _UMAP_AVAILABLE and n >= 4:  # UMAP needs at least 4 samples
            method = "umap"  # type: ignore
        else:
            method = "pca_tsne" if target_method_fallback == "tsne" else "pca_tsne"
    elif prefer == "tsne" or prefer == "auto":
        method = "pca_tsne"  # type: ignore
    else:
        method = "pca_tsne"  # safety

    # Parametrización por método
    if method == "pca":
        return reduce_embeddings_to_3d(
            M,
            method="pca",
            normalize="auto",
            random_state=random_state,
            return_metadata=return_metadata,
            pca_whiten=pca_whiten,
            pca_max_components=100,
            output_dims=3,
        )
    elif method == "umap":
        import math
        nn = int(min(max(min_umap_neighbors, math.log2(max(2, n)) * 2), max_umap_neighbors))
        if n < 25:
            min_dist = 0.15
        elif n < 150:
            min_dist = 0.12
        elif n < 500:
            min_dist = 0.08
        else:
            min_dist = 0.05
        return reduce_embeddings_to_3d(
            M,
            method="umap",
            normalize="auto",
            umap_n_neighbors=nn,
            umap_min_dist=min_dist,
            umap_metric="cosine",
            umap_n_components_pca=100,
            random_state=random_state,
            return_metadata=return_metadata,
            pca_whiten=pca_whiten,
            output_dims=3,
        )

    # Parametrización t-SNE
    # Nuevo: usar estrategia 'auto' por defecto para perplexity más alta en n medianos
    perplexity = float(max(min_perplexity, min(n / 5.0, max_perplexity)))  # legacy (para metadata de fallback)
    return reduce_embeddings_to_3d(
        M,
        method="pca_tsne",
        normalize="auto",
        tsne_metric="cosine",
        tsne_perplexity=perplexity,
        tsne_perplexity_strategy="auto",
        tsne_auto_min_perplexity=35.0,
        tsne_auto_max_perplexity=120.0,
        tsne_auto_scale=45.0,
        random_state=random_state,
        return_metadata=return_metadata,
        pca_whiten=pca_whiten,
        pca_max_components=100,
        output_dims=3,
    )


def evaluate_projection_quality(
    original: Sequence[Sequence[float]] | np.ndarray,
    projected: Sequence[Sequence[float]] | np.ndarray,
    *,
    k: int = 15,
    metric: Literal["cosine", "euclidean"] = "cosine",
) -> Dict[str, Any]:
    """Compute simple quality metrics for a projection.

    Metrics:
      - neighbor_overlap: average Jaccard of top-k neighbor sets (excluding self)
      - trustworthiness (if sklearn>=1.1 has manifold.trustworthiness)

    Notes: For large n this can be O(n^2); no sampling here (user can pre-subset).
    """
    if isinstance(original, np.ndarray):
        X = original.astype(np.float32, copy=False)
    else:
        X = np.asarray(list(original), dtype=np.float32)
    if isinstance(projected, np.ndarray):
        Y = projected.astype(np.float32, copy=False)
    else:
        Y = np.asarray(list(projected), dtype=np.float32)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Original and projected must have same number of samples")
    n = X.shape[0]
    if n == 0:
        return {"n": 0}
    k_eff = min(k, n - 1)
    if k_eff <= 0:
        return {"n": n, "neighbor_overlap": None}

    # Distance / similarity matrices
    if metric == "cosine":
        Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        Yn = Y / np.clip(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12, None)
        # Higher is closer, convert to distances 1 - cos for ranking consistency
        dist_X = 1 - (Xn @ Xn.T)
        dist_Y = 1 - (Yn @ Yn.T)
    else:
        # Euclidean distances
        def pdist_m(a: np.ndarray) -> np.ndarray:
            aa = np.sum(a * a, axis=1, keepdims=True)
            return np.sqrt(np.maximum(aa - 2 * (a @ a.T) + aa.T, 0.0))
        dist_X = pdist_m(X)
        dist_Y = pdist_m(Y)

    np.fill_diagonal(dist_X, np.inf)
    np.fill_diagonal(dist_Y, np.inf)
    idx_X = np.argpartition(dist_X, k_eff, axis=1)[:, :k_eff]
    idx_Y = np.argpartition(dist_Y, k_eff, axis=1)[:, :k_eff]
    overlaps = []
    for i in range(n):
        set_X = set(idx_X[i].tolist())
        set_Y = set(idx_Y[i].tolist())
        inter = len(set_X & set_Y)
        union = len(set_X | set_Y)
        overlaps.append(inter / union if union else 0.0)
    neighbor_overlap = float(np.mean(overlaps))

    trust = None
    try:  # trustworthiness (optional)
        from sklearn.manifold import trustworthiness
        trust = float(trustworthiness(X, Y, n_neighbors=k_eff))
    except Exception:  # pragma: no cover
        pass

    return {"n": n, "k": k_eff, "neighbor_overlap": neighbor_overlap, "trustworthiness": trust}