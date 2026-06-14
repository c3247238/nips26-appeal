"""
necessary_compositionality: Principal angle analysis for latent space geometry.
Local implementation for lewm-generalization project.
Implements principal angle computation between subspaces (Uselis et al. 2026 style).
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute principal angles between two subspaces spanned by A and B.
    
    Args:
        A: (n, k1) matrix — columns span first subspace
        B: (n, k2) matrix — columns span second subspace
    
    Returns:
        angles: array of min(k1, k2) principal angles in radians
    """
    # Orthonormalize
    Q_A, _ = np.linalg.qr(A)
    Q_B, _ = np.linalg.qr(B)
    
    # Compute SVD of Q_A^T Q_B
    M = Q_A.T @ Q_B
    sigma = np.linalg.svd(M, compute_uv=False)
    
    # Clamp for numerical stability
    sigma = np.clip(sigma, -1.0, 1.0)
    angles = np.arccos(sigma)
    return angles


def mean_cosine_similarity(A: np.ndarray, B: np.ndarray, n_components: int = 8) -> float:
    """
    Compute mean cosine similarity between principal directions of A and B subspaces.
    
    Args:
        A: (n, d) embedding matrix for group A
        B: (n, d) embedding matrix for group B
        n_components: number of principal components to use
    
    Returns:
        mean cosine similarity (scalar)
    """
    from sklearn.decomposition import PCA
    
    k = min(n_components, A.shape[0] - 1, B.shape[0] - 1, A.shape[1])
    if k < 1:
        return 0.0
    
    pca_A = PCA(n_components=k)
    pca_B = PCA(n_components=k)
    
    pca_A.fit(A)
    pca_B.fit(B)
    
    # Principal components as subspace bases
    basis_A = pca_A.components_.T  # (d, k)
    basis_B = pca_B.components_.T  # (d, k)
    
    angles = principal_angles(basis_A, basis_B)
    return float(np.cos(angles).mean())


def principal_angle_matrix(
    embeddings_by_factor: dict,
    n_components: int = 8
) -> np.ndarray:
    """
    Compute matrix of mean cosine similarities between all pairs of factor subspaces.
    
    Args:
        embeddings_by_factor: dict mapping factor_name -> (n, d) embeddings
        n_components: number of PCA components for each factor subspace
    
    Returns:
        matrix: (n_factors, n_factors) cosine similarity matrix
    """
    factors = list(embeddings_by_factor.keys())
    n = len(factors)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                cos_sim = mean_cosine_similarity(
                    embeddings_by_factor[factors[i]],
                    embeddings_by_factor[factors[j]],
                    n_components=n_components
                )
                matrix[i, j] = cos_sim
    
    return matrix


def displacement_consistency(
    embeddings_by_level: List[np.ndarray],
    n_reference_points: int = 3
) -> float:
    """
    Compute displacement vector consistency (parallelogram test).
    
    For each factor, computes centroid delta-vectors at multiple reference points
    and measures their cosine consistency.
    
    Args:
        embeddings_by_level: list of (n_i, d) embeddings for each factor level
        n_reference_points: number of reference points to sample
    
    Returns:
        consistency score in [0, 1]
    """
    if len(embeddings_by_level) < 2:
        return 0.0
    
    # Compute centroids per level
    centroids = [emb.mean(axis=0) for emb in embeddings_by_level]
    
    # Compute delta vectors between consecutive levels
    deltas = []
    for i in range(len(centroids) - 1):
        delta = centroids[i+1] - centroids[i]
        norm = np.linalg.norm(delta)
        if norm > 1e-8:
            deltas.append(delta / norm)
    
    if len(deltas) < 2:
        return 1.0  # Only one delta, trivially consistent
    
    # Compute pairwise cosine similarities
    cos_sims = []
    for i in range(len(deltas)):
        for j in range(i+1, len(deltas)):
            cos_sim = np.dot(deltas[i], deltas[j])
            cos_sims.append(cos_sim)
    
    return float(np.mean(cos_sims))
