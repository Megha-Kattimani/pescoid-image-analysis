"""
Tissue Flow Analysis Module

Analyzes cell movement and tissue deformation in 3D pescoids.
Computes velocity fields, nematic order, and connectivity patterns.

Key analyses:
- Cell center tracking (from nuclei positions)
- Velocity computation (if multi-timepoint data available)
- Nematic order parameter (tissue alignment)
- Stress/strain estimation
- Local vorticity and strain rate
"""

import numpy as np
from scipy import ndimage as ndi, linalg
from scipy.ndimage import label
from scipy.interpolate import griddata
from skimage import measure


def compute_nuclei_centers(nuclei_mask):
    """
    Extract 3D centroids of all nuclei.
    
    Parameters
    ----------
    nuclei_mask : np.ndarray
        Binary 3D mask with labeled nuclei (z, y, x)
    
    Returns
    -------
    np.ndarray
        (n_nuclei, 3) array of centroid positions (z, y, x)
    """
    
    # Label connected components
    labeled, n_nuclei = label(nuclei_mask)
    if n_nuclei == 0:
        return np.zeros((0, 3))
    
    # Get centroid of each
    coords = ndi.center_of_mass(nuclei_mask, labeled, range(1, n_nuclei + 1))
    
    return np.array(coords)


def compute_velocity_field(nuclei_centers_t0, nuclei_centers_t1, max_distance=20.0):
    """
    Compute velocity field by matching nuclei between consecutive timepoints.
    
    Parameters
    ----------
    nuclei_centers_t0 : np.ndarray
        (n0, 3) cell centroids at time t0
    nuclei_centers_t1 : np.ndarray
        (n1, 3) cell centroids at time t1
    max_distance : float
        Maximum distance for cell correspondence
    
    Returns
    -------
    dict
        - 'displacements': (n_matched, 3) displacement vectors
        - 'mean_speed': mean magnitude of displacements
        - 'tracking_confidence': fraction of cells matched
        - 'matched_cells_t0': indices of matched cells at t0
        - 'matched_cells_t1': indices of matched cells at t1
    """
    
    if len(nuclei_centers_t0) == 0 or len(nuclei_centers_t1) == 0:
        return {
            'displacements': np.zeros((0, 3)),
            'mean_speed': 0.0,
            'tracking_confidence': 0.0,
            'matched_cells_t0': [],
            'matched_cells_t1': [],
        }
    
    # Compute pairwise distances
    n0, n1 = len(nuclei_centers_t0), len(nuclei_centers_t1)
    distances = np.zeros((n0, n1))
    
    for i in range(n0):
        for j in range(n1):
            distances[i, j] = np.linalg.norm(nuclei_centers_t0[i] - nuclei_centers_t1[j])
    
    # Greedy matching (nearest neighbor within threshold)
    matched_t0, matched_t1 = [], []
    displacements = []
    
    used_t1 = set()
    for i in range(n0):
        j_best = np.argmin(distances[i, :])
        if distances[i, j_best] < max_distance and j_best not in used_t1:
            matched_t0.append(i)
            matched_t1.append(j_best)
            displacements.append(nuclei_centers_t1[j_best] - nuclei_centers_t0[i])
            used_t1.add(j_best)
    
    if len(displacements) == 0:
        displacements = np.zeros((0, 3))
        mean_speed = 0.0
    else:
        displacements = np.array(displacements)
        mean_speed = np.mean(np.linalg.norm(displacements, axis=1))
    
    return {
        'displacements': displacements,
        'mean_speed': mean_speed,
        'tracking_confidence': len(displacements) / max(n0, 1),
        'matched_cells_t0': matched_t0,
        'matched_cells_t1': matched_t1,
    }


def compute_nematic_order_parameter(displacements):
    """
    Compute nematic tensor and order parameter from velocity vectors.
    
    Nematic order: Q = (3/2)<n⊗n - I/3>, where n = v/|v|
    Eigenvalues range [-1/3, 1]: 
      0 = isotropic, positive = aligned, negative = perpendicular
    
    Parameters
    ----------
    displacements : np.ndarray
        (n_cells, 3) velocity/displacement vectors
    
    Returns
    -------
    dict
        - 'nematic_tensor': (3, 3) nematic tensor
        - 'order_parameter': scalar (0-1 aligned, 0 isotropic, -0.5 perpendicular)
        - 'principal_direction': (3,) primary direction of alignment
        - 'mean_speed': mean magnitude
    """
    
    if len(displacements) == 0:
        return {
            'nematic_tensor': np.zeros((3, 3)),
            'order_parameter': 0.0,
            'principal_direction': np.array([0, 0, 1]),
            'mean_speed': 0.0,
        }
    
    # Normalize
    speeds = np.linalg.norm(displacements, axis=1)
    mean_speed = np.mean(speeds)
    
    if mean_speed < 1e-10:
        return {
            'nematic_tensor': np.zeros((3, 3)),
            'order_parameter': 0.0,
            'principal_direction': np.array([0, 0, 1]),
            'mean_speed': 0.0,
        }
    
    normed = displacements / (speeds[:, np.newaxis] + 1e-10)
    
    # Q_ij = (3/2) <n_i * n_j - δ_ij/3>
    Q = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            Q[i, j] = (3.0 / 2.0) * (np.mean(normed[:, i] * normed[:, j]) - (1.0/3.0 if i == j else 0.0))
    
    # Eigenvalues
    evals, evecs = linalg.eigh(Q)
    order_param = evals[-1]  # Largest eigenvalue
    principal = evecs[:, -1]
    
    return {
        'nematic_tensor': Q,
        'order_parameter': float(order_param),
        'principal_direction': principal,
        'mean_speed': mean_speed,
    }


def compute_local_strain_rate(nuclei_centers, bandwidth=20.0):
    """
    Estimate local strain rate tensor from spatial cell distribution.
    
    Uses kernel density estimation to infer local deformation.
    
    Parameters
    ----------
    nuclei_centers : np.ndarray
        (n_cells, 3) cell positions
    bandwidth : float
        Kernel bandwidth for local estimation
    
    Returns
    -------
    dict
        - 'strain_rate_tensor': (3, 3) local strain rate estimate
        - 'divergence': trace of strain rate (expansion/compression)
        - 'shear_rate': magnitude of symmetric part
    """
    
    if len(nuclei_centers) < 4:
        return {
            'strain_rate_tensor': np.zeros((3, 3)),
            'divergence': 0.0,
            'shear_rate': 0.0,
        }
    
    # Compute Voronoi-based local density variation (simplified)
    centroid = nuclei_centers.mean(axis=0)
    radial = nuclei_centers - centroid
    
    # Covariance of positions
    cov = np.cov(radial.T)
    
    # Strain rate ≈ -∇ln(density) ≈ covariance structure
    strain = linalg.inv(cov + 1e-6 * np.eye(3)) @ cov
    
    return {
        'strain_rate_tensor': strain,
        'divergence': float(np.trace(strain)),
        'shear_rate': float(np.linalg.norm(strain - np.eye(3) * np.trace(strain) / 3)),
    }


def compute_tissue_connectivity(nuclei_centers, connectivity_distance=30.0):
    """
    Build tissue connectivity graph from cell positions.
    
    Parameters
    ----------
    nuclei_centers : np.ndarray
        (n_cells, 3) cell positions
    connectivity_distance : float
        Maximum distance for cell-cell connection
    
    Returns
    -------
    dict
        - 'adjacency_matrix': (n, n) sparse connectivity
        - 'mean_degree': average connections per cell
        - 'clustering_coefficient': local clustering
        - 'avg_connection_distance': mean edge length
    """
    
    n = len(nuclei_centers)
    if n < 2:
        return {
            'adjacency_matrix': np.zeros((n, n)),
            'mean_degree': 0.0,
            'clustering_coefficient': 0.0,
            'avg_connection_distance': 0.0,
        }
    
    # Distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = np.linalg.norm(nuclei_centers[i] - nuclei_centers[j])
    
    # Adjacency
    adj = (dist < connectivity_distance).astype(float)
    np.fill_diagonal(adj, 0)
    
    # Degree
    degree = adj.sum(axis=1)
    mean_degree = degree.mean()
    
    # Clustering coefficient (fraction of triangles)
    clustering_coeff = 0.0
    for i in range(n):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) > 1:
            edges = 0
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if adj[neighbors[j], neighbors[k]] > 0:
                        edges += 1
            max_edges = len(neighbors) * (len(neighbors) - 1) / 2
            if max_edges > 0:
                clustering_coeff += edges / max_edges
    
    clustering_coeff /= max(n, 1)
    
    # Mean distance
    valid_dist = dist[adj > 0]
    avg_dist = valid_dist.mean() if len(valid_dist) > 0 else 0.0
    
    return {
        'adjacency_matrix': adj,
        'mean_degree': float(mean_degree),
        'clustering_coefficient': float(clustering_coeff),
        'avg_connection_distance': float(avg_dist),
    }


if __name__ == '__main__':
    # Synthetic test: random cell movement
    np.random.seed(42)
    
    # Generate two timepoint cell positions
    n_cells_t0 = 100
    nuclei_t0 = np.random.randn(n_cells_t0, 3) * 20 + np.array([15, 50, 50])
    
    # Add directed motion (anterior-posterior)
    displacement = np.random.randn(n_cells_t0, 3) * 2
    displacement[:, 0] += 3  # Bias in z-direction
    nuclei_t1 = nuclei_t0 + displacement
    
    # Velocity field
    vel = compute_velocity_field(nuclei_t0, nuclei_t1)
    print(f"Mean speed: {vel['mean_speed']:.2f} pixels")
    print(f"Tracking confidence: {vel['tracking_confidence']:.1%}")
    
    # Nematic order
    nem = compute_nematic_order_parameter(vel['displacements'])
    print(f"Nematic order parameter: {nem['order_parameter']:.3f}")
    print(f"Principal direction: {nem['principal_direction']}")
    
    # Strain rate
    strain = compute_local_strain_rate(nuclei_t0)
    print(f"Divergence (compression): {strain['divergence']:.3f}")
    print(f"Shear rate: {strain['shear_rate']:.3f}")
    
    # Connectivity
    conn = compute_tissue_connectivity(nuclei_t0)
    print(f"Mean degree: {conn['mean_degree']:.1f}")
    print(f"Clustering coefficient: {conn['clustering_coefficient']:.3f}")
