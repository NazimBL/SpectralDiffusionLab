import numpy as np
from scipy.signal import savgol_filter
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from sklearn.metrics.pairwise import cosine_similarity

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = sp.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L)).toarray()
    D = lam * D.T @ D
    w = np.ones(L)
    for _ in range(niter):
        W = sp.diags(w, 0)
        Z = splinalg.spsolve(W + D, w * y)
        w = p * (y > Z) + (1 - p) * (y < Z)
    return Z

def apply_baseline_correction(X, lam=1e5, p=0.01, niter=10):
    X_corrected = np.zeros_like(X)
    for i in range(X.shape[0]):
        baseline = baseline_als(X[i, :], lam=lam, p=p, niter=niter)
        X_corrected[i, :] = X[i, :] - baseline
    return X_corrected

def remove_quadratic_drift(signal):
    x = np.arange(len(signal))
    coeffs = np.polyfit(x, signal, 2)
    trend = np.polyval(coeffs, x)
    return signal - trend

def invert_preprocessing(synthetic_preprocessed, reference_raw, lam=1e5, p=0.01, niter=10):
    reference_bc = apply_baseline_correction(reference_raw, lam=lam, p=p, niter=niter)
    reference_sg = savgol_filter(reference_bc, window_length=5, polyorder=2, deriv=2, axis=1)
    reference_norm = np.linalg.norm(reference_sg, axis=1, keepdims=True)
    reference_norm[reference_norm == 0] = 1
    reference_normed = reference_sg / reference_norm
    reference_baseline = reference_raw - reference_bc

    sim_matrix = cosine_similarity(synthetic_preprocessed, reference_normed)
    best_matches = np.argmax(sim_matrix, axis=1)

    reconstructed_raw_like = []
    for i, syn in enumerate(synthetic_preprocessed):
        idx = best_matches[i]
        norm = reference_norm[idx]
        baseline = reference_baseline[idx]

        unnormalized = syn * norm
        first_integral = np.cumsum(unnormalized)
        second_integral = np.cumsum(first_integral)
        de_drifted = remove_quadratic_drift(second_integral)
        raw_approx = de_drifted + baseline

        # Final ALS baseline correction to remove curvature artifacts
        baseline_final = baseline_als(raw_approx.flatten(), lam=1e5, p=0.01, niter=10)
        final_corrected = raw_approx.flatten() - baseline_final

        reconstructed_raw_like.append(final_corrected)

    return np.array(reconstructed_raw_like)
