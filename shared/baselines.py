import numpy as np
from .channels import ula_corr


# ─── LS ───────────────────────────────────────────────────────
def ls_estimate(h_noisy_last):
    return h_noisy_last.copy()


# ─── MMSE oracle ──────────────────────────────────────────────
def mmse_estimate(h_noisy_last, snr_db, Nt):
    s2  = 10**(-snr_db/10)
    Hls = h_noisy_last[..., 0] + 1j*h_noisy_last[..., 1]
    R   = ula_corr(Nt).real.astype(np.float64)
    W   = R @ np.linalg.inv(R + s2*np.eye(Nt))
    Hm  = (W @ Hls.real.T).T + 1j*(W @ Hls.imag.T).T
    return np.stack([Hm.real, Hm.imag], axis=-1).astype(np.float32)


# ─── Sa-DLCS ISTA refinement ──────────────────────────────────
def sadlcs_postprocess(pred, h_noisy_last, lam=0.05, K=3, mu=0.5):
    h = pred.copy()
    for _ in range(K):
        h = h + mu * (h_noisy_last - h)
        h = np.sign(h) * np.maximum(np.abs(h) - lam, 0)
    return h


# ─── CNN-Sparse mask ──────────────────────────────────────────
def cnnsparse_mask(h_noisy_last, k=500):
    mag = np.abs(h_noisy_last[...,0] + 1j*h_noisy_last[...,1])
    flat = mag.flatten()
    if k >= len(flat): return np.ones_like(h_noisy_last)
    thresh = np.partition(flat, -k)[-k]
    mask2d = (mag >= thresh).astype(np.float32)
    return np.stack([mask2d, mask2d], axis=-1)


# ─── SNR conditioning helper (NCRN) ───────────────────────────
def add_snr_channel(h_noisy_seq, snr_db):
    T, R, Nt2, _ = h_noisy_seq.shape
    snr_norm = np.full((T,R,Nt2,1), snr_db/30.0, dtype=np.float32)
    return np.concatenate([h_noisy_seq, snr_norm], axis=-1)


# ─── 3-regime SNR-aware blend (NCRN inference) ────────────────
def snr_aware_blend(raw_residual, h_noisy_last, snr_db_known):
    s = float(snr_db_known)
    if s >= 5.0:
        t = 1.0/(1.0 + np.exp(-(s-10.0)/2.5)); alpha = 0.85 + 0.15*t
    elif s >= 0.0:
        t = 1.0/(1.0 + np.exp(-(s-2.5)/2.5));  alpha = 0.55 + 0.30*t
    else:
        t = 1.0/(1.0 + np.exp(-(s+7.5)/2.5));  alpha = 0.30 + 0.25*t
    alpha = float(np.clip(alpha, 0.30, 1.0))
    return h_noisy_last + alpha * raw_residual
