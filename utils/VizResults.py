import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error


def compute_metrics(y_true, y_pred, target_names=None):
    """
    y_true, y_pred: arrays of shape (N, n_future, D)
    Computes overall + per‐target R^2 & RMSE, flattening across both samples and horizons.
    """
    N, T, D = y_true.shape
    flat_true = y_true.reshape(-1, D)
    flat_pred = y_pred.reshape(-1, D)

    if target_names is None:
        target_names = [f'target_{i}' for i in range(D)]

    overall_r2 = r2_score(flat_true.flatten(), flat_pred.flatten())
    metrics = {'overall': {'r2': float(overall_r2)}}

    for i, name in enumerate(target_names):
        r2   = r2_score(flat_true[:, i], flat_pred[:, i])
        rmse = mean_squared_error(flat_true[:, i], flat_pred[:, i], squared=False)
        metrics[name] = {'r2': float(r2), 'rmse': float(rmse)}

    return metrics


def save_metrics_json(metrics, out_dir, filename='metrics.json'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


def plot_training_history(history, out_dir):
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #plt.savefig(os.path.join(out_dir, 'training_history.png'))
    #plt.close()


def plot_temporal_uncertainty(
    dates,
    y_true,
    y_pred,
    y_sigma,
    var_name,
    dataset_label,
    n_future,
    out_dir=None
):
    """
    plots only the final forecast step (n_future-th).
    """
    # take the last step
    true_k  = y_true[:, -1]
    pred_k  = y_pred[:, -1]
    sigma_k = y_sigma[:, -1]

    plt.figure(dpi=300, figsize=(6,4))
    plt.plot(dates, true_k,  'r', label='Obs')
    plt.plot(dates, pred_k,  'b', label=f'{n_future}-step pred')
    plt.fill_between(
        dates,
        pred_k - sigma_k,
        pred_k + sigma_k,
        alpha=0.2, label=r'$\pm \sigma$'
    )
    r2 = r2_score(true_k, pred_k)
    plt.title(fr'{dataset_label}: {var_name} ($R^2={r2:.3f}$)')
    plt.xlabel('Time'); plt.ylabel(var_name)
    plt.legend(); plt.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'{dataset_label}_{var_name}_nFuture_{n_future}.png'))
    plt.close()

def plot_spatial_uncertainty(
    dates,            # DatetimeIndex, length = blocks
    true_p95, pred_p95, sigma_p95,
    true_p50, pred_p50, sigma_p50,
    n_future,
    dataset_label,
    out_dir=None
):

    t95, p95, s95 = true_p95, pred_p95, sigma_p95
    t50, p50, s50 = true_p50, pred_p50, sigma_p50

    if out_dir:
        base = f"{dataset_label}_spatial_nFuture_{n_future}"

    plt.figure(dpi=300, figsize=(10,4))
    plt.fill_between(dates, p95 - s95, p95 + s95, alpha=0.2, label=r'$P_{95} \pm \sigma$')
    plt.plot(dates, t95, '-',   label=r'$P_{95}$ obs')
    plt.plot(dates, p95, '--',  label=r'$P_{95}$ pred')

    plt.fill_between(dates, p50 - s50, p50 + s50, alpha=0.2, label=r'$P_{50} \pm \sigma$')
    plt.plot(dates, t50, '-',   label=r'$P_{50}$ obs')
    plt.plot(dates, p50, '--',  label=r'$P_{50}$ pred')

    plt.title(f'{dataset_label}: MEQ Distances (nFuture={n_future})')
    plt.xlabel('Time'); plt.ylabel('Distance (m)')
    plt.legend(loc='upper left'); plt.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'{dataset_label}_spatial_unc_nFuture_{n_future}.png'))
    plt.close()



def rescale_predictions(preds, scalers, original_dim, target_cols):
    """
    preds: (N, n_future, D)
    returns same shape, but in original units.
    """
    N, T, D = preds.shape
    out = np.zeros_like(preds, dtype=np.float32)
    for i in range(N):
        scl = scalers[i]
        for t in range(T):
            full = np.zeros((1, original_dim), dtype=np.float32)
            full[0, target_cols] = preds[i, t]
            inv = scl.inverse_transform(full)[0]
            out[i, t] = inv[target_cols]
    return out


def rescale_sigma(sigmas, scalers, target_cols):
    """
    sigmas: (N, n_future, D)
    returns same shape, but converted via s_raw / scale.
    """
    N, T, D = sigmas.shape
    out = np.zeros_like(sigmas, dtype=np.float32)
    for i in range(N):
        scl = scalers[i]
        for t in range(T):
            out[i, t] = [
                float(sigmas[i, t, j] / scl.scale_[col])
                for j, col in enumerate(target_cols)
            ]
    return out
