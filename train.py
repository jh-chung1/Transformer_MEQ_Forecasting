import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from utils.DataPreprocessing import DataPreprocessor
from utils.TransformerArchitecture import transformer_with_uncertainty
from utils.VizResults import (
    save_metrics_json,
    plot_training_history,
    plot_temporal_uncertainty,
    plot_spatial_uncertainty,
    compute_metrics,
    rescale_predictions,
    rescale_sigma
)

def parse_args():
    p = argparse.ArgumentParser()
    p.set_defaults(normalize_local=True)
    p.add_argument('--no-normalize_local', dest='normalize_local', action='store_false')
    p.add_argument('--data_dir',   type=str, default='./data/')
    p.add_argument('--out_dir',    type=str, default='./results/')
    p.add_argument('--n_start',    type=int, default=100)
    p.add_argument('--n_future',   type=int, default=5)
    p.add_argument('--normalize_local', action='store_true')
    p.add_argument('--num_heads',  type=int, default=4)
    p.add_argument('--ff_dim',     type=int, default=32)
    p.add_argument('--dropout_rate', type=float, default=0.3)
    p.add_argument('--beta',       type=float, default=10)
    p.add_argument('--lamda',      type=float, default=1e-1)
    p.add_argument('--epochs',     type=int, default=30)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--model_out',  type=str, default='./results/best_model.h5')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Data prep
    dp = DataPreprocessor(
        n_start=args.n_start,
        n_future=args.n_future,
        normalize_local=args.normalize_local
    )
    stim3 = dp.data_load(os.path.join(args.data_dir, 'Stim_3.npy'))
    stim4 = dp.data_load(os.path.join(args.data_dir, 'Stim_4.npy'))
    stim5 = dp.data_load(os.path.join(args.data_dir, 'Stim_5.npy'))

    trainX_list, trainY, train_scalers, _ = dp.prepare_data(stim3)
    valX_list,   valY,   val_scalers,   _ = dp.prepare_data(stim4)
    testX_list,  testY,  test_scalers,  _ = dp.prepare_data(stim5)

    trainX = dp.pad_windows(trainX_list)
    valX   = dp.pad_windows(valX_list,   maxlen=trainX.shape[1])
    testX  = dp.pad_windows(testX_list,  maxlen=trainX.shape[1])

    # 2) Build & train model
    feature_dim = trainX.shape[2]
    n_future    = trainY.shape[1]
    output_dim  = trainY.shape[2]

    model = transformer_with_uncertainty(
        feature_dim=feature_dim,
        output_dim=output_dim,
        n_future=n_future,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout_rate=args.dropout_rate,
        beta=args.beta,
        lamda=args.lamda
    )
    ckpt  = ModelCheckpoint(args.model_out, monitor='val_loss', save_best_only=True)
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        trainX, trainY,
        validation_data=(valX, valY),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[ckpt, early]
    )
    np.save(os.path.join(args.out_dir, 'history.npy'), history.history)
    plot_training_history(history, args.out_dir)

    # 3) Predict
    train_preds = model.predict(trainX)
    val_preds   = model.predict(valX)
    test_preds  = model.predict(testX)

    mu_train    = train_preds[..., :output_dim]
    sigma_train = np.sqrt(np.exp(train_preds[..., output_dim:]))
    mu_val      = val_preds[...,   :output_dim]
    sigma_val   = np.sqrt(np.exp(val_preds[...,   output_dim:]))
    mu_test     = test_preds[...,  :output_dim]
    sigma_test  = np.sqrt(np.exp(test_preds[...,  output_dim:]))

    # 4) Compute metrics
    true_train_r = rescale_predictions(trainY,    train_scalers, feature_dim, dp.target_cols)
    sig_train_r  = rescale_sigma(     sigma_train, train_scalers, dp.target_cols)
    mu_train_r   = rescale_predictions(mu_train,   train_scalers, feature_dim, dp.target_cols)

    true_val_r   = rescale_predictions(valY,      val_scalers,   feature_dim, dp.target_cols)
    sig_val_r    = rescale_sigma(     sigma_val,   val_scalers,   dp.target_cols)
    mu_val_r     = rescale_predictions(mu_val,     val_scalers,   feature_dim, dp.target_cols)

    true_test_r  = rescale_predictions(testY,     test_scalers,  feature_dim, dp.target_cols)
    sig_test_r   = rescale_sigma(     sigma_test,  test_scalers,  dp.target_cols)
    mu_test_r    = rescale_predictions(mu_test,    test_scalers,  feature_dim, dp.target_cols)

    target_names = [dp.feature_cols[i] for i in dp.target_cols]
    all_metrics = {
        'train': compute_metrics(true_train_r, mu_train_r, target_names),
        'val':   compute_metrics(true_val_r,   mu_val_r,   target_names),
        'test':  compute_metrics(true_test_r,  mu_test_r,  target_names)
    }
    save_metrics_json(all_metrics, args.out_dir)

    # 5) Prepare “block” vs “sample” indices separately for each split
    block_train = np.arange(trainY.shape[0])
    sample_train = args.n_start + block_train * args.n_future

    block_val   = np.arange(valY.shape[0])
    sample_val   = args.n_start + block_val   * args.n_future

    block_test  = np.arange(testY.shape[0])
    sample_test  = args.n_start + block_test  * args.n_future

    splits = {
        'Train': (true_train_r, mu_train_r, sig_train_r, stim3['Date'], sample_train),
        'Val':   (true_val_r,   mu_val_r,   sig_val_r,   stim4['Date'], sample_val),
        'Test':  (true_test_r,  mu_test_r,  sig_test_r,  stim5['Date'], sample_test),
    }

    # 6) Visualization
    for label, (T_r, M_r, S_r, raw_dates, sample_idxs) in splits.items():
        full_dates = pd.to_datetime(raw_dates).reset_index(drop=True)

        last_offset = args.n_future - 1
        pred_pos    = sample_idxs + last_offset
        pred_dates  = full_dates.iloc[pred_pos]   # length = #blocks

        for i, name in enumerate(target_names):
            y_true_blk  = T_r[:,:,i]    # shape = (blocks, n_future)
            y_pred_blk  = M_r[:,:,i]
            y_sigma_blk = S_r[:,:,i]
            plot_temporal_uncertainty(
                pred_dates,
                y_true_blk,
                y_pred_blk,
                y_sigma_blk,
                var_name=name,
                dataset_label=label,
                n_future=args.n_future,
                out_dir=args.out_dir
            )

        offsets  = np.arange(args.n_future)
        pos_mat  = sample_idxs[:,None] + offsets[None,:]
        flat_pos = pos_mat.ravel()
        flat_dates = full_dates.iloc[flat_pos]

        true_p95  = T_r[:,:,2].ravel()
        pred_p95  = M_r[:,:,2].ravel()
        sig_p95   = S_r[:,:,2].ravel()
        true_p50  = T_r[:,:,3].ravel()
        pred_p50  = M_r[:,:,3].ravel()
        sig_p50   = S_r[:,:,3].ravel()

        plot_spatial_uncertainty(
            flat_dates,
            true_p95, pred_p95, sig_p95,
            true_p50, pred_p50, sig_p50,
            n_future=args.n_future,
            dataset_label=label,
            out_dir=args.out_dir
        )

if __name__ == '__main__':
    main()
