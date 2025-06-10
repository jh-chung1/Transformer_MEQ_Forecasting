import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(
        self,
        feature_cols=None,
        target_cols=None,
        n_start=100,
        n_future=1,
        normalize_local=True
    ):
        self.feature_cols = feature_cols or [
            'Cum_MEQ',
            'Pressure [MPa]',
            'Flow rate[L/min]',
            'Cum_log_seismic_moment',
            'percentile_95_distance_meters',
            'median_distance_meters'
        ]
        self.target_cols = target_cols or [0, 3, 4, 5]
        self.n_start = n_start
        self.n_future = n_future
        self.normalize_local = normalize_local

    @staticmethod
    def data_load(data_path: str) -> pd.DataFrame:
        arr = np.load(data_path, allow_pickle=True)
        df = pd.DataFrame({
            'Date': arr[:, 0],
            'Cum_MEQ': arr[:, 17],
            'Pressure [MPa]': arr[:, 22],
            'Flow rate[L/min]': arr[:, 23],
            'Cum_log_seismic_moment': arr[:, 19],
            'percentile_95_distance_meters': arr[:, 28],
            'median_distance_meters': arr[:, 27]
        })
        df['median_distance_meters'] = df['median_distance_meters'].fillna(0)
        return df

    def prepare_data(self, df: pd.DataFrame):
        data = df[self.feature_cols].to_numpy(dtype=np.float32)
        N, F = data.shape
        X_list, Y_list = [], []
        scalers = [] if self.normalize_local else None

        dates_list = []
        for i in range(self.n_start, N - self.n_future + 1, self.n_future):
            window = data[:i]
            if self.normalize_local:
                scaler = MinMaxScaler().fit(window)
                win_scaled = scaler.transform(window)
                tgt_block = data[i : i + self.n_future]              # (n_future, F)
                tgt_scaled = scaler.transform(tgt_block)[:, self.target_cols]  # (n_future, D)
                scalers.append(scaler)
            else:
                win_scaled = window
                tgt_scaled = data[i : i + self.n_future, self.target_cols]

            X_list.append(win_scaled)   # (i, F)
            Y_list.append(tgt_scaled)   # (n_future, D)
            dates_list.append(pd.to_datetime(df['Date'].iloc[i]))

        dates = pd.DatetimeIndex(dates_list)
        X = X_list
        Y = np.stack(Y_list, axis=0)    # (num_blocks, n_future, D)
        return X, Y, scalers, dates

    @staticmethod
    def pad_windows(X_list, maxlen=None):
        """
        Pad or truncate variable‐length windows to identical shape (pre‐padding/truncation).
        If a window is longer than maxlen, drop its earliest rows.
        """
        lengths = [x.shape[0] for x in X_list]
        if maxlen is None:
            maxlen = max(lengths)
        F = X_list[0].shape[1]
        X_pad = np.zeros((len(X_list), maxlen, F), dtype=np.float32)
        for i, x in enumerate(X_list):
            if x.shape[0] > maxlen:
                x_trunc = x[-maxlen:, :]
                X_pad[i] = x_trunc
            else:
                X_pad[i, maxlen - x.shape[0] :, :] = x
        return X_pad
