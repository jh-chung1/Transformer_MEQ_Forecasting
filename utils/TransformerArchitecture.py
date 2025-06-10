import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Masking, MultiHeadAttention, Dropout,
    LayerNormalization, Dense, GlobalAveragePooling1D,
    Reshape, Concatenate
)
from tensorflow.keras.models import Model

def transformer_with_uncertainty(
    feature_dim,
    output_dim,
    n_future=1,
    num_heads=4,
    ff_dim=32,
    dropout_rate=0.3,
    beta=1e-2,
    lamda = 1e-2
):
    """
    feature_dim: # input features
    output_dim: # targets per step
    n_future:   # steps to predict
    """
    inp = Input(shape=(None, feature_dim))
    x   = Masking(mask_value=0.0)(inp)

    # encoder block
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=feature_dim)(x, x)
    attn = Dropout(dropout_rate)(attn)
    attn = LayerNormalization(epsilon=1e-6)(attn + x)

    ff   = Dense(ff_dim, activation='relu')(attn)
    ff   = Dropout(dropout_rate)(ff)
    ff   = Dense(feature_dim)(ff)
    ff   = LayerNormalization(epsilon=1e-6)(ff + attn)

    # pool and predict all future steps at once
    pooled = GlobalAveragePooling1D()(ff)
    total_D = n_future * output_dim

    mu      = Dense(total_D)(pooled)
    log_var = Dense(total_D)(pooled)

    # reshape to (batch, n_future, output_dim)
    mu      = Reshape((n_future, output_dim))(mu)
    log_var = Reshape((n_future, output_dim))(log_var)

    out = Concatenate(axis=-1)([mu, log_var])  # shape = (batch, n_future, 2*output_dim)

    model = Model(inputs=inp, outputs=out)

    def nll(y_true, y_pred):
        # split μ and log‐variance
        mu_pred, logvar_pred = tf.split(y_pred, 2, axis=-1)
        # standard NLL term
        nll_term = 0.5 * (
            tf.exp(-logvar_pred) * tf.square(y_true - mu_pred)
            + logvar_pred
        )
        base_loss = tf.reduce_mean(nll_term) + beta * tf.reduce_mean(logvar_pred)

        # monotonicity penalties on two of the D targets:
        # target index 0 = Cum. MEQ, index 1 = Cum_log_moment
        penalties = []
        for k in (0, 1):
            # mu_pred[:, t, k], t=0…n_future-1
            d = mu_pred[:, 1:, k] - mu_pred[:, :-1, k]       # shape (batch, n_future-1)
            penalties.append(tf.reduce_sum(tf.nn.relu(-d)))
        mono_penalty = tf.add_n(penalties)

        return base_loss + lamda * mono_penalty

    model.compile(optimizer='adam', loss=nll)
    return model
