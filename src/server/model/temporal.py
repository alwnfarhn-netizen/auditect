"""
model/temporal.py  —  PointNet + LSTM Temporal Model
======================================================
Model utama untuk deteksi stimming: menggabungkan PointNet (ekstraksi
fitur per frame) dengan LSTM (analisis pola temporal antar frame).

Arsitektur 2-tahap:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Frame 1     Frame 2     Frame 3   …  Frame 60                      │
  │  (33,3)      (33,3)      (33,3)        (33,3)                       │
  │     ↓           ↓           ↓              ↓                        │
  │  PointNet   PointNet   PointNet       PointNet   (shared weights)   │
  │     ↓           ↓           ↓              ↓                        │
  │  emb(256)  emb(256)   emb(256)      emb(256)                       │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 ↓
                          LSTM (128 units)
                                 ↓
                          LSTM (64 units)
                                 ↓
                          Dense → Softmax
                                 ↓
                    [normal, body_rocking, hand_flapping,
                     head_banging, spinning]

Keunggulan vs CNN-LSTM biasa:
  • PointNet tidak terpengaruh ukuran tubuh, posisi, atau latar kelas
  • Shared PointNet weights → parameter lebih efisien
  • LSTM menangkap dinamika temporal (kecepatan, ritme, repetisi)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from model.pointnet import build_pointnet_encoder, CLASSES, N_POINTS, N_DIMS, N_CLASSES


WINDOW_SIZE = 60     # jumlah frame per window (~2 detik @30FPS)
EMBED_DIM   = 256    # dimensi embedding PointNet per frame


def build_pointnet_lstm(
    window_size:  int   = WINDOW_SIZE,
    n_points:     int   = N_POINTS,
    n_dims:       int   = N_DIMS,
    embed_dim:    int   = EMBED_DIM,
    n_classes:    int   = N_CLASSES,
    lstm_units:   int   = 128,
    dropout:      float = 0.3,
) -> Model:
    """
    Bangun model PointNet + LSTM untuk klasifikasi stimming.

    Input  : (window_size, n_points, n_dims)  — sequence of pose frames
    Output : (n_classes,) softmax probabilities
    """
    # ── Shared PointNet Encoder ────────────────────────────────────────────
    encoder = build_pointnet_encoder(n_points, n_dims, embed_dim)

    # ── Input: sequence of point clouds ────────────────────────────────────
    sequence_input = Input(
        shape=(window_size, n_points, n_dims),
        name="pose_sequence"
    )

    # ── Apply PointNet to each frame (TimeDistributed) ─────────────────────
    # TimeDistributed membungkus encoder agar diaplikasikan per time-step
    embeddings = layers.TimeDistributed(
        encoder,
        name="shared_pointnet"
    )(sequence_input)   # output: (batch, window_size, embed_dim)

    # ── Temporal Analysis dengan LSTM ──────────────────────────────────────
    x = layers.LSTM(lstm_units, return_sequences=True, name="lstm_1")(embeddings)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units // 2, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(dropout)(x)

    # ── Fully Connected Head ───────────────────────────────────────────────
    x = layers.Dense(128, activation="relu", name="fc_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64,  activation="relu", name="fc_2")(x)

    output = layers.Dense(n_classes, activation="softmax", name="output")(x)

    # ── Compile ────────────────────────────────────────────────────────────
    model = Model(inputs=sequence_input, outputs=output, name="PointNet-LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model,
    X_train, y_train,
    X_val,   y_val,
    epochs:     int = 80,
    batch_size: int = 16,
    save_path:  str = "weights/stimming_model.keras",
) -> tf.keras.callbacks.History:
    """
    Training dengan callbacks standar:
      - EarlyStopping (patience=15)
      - ModelCheckpoint (simpan model terbaik)
      - ReduceLROnPlateau (turunkan LR jika stuck)
      - TensorBoard (log training untuk visualisasi)
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="logs/",
            histogram_freq=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history
