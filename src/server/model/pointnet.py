"""
model/pointnet.py  —  PointNet Architecture
============================================
PointNet adalah arsitektur deep learning yang dirancang untuk mengklasifikasi
point cloud 3D. Repo referensi (ASL processing) menggunakannya untuk
mengklasifikasi 21 titik tangan → huruf ASL.

Di sini kita adaptasi untuk 33 titik tubuh → fitur embedding per frame,
yang kemudian akan diproses oleh temporal model (LSTM) untuk mendeteksi
pola gerakan stimming.

Paper asli PointNet: https://arxiv.org/abs/1612.00593

Arsitektur:
  Input (33, 3)
      ↓
  T-Net (input transform) — belajar transformasi 3×3 untuk invariansi rotasi
      ↓
  MLP: 64 → 64
      ↓
  T-Net (feature transform) — transformasi 64×64
      ↓
  MLP: 64 → 128 → 1024
      ↓
  Global Max Pooling → (1024,) — "global feature" yang order-invariant
      ↓
  MLP: 512 → 256 → 128 → output
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers


N_POINTS  = 33      # jumlah keypoint pose
N_DIMS    = 3       # x, y, z
N_CLASSES = 5       # normal, body_rocking, hand_flapping, head_banging, spinning

CLASSES = [
    "normal",
    "body_rocking",
    "hand_flapping",
    "head_banging",
    "spinning",
]


# ── T-Net: Spatial/Feature Transform Network ──────────────────────────────
def build_tnet(input_shape: tuple, dim: int) -> Model:
    """
    T-Net mempelajari matriks transformasi (dim × dim) untuk membuat model
    invariant terhadap rotasi dan translasi — konsep kunci dari PointNet.
    """
    inp = Input(shape=input_shape)

    x = layers.Conv1D(64,   1, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128,  1, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(1024, 1, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    # Output: matriks dim×dim sebagai vektor dim² + identitas sebagai bias
    bias_init = tf.keras.initializers.Constant(
        tf.eye(dim).numpy().flatten()
    )
    x = layers.Dense(
        dim * dim,
        kernel_initializer="zeros",
        bias_initializer=bias_init,
    )(x)

    transform = layers.Reshape((dim, dim))(x)

    return Model(inputs=inp, outputs=transform, name=f"T-Net-{dim}")


# ── PointNet: Fitur Embedding per Frame ──────────────────────────────────
def build_pointnet_encoder(
    n_points: int  = N_POINTS,
    n_dims:   int  = N_DIMS,
    embed_dim: int = 256,
) -> Model:
    """
    PointNet encoder: (N_POINTS, N_DIMS) → embedding vektor (embed_dim,)
    yang merangkum postur tubuh dalam satu frame.

    Output digunakan sebagai input per-step ke temporal LSTM model.
    """
    inp = Input(shape=(n_points, n_dims), name="keypoints")

    # ── Input Transform ──────────────────────────────────────────────────
    tnet_input = build_tnet((n_points, n_dims), n_dims)
    transform_3x3 = tnet_input(inp)
    x = layers.Lambda(
        lambda inputs: tf.matmul(inputs[0], inputs[1]),
        name="input_transform"
    )([inp, transform_3x3])

    # ── MLP Layer 1 ─────────────────────────────────────────────────────
    x = layers.Conv1D(64, 1, activation="relu", name="mlp1_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 1, activation="relu", name="mlp1_2")(x)
    x = layers.BatchNormalization()(x)

    # ── Feature Transform ────────────────────────────────────────────────
    tnet_feat = build_tnet((n_points, 64), 64)
    transform_64x64 = tnet_feat(x)
    x = layers.Lambda(
        lambda inputs: tf.matmul(inputs[0], inputs[1]),
        name="feature_transform"
    )([x, transform_64x64])

    # ── MLP Layer 2 ─────────────────────────────────────────────────────
    x = layers.Conv1D(64,   1, activation="relu", name="mlp2_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128,  1, activation="relu", name="mlp2_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(1024, 1, activation="relu", name="mlp2_3")(x)
    x = layers.BatchNormalization()(x)

    # ── Symmetric Function: Global Max Pool ─────────────────────────────
    # Ini yang membuat PointNet order-invariant: max pool dari semua titik
    global_feat = layers.GlobalMaxPooling1D(name="global_max_pool")(x)  # (1024,)

    # ── Bottleneck → embedding dimensi lebih kecil ───────────────────────
    embedding = layers.Dense(512, activation="relu")(global_feat)
    embedding = layers.BatchNormalization()(embedding)
    embedding = layers.Dropout(0.3)(embedding)
    embedding = layers.Dense(embed_dim, activation="relu", name="embedding")(embedding)
    embedding = layers.BatchNormalization()(embedding)

    return Model(inputs=inp, outputs=embedding, name="PointNet-Encoder")


# ── PointNet Classifier (untuk training langsung, tanpa LSTM) ─────────────
def build_pointnet_classifier(
    n_points:  int = N_POINTS,
    n_dims:    int = N_DIMS,
    n_classes: int = N_CLASSES,
    embed_dim: int = 256,
    dropout:   float = 0.4,
) -> Model:
    """
    PointNet lengkap untuk klasifikasi satu frame.
    Dipakai saat dataset cukup kecil atau sebagai baseline.
    """
    encoder = build_pointnet_encoder(n_points, n_dims, embed_dim)
    inp = Input(shape=(n_points, n_dims), name="keypoints")
    x   = encoder(inp)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inputs=inp, outputs=out, name="PointNet-Classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
