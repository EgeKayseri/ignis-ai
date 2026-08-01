"""
Yangın Büyüme Modeli — U-Net Segmentasyon
=========================================
14 kanallı yamayı (bugünün çevresel koşulları + yangın maskesi) alır ve
ertesi günün YANGIN OLASILIK HARİTASINI (piksel başına 0-1) üretir.

Eski `yangin_model.keras` (Dense NN, tek 0/1 risk çıktısı) bu segmentasyon
modeliyle değiştirilir. TensorFlow importları fonksiyon içinde yapılır ki bu
dosya TF kurulu olmadan da içe aktarılabilsin.
"""

from __future__ import annotations
from config import (
    SPREAD_MODEL_CONFIG, SPREAD_LEARNING_RATE, SPREAD_POS_WEIGHT,
    MODEL_PATCH, SPREAD_N_CHANNELS, SPREAD_FOCAL_GAMMA, SPREAD_FOCAL_ALPHA,
)


# ------------------------------------------------------------------
# U-Net blokları
# ------------------------------------------------------------------
def _conv_block(x, filters, dropout):
    from tensorflow.keras import layers
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x


def build_unet(input_shape=None, base_filters=None, depth=None,
               dropout=None, final_activation=None):
    """Parametrik U-Net. Varsayılanlar config.SPREAD_MODEL_CONFIG'ten gelir."""
    from tensorflow.keras import layers, Model

    cfg = SPREAD_MODEL_CONFIG
    input_shape = input_shape or cfg["input_shape"]
    base_filters = base_filters or cfg["base_filters"]
    depth = depth or cfg["depth"]
    dropout = cfg["dropout"] if dropout is None else dropout
    final_activation = final_activation or cfg["final_activation"]

    inputs = layers.Input(shape=input_shape)
    x = inputs
    skips = []

    # Encoder
    for d in range(depth):
        f = base_filters * (2 ** d)
        x = _conv_block(x, f, dropout)
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)

    # Bottleneck
    x = _conv_block(x, base_filters * (2 ** depth), dropout)

    # Decoder
    for d in reversed(range(depth)):
        f = base_filters * (2 ** d)
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)
        x = layers.Concatenate()([x, skips[d]])
        x = _conv_block(x, f, dropout)

    # dtype="float32": mixed_float16 politikası açıkken bile çıktı ve kayıp float32
    # kalır (sayısal kararlılık). GPU'da mixed precision güvenle kullanılabilir.
    outputs = layers.Conv2D(1, 1, activation=final_activation,
                            dtype="float32", name="fire_next")(x)
    return Model(inputs, outputs, name="fire_spread_unet")


# ------------------------------------------------------------------
# Kayıp — seyrek yangın pikselleri için ağırlıklı / focal
# ------------------------------------------------------------------
def weighted_bce(pos_weight=SPREAD_POS_WEIGHT):
    """Yangın pikselleri az olduğundan pozitif sınıfa ağırlık ver."""
    import tensorflow as tf

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce = -(pos_weight * y_true * tf.math.log(y_pred)
                + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(bce)
    return loss


def focal_loss(gamma=SPREAD_FOCAL_GAMMA, alpha=SPREAD_FOCAL_ALPHA):
    """İkili focal loss — dengesiz maskeler için. Parametreler config'ten."""
    import tensorflow as tf

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        a = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        return tf.reduce_mean(-a * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    return loss


def compile_model(model=None, loss="focal", lr=SPREAD_LEARNING_RATE):
    from tensorflow.keras import optimizers, metrics
    model = model or build_unet()
    loss_fn = focal_loss() if loss == "focal" else weighted_bce()
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss=loss_fn,
        metrics=[metrics.AUC(name="auc", curve="PR"),
                 metrics.Precision(name="precision"),
                 metrics.Recall(name="recall")],
    )
    return model


if __name__ == "__main__":
    # Mimariyi özetle (TF kuruluysa)
    try:
        m = build_unet()
        m.summary()
        print(f"\nGirdi: {MODEL_PATCH}x{MODEL_PATCH}x{SPREAD_N_CHANNELS} "
              f"-> Cikti: {MODEL_PATCH}x{MODEL_PATCH}x1 (ertesi gun yangin olasiligi)")
    except Exception as e:
        print(f"TensorFlow gerekli: {e}")
