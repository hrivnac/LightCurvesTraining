# finetune_lsst_supervised.py (skeleton)
from __future__ import annotations
import os, json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reuse these custom layers if you used them
@keras.saving.register_keras_serializable()
class LabelVectorHead(layers.Layer):
    def __init__(self, num_labels, d_model, use_cosine=True, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = int(num_labels)
        self.d_model = int(d_model)
        self.use_cosine = bool(use_cosine)

    def build(self, input_shape):
        self.L = self.add_weight(shape=(self.num_labels, self.d_model),
                                 initializer="glorot_uniform", trainable=True, name="label_vectors")
        self.b = self.add_weight(shape=(self.num_labels,), initializer="zeros", trainable=True, name="label_bias")

    def call(self, z):
        if self.use_cosine:
            z_norm = tf.nn.l2_normalize(z, axis=-1)
            L_norm = tf.nn.l2_normalize(self.L, axis=-1)
            return tf.matmul(z_norm, L_norm, transpose_b=True) + self.b
        return tf.matmul(z, self.L, transpose_b=True) + self.b

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_labels": self.num_labels, "d_model": self.d_model, "use_cosine": self.use_cosine})
        return cfg

def build_classifier_from_encoder(encoder: keras.Model, num_labels: int):
    # encoder outputs {"seq":..., "emb":...}
    z = encoder.outputs["emb"]
    logits = LabelVectorHead(num_labels, z.shape[-1], use_cosine=True, name="label_head")(z)
    probs = layers.Activation("sigmoid", name="probs")(logits)
    return keras.Model(inputs=encoder.inputs, outputs=probs, name="lsst_classifier")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, help="Path to encoder.keras from pretraining")
    ap.add_argument("--label2id", required=True, help="label2id.json (when you have labels)")
    ap.add_argument("--out", default="runs/lsst_finetune/model.keras")
    args = ap.parse_args()

    with open(args.label2id, "r") as f:
        label2id = json.load(f)
    num_labels = len(label2id)

    encoder = tf.keras.models.load_model(args.encoder, compile=False)
    clf = build_classifier_from_encoder(encoder, num_labels)

    # Optionally freeze encoder at first:
    # for layer in encoder.layers: layer.trainable = False

    clf.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.AUC(curve="PR",  multi_label=True, name="auc_pr"),
            keras.metrics.AUC(curve="ROC", multi_label=True, name="auc_roc"),
        ],
    )

    # TODO: build labeled train_ds/val_ds from LSST JSON + label source, then:
    # clf.fit(train_ds, validation_data=val_ds, ...)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    clf.save(args.out)
    print("Saved (untrained) classifier scaffold to:", args.out)

if __name__ == "__main__":
    main()
