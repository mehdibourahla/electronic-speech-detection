import tensorflow as tf
from tensorflow.keras import layers


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerClassifier(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        ff_dim,
        input_shape,
        num_classes,
        rate=0.1,
    ):
        super(TransformerClassifier, self).__init__()
        self.num_layers = num_layers
        self.enc_layers = [
            TransformerEncoder(embed_dim, num_heads, ff_dim, rate)
            for _ in range(num_layers)
        ]

        self.average_pool = layers.GlobalAveragePooling1D()
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = inputs
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        x = self.average_pool(x)
        return self.classifier(x)
