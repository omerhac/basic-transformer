import tensorflow as tf


def scaled_dot_product_attention(q, k, v):
    """Compute the scaled dot product attention according to Attention(Q,K,V) = softmax(QK^T/sqrt(d_model))V"""
    d_model = tf.cast(tf.shape(q)[-1], tf.float32)
    qkt = tf.matmul(q, k, transpose_b=True)  # QK^T

    attention_weights = tf.nn.softmax(qkt / tf.sqrt(d_model))

    # compute attention values
    attention_values = tf.matmul(attention_weights, v)

    return attention_values


class MultiHeadAttention:
    """Multi Head Attention layer as in MHA(Q,K,V) = concat(attention_head1, attention_head2, .., attention_headn)Wc,
    where attention_headi = attention(QWq, KWk, VWv). Wc, Wq, Wk and Wv are projections to d_c, d_q, d_k and d_v.
    """

    def __init__(self, heads, d_model):
        self._d_model = d_model
        self._heads = heads

        assert d_model % heads == 0, print('d_model must be divided by heads')
        self._depth = d_model // heads

        # define projections (Wq, Wk, Wv):
        # its better to project the whole matrices and then split the heads since it can be done with a single layer
        # per matrix and not |heads| layers per projection

        self._wq = tf.keras.layers.Dense(d_model)
        self._wk = tf.keras.layers.Dense(d_model)
        self._wv = tf.keras.layers.Dense(d_model)

        # define projection Wc for applying after the heads concatenation
        self._wc = tf.keras.layers.Dense(d_model)

    def _split_heads(self, X):
        """Split input matrix X to |heads| matrices.
        X shape is [batch_size, sequence_length, d_model], this function will return X_split with
        shape [batch_size, num_heads, sequence_lenth, depth]
        """

        batch_size = tf.shape(X)[0]

        # split last dimension to |heads| heads each with d_model/|heads| depth
        # as in X_split' shape is [batch_size, sequence_length, num_heads, depth]
        X_split_tag = tf.reshape(X, [batch_size, -1, self._heads, self._depth])

        # rearrange X_split' so sequence_length and depth will be the last dimensions for computing attention
        # between words dense representations later on -> X_split shape is [batch_size, num_heads, seq_length, depth]
        X_split = tf.transpose(X_split_tag, perm=[0, 2, 1, 3])

        return X_split

    def __call__(self, q, k, v):
        """Compute the Multy Head Attention of Q, K, V"""
        batch_size = q.shape[0]

        # project matrices
        q = self._wq(q)
        k = self._wk(k)
        v = self._wv(v)

        # split each matrix
        q_split = self._split_heads(q)
        k_split = self._split_heads(k)
        v_split = self._split_heads(v)

        # compute attention
        attention = scaled_dot_product_attention(q_split, k_split, v_split)

        # concatenate and project
        concat_attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # reorder to [batch_size, seq_length, num_heads, depth]
        concat_attention = tf.reshape(concat_attention, [batch_size, -1, self._d_model])  # concat to [batch_size, seq_length, d_model]

        output = self._wc(concat_attention)

        return output


class FeedForwardLayer:
    """Two dense layers with relu on between"""
    def __init__(self, d_model, dff):
        self._f1 = tf.keras.layers.Dense(dff, activation='relu')
        self._f2 = tf.keras.layers.Dense(d_model, activation='linear')
        self._layer = tf.keras.Sequential([
            self._f1,
            self._f2
        ])

    def __call__(self, X):
        return self._layer(X)


class EncoderLayer:
    """One encoder layer. Includes multi head attention and feed forwards sublayers. each sublayer output is:
    output = LayerNorm(dropout(sublayer(sublayer_input)) + sublayer_input)
    """

    def __init__(self, d_model, attention_heads, d_feed_forward, dropout_rate):
        self._mha = MultiHeadAttention(attention_heads, d_model)
        self._ff = FeedForwardLayer(d_model, d_feed_forward)
        self._d_model = d_model
        self._mha_dropout = tf.keras.layers.Dropout(dropout_rate)  # multihead attention dropout
        self._ffl_dropout = tf.keras.layers.Dropout(dropout_rate)  # feed forward dropout
        self._mha_norm = tf.keras.layers.LayerNormalization()  # multihead attention LayerNorm
        self._ffl_norm = tf.keras.layers.LayerNormalization()  # feed forward LayerNorm

    def __call__(self, x):
        # multihead attention
        mha_output = self._mha(x, x, x)  # TODO: add padd mask
        mha_output = self._mha_dropout(mha_output)

        # add mha to input and norm
        s1 = self._mha_norm(x + mha_output)  # sublayer 1 output

        # apply feed forward layer
        ffl_output = self._ff(s1)
        ffl_output = self._ffl_dropout(ffl_output)

        # add ffl to s1 and norm
        s2 = self._ffl_norm(s1 + ffl_output)

        return s2


class DecoderLayer:
    """One decoder layer. Includes multihead attention over encoder outputs, masked multihead attention over previous
    decoder outputs and feed forward sublayers.
    """

    def __init__(self, d_model, attention_heads, d_feed_forward, dropout_rate):
        self._d_model = d_model
        self._mha = MultiHeadAttention(attention_heads, self._d_model)
        self._ffl = FeedForwardLayer(self._d_model, d_feed_forward)
        self._masked_mha_dropout = tf.keras.layers.Dropout(dropout_rate)
        self._encoder_mha_dropout = tf.keras.layers.Dropout(dropout_rate)
        self._ffl_dropout = tf.keras.layers.Dropout(dropout_rate)
        self._masked_mha_norm = tf.keras.layers.LayerNormalization()
        self._encoder_mha_norm = tf.keras.layers.LayerNormalization()
        self._ffl_norm = tf.keras.layers.LayerNormalization()

    def __call__(self, prev_dec_output, enc_output):
        """Decode next output from previous decoder output and the encoder output for this moment"""
        # masked multihead attention
        masked_mha = self._mha(prev_dec_output, prev_dec_output, prev_dec_output)  # TODO: add leftward ban mask
        masked_mha = self._masked_mha_dropout(masked_mha)

        # add and norm
        s1 = self._masked_mha_norm(masked_mha + prev_dec_output)

        # encoder multihead attention
        encoder_mha = self._mha(s1, enc_output, enc_output)  # TODO: add pad mask?
        encoder_mha = self._encoder_mha_dropout(encoder_mha)

        # add and norm
        s2 = self._encoder_mha_norm(encoder_mha + s1)

        # feed forward layer
        feed_forward = self._ffl(s2)
        feed_forward = self._ffl_dropout(feed_forward)

        # add and norm
        s3 = self._ffl_norm(feed_forward + s2)


if __name__ == '__main__':
    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)  # (4, 2)

    # This `query` aligns with the second `key`,
    # so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    a = scaled_dot_product_attention(temp_q, temp_k, temp_v)
    print(a)
    x = tf.random.uniform([128, 10, 8])
    mha = MultiHeadAttention(4, 8)
    print(mha(x, x, x).shape)
    ffl = FeedForwardLayer(8, 200)
    print(ffl(x).shape)
    enc = EncoderLayer(8, 4, 100, 0.1)
    print(enc(x).shape)

