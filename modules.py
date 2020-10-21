import tensorflow as tf
import numpy as np


def scaled_dot_product_attention(q, k, v, mask=None):
    """Compute the scaled dot product attention according to Attention(Q,K,V) = softmax(QK^T/sqrt(d_model))V"""
    d_model = tf.cast(tf.shape(q)[-1], tf.float32)
    qkt = tf.matmul(q, k, transpose_b=True)  # QK^T

    attention_logits = qkt / tf.sqrt(d_model)

    # add mask
    if mask is not None:
        attention_logits += mask * -1e9  # add -infinity to saturate the softmax to 0

    attention_weights = tf.nn.softmax(attention_logits)

    # compute attention values
    attention_values = tf.matmul(attention_weights, v)

    return attention_values


def pad_mask(x):
    """Create a tensor to mask pad_example tokens. Input x is assumed to be of shape [batch_size, sequence_length]
    as its before embedding.
    """
    x = tf.cast(x, tf.int32)  # for compatibility reasons
    mask = tf.cast(tf.equal(0, x), tf.float32)

    # This works because the broadcasting defacto causes the mask to operate on elements of the attention matrix that
    # corresponds to the weights each word attends to the padding words.
    # That is, the broadcasting masks the A[x, pad_example] of the attention matrix.
    mask = mask[:, tf.newaxis, tf.newaxis, :]

    return mask


def lookahead_mask(x):
    """
    Create mask to prevent the decoder from looking on the yet ungenerated sequence. Input x is assumed to be of shape
    [batch_size, sequence_length] as its before embedding.
    It is actually going to mask the attention weights matrix.
    """
    seq_length = tf.shape(x)[1]

    mask = tf.linalg.band_part(tf.ones((seq_length, seq_length)), 0, -1) - tf.linalg.band_part(tf.ones((seq_length, seq_length)), 0, 0)
    # band(0, -1) is lower triangle part and band(0,0) is diagonal

    # the mask should be the same for every sequence, it doesnt regard which sample it is
    return mask[tf.newaxis, tf.newaxis, :, :]


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi Head Attention layer as in MHA(Q,K,V) = concat(attention_head1, attention_head2, .., attention_headn)Wc,
    where attention_headi = attention(QWq, KWk, VWv). Wc, Wq, Wk and Wv are projections to d_c, d_q, d_k and d_v.
    """

    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
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

    def __call__(self, q, k, v, mask=None):
        """Compute the Multy Head Attention of Q, K, V"""
        batch_size = tf.shape(q)[0]

        # project matrices
        q = self._wq(q)
        k = self._wk(k)
        v = self._wv(v)

        # split each matrix
        q_split = self._split_heads(q)
        k_split = self._split_heads(k)
        v_split = self._split_heads(v)

        # compute attention
        attention = scaled_dot_product_attention(q_split, k_split, v_split, mask=mask)

        # concatenate and project
        concat_attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # reorder to [batch_size, seq_length, num_heads, depth]
        concat_attention = tf.reshape(concat_attention, (batch_size, -1, self._d_model))  # concat to [batch_size, seq_length, d_model]

        output = self._wc(concat_attention)

        return output


class FeedForwardLayer(tf.keras.layers.Layer):
    """Two dense layers with relu on between"""
    def __init__(self, d_model, dff):
        super(FeedForwardLayer, self).__init__()
        self._f1 = tf.keras.layers.Dense(dff, activation='relu')
        self._f2 = tf.keras.layers.Dense(d_model, activation='linear')
        self._layer = tf.keras.Sequential([
            self._f1,
            self._f2
        ])

    def __call__(self, X):
        return self._layer(X)


class EncoderLayer(tf.keras.layers.Layer):
    """One encoder layer. Includes multi head attention and feed forwards sublayers. each sublayer output is:
    output = LayerNorm(dropout(sublayer(sublayer_input)) + sublayer_input)
    """

    def __init__(self, d_model, attention_heads, d_feed_forward, dropout_rate):
        super(EncoderLayer, self).__init__()
        self._mha = MultiHeadAttention(attention_heads, d_model)
        self._ff = FeedForwardLayer(d_model, d_feed_forward)
        self._d_model = d_model
        self._mha_dropout = tf.keras.layers.Dropout(dropout_rate)  # multihead attention dropout
        self._ffl_dropout = tf.keras.layers.Dropout(dropout_rate)  # feed forward dropout
        self._mha_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # multihead attention LayerNorm
        self._ffl_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # feed forward LayerNorm

    def __call__(self, x, pad_mask, training):
        # multihead attention
        mha_output = self._mha(x, x, x, pad_mask)
        mha_output = self._mha_dropout(mha_output, training=training)

        # add mha to input and norm
        s1 = self._mha_norm(x + mha_output)  # sublayer 1 output

        # apply feed forward layer
        ffl_output = self._ff(s1)
        ffl_output = self._ffl_dropout(ffl_output, training=training)

        # add ffl to s1 and norm
        s2 = self._ffl_norm(s1 + ffl_output)

        return s2


class DecoderLayer(tf.keras.layers.Layer):
    """One decoder layer. Includes multihead attention over encoder outputs, masked multihead attention over previous
    decoder outputs and feed forward sublayers.
    """

    def __init__(self, d_model, attention_heads, d_feed_forward, dropout_rate):
        super(DecoderLayer, self).__init__()
        self._d_model = d_model
        self._mha = MultiHeadAttention(attention_heads, self._d_model)
        self._ffl = FeedForwardLayer(self._d_model, d_feed_forward)
        self._masked_mha_dropout = tf.keras.layers.Dropout(dropout_rate)
        self._encoder_mha_dropout = tf.keras.layers.Dropout(dropout_rate)
        self._ffl_dropout = tf.keras.layers.Dropout(dropout_rate)
        self._masked_mha_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._encoder_mha_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self._ffl_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def __call__(self, prev_dec_output, enc_output, pad_mask, lookahead_mask, training):
        """Decode next output from previous decoder output and the encoder output for this moment"""
        # masked multihead attention
        masked_mha = self._mha(prev_dec_output, prev_dec_output, prev_dec_output, mask=lookahead_mask)
        masked_mha = self._masked_mha_dropout(masked_mha, training=training)

        # add and norm
        s1 = self._masked_mha_norm(masked_mha + prev_dec_output)

        # encoder multihead attention
        encoder_mha = self._mha(s1, enc_output, enc_output, mask=pad_mask)
        encoder_mha = self._encoder_mha_dropout(encoder_mha, training=training)

        # add and norm
        s2 = self._encoder_mha_norm(encoder_mha + s1)

        # feed forward layer
        feed_forward = self._ffl(s2)
        feed_forward = self._ffl_dropout(feed_forward, training=training)

        # add and norm
        s3 = self._ffl_norm(feed_forward + s2)

        return s3


class Encoder(tf.keras.layers.Layer):
    """Transformer encoder"""

    def __init__(self, d_model, n_layers, attention_heads, d_forward_layer, dropout_rate):
        super(Encoder, self).__init__()
        self._d_model = d_model
        self._attention_heads = attention_heads
        self.d_forward_layer = d_forward_layer
        self._n_layers = n_layers

        # initiate n_layers encoder layers
        self._layers = [EncoderLayer(self._d_model, self._attention_heads, self.d_forward_layer, dropout_rate)
                        for _ in range(n_layers)]

    def __call__(self, x, pad_mask, training):
        current_layer_output = x

        # move x through all layers
        for layer in self._layers:
            current_layer_output = layer(current_layer_output, pad_mask, training=training)

        return current_layer_output


class Decoder(tf.keras.layers.Layer):
    """Transformer decoder"""

    def __init__(self, d_model, n_layers, attention_heads, d_forward_layer, dropout_rate):
        super(Decoder, self).__init__()
        self._d_model = d_model
        self._attention_heads = attention_heads
        self.d_forward_layer = d_forward_layer
        self._n_layers = n_layers

        # initiate n_layers encoder layers
        self._layers = [DecoderLayer(self._d_model, self._attention_heads, self.d_forward_layer, dropout_rate)
                        for _ in range(n_layers)]

    def __call__(self, prev_dec_output, enc_output, pad_mask, lookahead_mask, training):
        current_layer_output = prev_dec_output

        # move the previous decoder output through all the layers. encoder output is injected to every layer
        for layer in self._layers:
            current_layer_output = layer(current_layer_output, enc_output, pad_mask, lookahead_mask, training=training)

        return current_layer_output


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def get_positional_encodings(d_model, max_seq_length):
    """Get a matrix of shape [1, max_seq_length, d_model] of positional encodings.
    They apply the same for each example hence the 1 row on first dim.
    Args:
        d_model: dimensionality of input embeddings
        max_seq_length: maximum length of the input sequence
    """
    angle_rads = get_angles(np.arange(max_seq_length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class Transformer(tf.keras.Model):
    """Transformer model"""

    def __init__(self, vocab_size, target_vocab_size, max_seq_length, d_model=512, n_layers=6, attention_heads=8, d_forward_layer=2045,
                 dropout_rate=0.1, encoder=None, decoder=None):
        """
        Init the model with the default parameters from the paper.
        If a trained encoder / decoder is provided the model will not generate a new one.

        Args:
            vocab_size: size of input language vocabulary
            target_vocab_size: size of output language vocabulary
            max_seq_length: maximu allowed length of input sequence
            d_model: dimension of input embeddings
            n_layers: number of encoder / decoder layers
            attention_heads: number of heads for multihead attention. (how many times to split d_model)
            d_forward_layer: number of units in the second dense layer of each pointwise forward sublayer
            dropout_rate: dropout rate at each dropout layer
            encoder: pretrained encoder if available
            decoder: pretrained decoder if available
        """

        super(Transformer, self).__init__()
        self._encoder = encoder if encoder else Encoder(d_model, n_layers, attention_heads, d_forward_layer, dropout_rate)
        self._decoder = decoder if decoder else Decoder(d_model, n_layers, attention_heads, d_forward_layer, dropout_rate)
        self._d_model = d_model
        self._attention_heads = attention_heads
        self._d_forward_layer = d_forward_layer
        self._dropout_rate = dropout_rate
        self._vocab_size = vocab_size
        self._encoder_dropout = tf.keras.layers.Dropout(dropout_rate)
        self._decoder_dropout = tf.keras.layers.Dropout(dropout_rate)
        self._pos_encoding = get_positional_encodings(self._d_model, max_seq_length)

        # create embedding layer
        self._input_embedding = tf.keras.layers.Embedding(vocab_size, d_model)

        # create end linear projection
        self._linear_projection = tf.keras.layers.Dense(target_vocab_size, activation='linear')

    def __call__(self, x, prev_dec_output, training):
        # embed x and previous decoder output and scale by sqrt d_model
        x_embedd = self._input_embedding(x) * tf.sqrt(tf.cast(self._d_model, tf.float32))
        prev_dec_output_embedd = self._input_embedding(prev_dec_output) * tf.sqrt(tf.cast(self._d_model, tf.float32))

        # get positional encodings
        inp_seq_length = tf.shape(x)[1]  # input and target can have different lengths because of target shifting and on
        tar_seq_length = tf.shape(prev_dec_output)[1]  # inferring
        inp_positional_enc = self._pos_encoding[:, :inp_seq_length, :]
        out_positional_enc = self._pos_encoding[:, :tar_seq_length, :]

        # add positional encodings to encoder and decoder inputs
        x_embedd = x_embedd + inp_positional_enc
        prev_dec_output_embedd = prev_dec_output_embedd + out_positional_enc

        # apply dropouts
        x_embedd = self._encoder_dropout(x_embedd, training=training)
        prev_dec_output_embedd = self._decoder_dropout(prev_dec_output_embedd, training=training)

        # get masks
        inp_p_mask = pad_mask(x)  # encoder inputs pad_example mask
        out_p_mask = pad_mask(prev_dec_output)  # decoder input pad_example mask
        la_mask = lookahead_mask(prev_dec_output)  # decoder lookahead mask
        dec_combined_mask = tf.maximum(out_p_mask, la_mask)  # combine the lookahead and padding mask for the input of the decoder

        # apply encoder
        encoder_output = self._encoder(x_embedd, inp_p_mask, training=training)

        # apply decoder
        # since the attention keys are from the encoder, tha masking should be with regard to them
        decoder_output = self._decoder(prev_dec_output_embedd, encoder_output, inp_p_mask,
                                       dec_combined_mask, training=training)

        # apply linear layer and softmax
        output_logits = self._linear_projection(decoder_output)
        output_probas = tf.nn.softmax(output_logits)

        return output_probas


if __name__ == '__main__':
    print(tf.__version__)
    a = tf.constant([
        [1, 0, 1],
    ])
    print(lookahead_mask(a))