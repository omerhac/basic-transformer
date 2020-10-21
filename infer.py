import tensorflow as tf
import preprocess
import pickle
import modules
import train


def infer(inp_sentence,
          transformer,
          inp_tokenizer_path='tokenizers/inp_tokenizer.pkl',
          tar_tokenizer_path='tokenizers/tar_tokenzier.pkl',
          max_target_sent_length=40,
          translate=True):
    """Infer the output sentence based on the input sentence
    Args:
        inp_sentence: input sentence as a string Tensor
        transformer: trained transformer model
        inp_tokenizer_path: path to serialized input language tokenizer
        tar_tokenizer_path: path to serialized target language tokenizer
        max_target_sent_length: maximum length of target sentence
        translate: flag whether to translate the output tokens to the target language
    """

    # get tokenizers
    inp_tokenizer = pickle.load(open(inp_tokenizer_path, 'rb'))
    tar_tokenizer = pickle.load(open(tar_tokenizer_path, 'rb'))

    # tokenize and pad
    inp_tokens = tf.cast(preprocess.tokenize_sentence(inp_sentence, inp_tokenizer), tf.int64)  # cast for padding
    inp_tokens = tf.expand_dims(preprocess.pad_sentence(inp_tokens, max_target_sent_length), axis=0) # expand for model digestion

    # start new output sentece
    out_tokens = tf.expand_dims([tar_tokenizer.num_words], axis=0)

    # inferring loop
    for i in range(max_target_sent_length):
        # predict next token, take argmax of log probabilities at each token
        predicted_next_token = tf.argmax(transformer(inp_tokens, out_tokens, training=False)[0][i])
        predicted_next_token = tf.expand_dims(predicted_next_token, axis=0)

        # add prediction to output sentence
        out_tokens = tf.concat([out_tokens, [predicted_next_token]], axis=1)
        print(out_tokens)

        # finish if end of sentence token is generated
        if predicted_next_token.numpy() == tar_tokenizer.num_words + 1:
            break

    out_tokens = out_tokens.numpy()
    if translate:
        return tar_tokenizer.sequences_to_texts(out_tokens)[0]
    else:
        return out_tokens


if __name__ == '__main__':
    t = modules.Transformer(8002, 8002, 40, d_model=128, n_layers=4, d_forward_layer=128, attention_heads=8)
    data = preprocess.load_dataset(data_dir='data')[0]
    for pt, en in data.take(2):
        print(en)
    print(infer(tf.constant(['going to the'], dtype='string', shape=()), t, inp_tokenizer_path='tokenizers/inp_tokenizer.pkl',
                tar_tokenizer_path='tokenizers/tar_tokenizer.pkl'))