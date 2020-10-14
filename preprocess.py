import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pickle

# CONSTANTS
MAX_LENGTH = 5


def load_dataset():
    """Load ted talks portugese to english dataset"""
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)

    return examples['train'], examples['validation']


def show_data(examples):
    """Show 10 rows"""
    for row in tfds.as_numpy(examples.take(10)):
        print('Portugese: {}'.format(row[0].decode('utf8')), 'English: {}'.format(row[1].decode('utf8')))


def get_tokenizers(corpus, dump_location=None):
    """Create text tokenizers.
    Args:
        corpus: text corpus to train on
        dump_location: location to save pickled tokenizers, if None the function will retrun them
    """

    # create tokenizers
    pt_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    en_tokenizer = tf.keras.preprocessing.text.Tokenizer()

    # fit tokenizers
    pt_tokenizer.fit_on_texts([pt.numpy().decode('utf8') for pt, en in corpus] + ['PTSTART PTEND'])
    pt_tokenizer.fit_on_texts([en.numpy().decode('utf8') for pt, en in corpus] + ['ENSTART ENEND'])

    if dump_location:
        pickle.dump(pt_tokenizer, open('{}/pt_tokenizer.pkl'.format(dump_location), 'wb'))
        pickle.dump(pt_tokenizer, open('{}/en_tokenizer.pkl'.format(dump_location), 'wb'))

    else:
        return pt_tokenizer, en_tokenizer


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


def tokenize(pt_sent, en_sent, pt_tokenizer_path='tokenizers/pt_tokenizer.pkl',
             en_tokenzier_path='tokenizers/en_tokenizer.pkl'):
    """
    Tokenize
    :param pt_sent: Portugese sentence
    :param en_sent: English sentence
    :param pt_tokenizer_path: path to pt tokenizer pickle
    :param en_tokenzier_path: path to english tokenizer pickle
    :return: tokenized sentences with start and stop tokens
    """

    # load tokenizers
    pt_tokenizer = pickle.load(open(pt_tokenizer_path, 'rb'))
    en_tokenizer = pickle.load(open(en_tokenzier_path, 'rb'))

    # start and end tokens
    pt_start_token = pt_tokenizer.texts_to_sequences(['PTSTART'])[0][0]
    pt_end_token = pt_tokenizer.texts_to_sequences(['PTEND'])[0][0]
    en_start_token = en_tokenizer.texts_to_sequences(['ENSTART'])[0][0]
    en_end_token = en_tokenizer.texts_to_sequences(['ENEND'])[0][0]

    # tokenize and add start and add tokens
    pt_tokens = [pt_start_token] + pt_tokenizer.texts_to_sequences([pt_sent.numpy().decode('utf8')])[0] + [pt_end_token]
    en_tokens = [en_start_token] + en_tokenizer.texts_to_sequences([en_sent.numpy().decode('utf8')])[0] + [en_end_token]

    return pt_tokens, en_tokens


def graph_tokenize(pt_sent, en_sent):
    """
    Graphiphy tokenize method
    """
    pt_tokens, en_tokens = tf.py_function(tokenize, [pt_sent, en_sent], Tout=[tf.int64, tf.int64])

    return pt_tokens, en_tokens


def pad(pt_tokens, en_tokens, padded_length):
    """Pad the sentences up to padded_length. It is assumed that the length of the sentences is <= padded_length"""
    padded_en_tokens = list(en_tokens.numpy())
    padded_pt_tokens = list(pt_tokens.numpy())

    # pad
    padded_en_tokens = padded_en_tokens + [0 for i in range(padded_length - len(padded_en_tokens))]
    padded_pt_tokens = padded_pt_tokens + [0 for i in range(padded_length - len(padded_pt_tokens))]

    return padded_pt_tokens, padded_en_tokens


def graph_pad(pt_tokens, en_tokens, padded_length=10):
    """Graphiphy pad method"""
    padded_pt_tokens, padded_en_tokens = tf.py_function(pad, [pt_tokens, en_tokens, padded_length], Tout=[tf.int64, tf.int64])

    return padded_pt_tokens, padded_en_tokens


def get_transformer_datasets(batch_size, max_length, buffer_size):
    """Return a tensorflow datasets of pairs of portugese-english ted translations, tokenized, padded and batched."""
    train_data, val_data = load_dataset()

    # tokenize
    train_data = train_data.map(graph_tokenize)
    val_data = val_data.map(graph_tokenize)

    # filter long sentences
    train_data = train_data.filter(lambda x, y: filter_max_length(x, y, max_length=max_length))
    val_data = val_data.filter(lambda x, y: filter_max_length(x, y, max_length=max_length))

    # pad
    train_data = train_data.map(lambda x, y: graph_pad(x,y, padded_length=max_length))
    val_data = val_data.map(lambda x, y: graph_pad(x, y, padded_length=max_length))

    # shuffle, batch and cache
    train_data = train_data.cache()
    train_data = train_data.shuffle(buffer_size).batch(batch_size)
    train_data.prefetch(tf.data.experimental.AUTOTUNE)
    val_data = val_data.batch(batch_size)

    return  train_data, val_data


if __name__ == '__main__':
    train_data, val_data = get_transformer_datasets(10, 10, 20)
    for pt, en in train_data.take(1):
        print(en)
