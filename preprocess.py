import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pickle
import time

# CONSTANTS
MAX_LENGTH = 5
AUTO = tf.data.experimental.AUTOTUNE


def load_dataset(dataset_name='ted_hrlr_translate/pt_to_en', data_dir=None):
    """Load dataset_name from tensorflow datasets. Return train and validation datasets"""
    examples, metadata = tfds.load(dataset_name, with_info=True,
                                   as_supervised=True, data_dir=data_dir)

    return examples['train'], examples['validation']


def show_data(examples):
    """Show 10 rows"""
    for row in tfds.as_numpy(examples.take(10)):
        print('Portugese: {}'.format(row[0].decode('utf8')), 'English: {}'.format(row[1].decode('utf8')))


def get_tokenizers(corpus, dump_location=None, num_words=None):
    """Create text tokenizers.
    Args:
        corpus: text corpus to train on
        dump_location: location to save pickled tokenizers, if None the function will retrun them
        num_words: maximun number of words to keep
    """

    # create tokenizers
    inp_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words)
    tar_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words)

    # fit tokenizers
    inp_tokenizer.fit_on_texts([pt.numpy().decode('utf8') for pt, en in corpus])
    tar_tokenizer.fit_on_texts([en.numpy().decode('utf8') for pt, en in corpus])

    if dump_location:
        pickle.dump(inp_tokenizer, open('{}/inp_tokenizer.pkl'.format(dump_location), 'wb'))
        pickle.dump(tar_tokenizer, open('{}/tar_tokenizer.pkl'.format(dump_location), 'wb'))

    else:
        return inp_tokenizer, tar_tokenizer


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


def tokenize_sentence(sentence, tokenizer):
    """Tokenize the sentence based on its tokenizer"""
    # start and end tokens
    start_token = tokenizer.num_words
    end_token = tokenizer.num_words + 1

    # tokenize_example sentence
    tokens = [start_token] + tokenizer.texts_to_sequences([sentence.numpy().decode('utf8')])[0] + [end_token]

    return tokens


def tokenize_example(inp_sent, tar_sent, inp_tokenizer=None, tar_tokenizer=None):
    """
    Tokenize one example from the corpus. As in tokenize both the input and output sentences
    :param inp_sent: inpyt sentence
    :param tar_sent: target sentence
    :param inp_tokenizer: input tokenizer pickle
    :param tar_tokenizer_tokenzier: target tokenizer pickle
    :return: tokenized sentences with start and stop tokens
    """

    # load tokenizers
    if not inp_tokenizer:
        inp_tokenizer = pickle.load(open('tokenizers/inp_tokenizer.pkl', 'rb'))
    if not tar_tokenizer:
        tar_tokenizer = pickle.load(open('tokenizers/out_tokenizer.pkl', 'rb'))

    # tokenize_example sentences
    inp_tokens = tokenize_sentence(inp_sent, inp_tokenizer)
    tar_tokens = tokenize_sentence(tar_sent, tar_tokenizer)

    return inp_tokens, tar_tokens


def pad_sentence(tokens, padded_length):
    """Pad the sentences up to padded_length. It is assumed that the length of the sentences is <= padded_length"""
    padded_tokens = list(tokens.numpy())
    padded_tokens = padded_tokens + [0 for i in range(padded_length - len(padded_tokens))]
    return  padded_tokens


def pad_example(inp_tokens, tar_tokens, padded_length):
    """Pad the sentences up to padded_length. It is assumed that the length of the sentences is <= padded_length"""
    # pad_example
    padded_tar_tokens = pad_sentence(inp_tokens, padded_length)
    padded_inp_tokens = pad_sentence(tar_tokens, padded_length)

    return padded_inp_tokens, padded_tar_tokens


def graph_pad(inp_tokens, tar_tokens, padded_length=10):
    """Graphiphy pad_example method"""
    padded_pt_tokens, padded_en_tokens = tf.py_function(pad_example, [inp_tokens, tar_tokens, padded_length], Tout=[tf.int64, tf.int64])

    return padded_pt_tokens, padded_en_tokens


def get_transformer_datasets(batch_size, max_length, buffer_size,
                             inp_tokenizer_path='tokenizers/pt_tokenizer.pkl',
                             tar_tokenizer_path='tokenizers/pt_tokenizer.pkl'):
    """Return a tensorflow datasets of pairs of portugese-english ted translations, tokenized, padded and batched.

    Args:
        batch_size: batch size
        max_length: maximum sequence length. longer sequences will be pruned and shorter ones padded
        inp_tokenizer_path: path to serialized tokenizer for the input language. must implement text_to_sequences
        tar_tokenizer_path: path to serialized tokenizer for the target language. must implement text_to_sequences
    """

    # get dataset
    train_data, val_data = load_dataset(data_dir='data')

    # tokenize_example
    pt_tokenizer = pickle.load(open(inp_tokenizer_path, 'rb'))
    en_tokenizer = pickle.load(open(tar_tokenizer_path, 'rb'))

    # helper functions
    tokenizer = lambda x, y: tokenize_example(x, y, pt_tokenizer, en_tokenizer)

    def graph_tokenize(pt_sent, en_sent):
        """
        Graphiphy tokenize_example method
        """
        pt_tokens, en_tokens = tf.py_function(tokenizer, [pt_sent, en_sent], Tout=[tf.int64, tf.int64])

        return pt_tokens, en_tokens

    train_data = train_data.map(graph_tokenize)
    val_data = val_data.map(graph_tokenize)

    # filter long sentences
    train_data = train_data.filter(lambda x, y: filter_max_length(x, y, max_length=max_length))
    val_data = val_data.filter(lambda x, y: filter_max_length(x, y, max_length=max_length))

    # pad_example
    train_data = train_data.map(lambda x, y: graph_pad(x,y, padded_length=max_length), num_parallel_calls=AUTO)
    val_data = val_data.map(lambda x, y: graph_pad(x, y, padded_length=max_length), num_parallel_calls=AUTO)

    # shuffle, batch and cache
    train_data = train_data.cache()
    train_data = train_data.shuffle(buffer_size).batch(batch_size)
    train_data.prefetch(tf.data.experimental.AUTOTUNE)
    val_data = val_data.batch(batch_size)

    return train_data, val_data


def time_dataset(dataset, iterations=5):
    """Time |iterations| elements from dataset"""
    curr_time = time.time()

    for _ in dataset.take(iterations):
        print("Time for element: {}".format(time.time() - curr_time))
        curr_time = time.time()


if __name__ == '__main__':
    time_dataset(get_transformer_datasets(64, 40, 2000)[0])
