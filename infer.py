import tensorflow as tf
import preprocess
import pickle

def infer(inp_sentence, transformer,
          inp_tokenizer_path='tokenizers/inp_tokenizer.pkl',
          tar_tokenizer_path='tokenizers/tar_tokenzier.pkl'):
    """Infer the output sentence based on the input sentence
    Args:
        inp_sentence: input sentence as a string Tensor
        transformer: trained transformer model
        inp_tokenizer_path: path to serialized input language tokenizer
        tar_tokenizer_path: path to serialized target language tokenizer
    """

    # get tokenizers
    inp_tokenizer = pickle.load(open(inp_tokenizer_path, 'rb'))
    tar_tokenizer = pickle.load(open(tar_tokenizer_path, 'rb'))

