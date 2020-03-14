import collections
import tensorflow as tf
import numpy as np

# flatten list/dict of string
def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

#convert string to one-hot numerical embedding
def str2onehot(sample, alphabet):  # idxs is list of integers
    #return numpy 2D array where each character is a one-hot column vector
    #convert to indexes
    idxs = [alphabet.index(c) for c in sample]
    #convert to one-hot
    idxs_arr = np.array(idxs)
    length = len(alphabet)
    b = np.zeros((idxs_arr.size, length))
    b[np.arange(idxs_arr.size), idxs_arr] = 1
    return b.T.astype(np.int8)

## Functions to write/read to/from TFrecord files ##
def text_example(text, label):
    """
    Creates a tf.Example message ready to be written to a file.
    text is a NumPy ndarray
    """

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    text_tensor = tf.convert_to_tensor(text)
    text_str = tf.io.serialize_tensor(text_tensor)
    feature = {
        'text': _bytes_feature(text_str),
        'label': _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_text_example(example_proto):
    text_feature_description = {
        'text': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    ex= tf.io.parse_single_example(example_proto, text_feature_description)
    text_ser = ex['text']
    text_tensor = tf.io.parse_tensor(text_ser, tf.int8)
    return text_tensor.numpy(), ex['text'].numpy()
    # text_tensor = tf.convert_to_tensor(text)
    # text_str = tf.io.serialize_tensor(text_tensor)
    # return tf.train.Example(features=tf.train.Features(feature=feature))