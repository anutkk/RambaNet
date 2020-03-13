# TODO: convert to TensorFlow Dataset class - see https://www.tensorflow.org/datasets/add_dataset

import pathlib
import warnings
import json
import collections
import re
import numpy as np
# import h5py
import tables
import tensorflow as tf

def preprocess_data(dataset_directory, input_size=1024, alphabet='אבגדהוזחטיכךלמםנןסעפףצץקרשת "', output_filename='./sample_dataset/sample_dataset'):
    """ 
    Gets dataset directory path (which has structure as detailed behind TODO), and writes to file data as numeric NumPy ndarray in HDF5 file and TFrecord file.

    If the output files already exists the preprocessed data is *overwritten*.
    """
    #initialize variables
    preprocessed_samples = np.array([], dtype=np.int8)
    preprocessed_labels =  np.array([], dtype=np.int8)

    h5_fn = output_filename+'.h5'
    tfr_fn = output_filename+'.tfrecords'

    #initialize files dataset will be stored in
    with tables.open_file(h5_fn, mode='w') as h5file, tf.io.TFRecordWriter(tfr_fn) as tfwriter:
        typeAtom = tables.Int8Atom()
        print('Processing...')
        #iterate over authors
        ds_path = pathlib.Path(dataset_directory)
        for author_label, author_dir in enumerate(ds_path.iterdir()):
            #validate
            print('Processing ' + str(author_dir) + '...')
            if not author_dir.is_dir():
                print('File '+str(author_dir) +' ignored (invalid location).')
                continue

            #create h5 group and table
            gauthor = h5file.create_group(h5file.root, 'author'+str(author_label), author_dir.name)
            array_c = h5file.create_earray(gauthor, 'samples', typeAtom, (0,len(alphabet), input_size), author_dir.name+" Samples")

            # author_dict[author_label] = author_dir.name
            for book_path in author_dir.iterdir():
                # validation check
                if not book_path.is_file():
                    print('Directory ' + str(author_dir) + ' ignored (invalid location).')
                    continue
                if book_path.suffix != '.json':
                    print('File ' + str(author_dir) + ' ignored (type should be JSON).')
                    continue

                # load JSON data
                with book_path.open(mode='r', encoding='utf8') as book_file:
                    try:
                        book_raw_text = json.load(book_file)['text']
                        # book_raw_text = book_raw_data
                    except:
                        warnings.warn('File '+str(author_dir) +
                                    ' ignored (impossible to read JSON).')
                        continue

                # flatten
                if isinstance(book_raw_text, list):  # no internal separation of text
                    flattened_raw_lst = list(flatten(book_raw_text))
                elif isinstance(book_raw_text, dict):# internal separation of text - dict of dicts
                    tmp = []
                    for d in book_raw_text.values():
                        if isinstance(d, dict):
                            tmp.extend(list(d.values()))
                        elif isinstance(d, list):
                            tmp.extend(d)
                    flattened_raw_lst = list(flatten(tmp))
                else:
                    raise ValueError(str(book_path)+ ': Could not parse.')

                # ensure file does not have different structure from expected
                assert(all(isinstance(x, str) for x in flattened_raw_lst))
                # TODO: check manually all is well

                # concatenate
                flattened_raw_str = ''.join(flattened_raw_lst)

                # TODO: handle single quote characters

                # keep only letters in alphabet and remove multiple spaces
                filtered = re.sub('[^'+alphabet+']', ' ', flattened_raw_str)
                filtered = re.sub(' +', ' ', filtered)
                # TODO: is it always correct to replace out-of-alphabet characters by spaces?

                # split to samples
                #TODO: prevent cutting in the middle of words
                n = input_size
                samples = [filtered[i:i+n] for i in range(0, len(filtered), n)]

                #convert to numerical one-hot
                samples_onehot_minus1 = np.stack([str2onehot(sample, alphabet) for sample in samples[0:-1]], axis=0)
                #pad last sample and add it to 3d array
                lastsample_onehot = str2onehot(samples[-1], alphabet)
                lastsample_onehot_padded = np.zeros_like(samples_onehot_minus1[-1,:,:], dtype=np.int8)
                lastsample_onehot_padded[0:lastsample_onehot.shape[0], 0:lastsample_onehot.shape[1]] = lastsample_onehot
                samples_onehot = np.concatenate((samples_onehot_minus1, lastsample_onehot_padded[np.newaxis,:,:]))

                ## write to file
                #write to h5
                array_c.append(samples_onehot)
                #write to tfrecord
                for text_arr in samples_onehot:
                    tf_example = text_example(text_arr, author_label)
                    tfwriter.write(tf_example.SerializeToString())
            h5file.flush()
            tfwriter.flush()


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
    return tf.io.parse_single_example(example_proto, text_feature_description)

    # text_tensor = tf.convert_to_tensor(text)
    # text_str = tf.io.serialize_tensor(text_tensor)
    # return tf.train.Example(features=tf.train.Features(feature=feature))

preprocess_data('./sample_dataset/organized') #DEBUG!