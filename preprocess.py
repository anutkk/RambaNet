# TODO: convert to TensorFlow Dataset class - see https://www.tensorflow.org/datasets/add_dataset

import pathlib
import warnings
import json
import collections
import re
import numpy as np


def preprocess_data(dataset_directory, input_size=1024, alphabet='אבגדהוזחטיכךלמםנןסעפףצץקרשת "'):
    """ 
    Gets dataset directory path (which has structure as detailed behind TODO), and returns:
        - data as a list of (sample, label). Each sample is a numpy array of one-hot vectors, each label is an integer.
        - a dict where dict[label]=author_name
    """
    preprocessed_data = []
    author_dict = {}
    ds_path = pathlib.Path(dataset_directory)
    for author_label, author_dir in enumerate(ds_path.iterdir()):
        if ~author_dir.is_dir():
            warnings.warn('File '+str(author_dir) +
                          ' ignored (invalid location).')
            continue
        author_dict[author_label] = author_dir.name
        for book_path in author_dir.iterdir():
            # validation check
            if ~book_path.is_file():
                warnings.warn('Directory '+str(author_dir) +
                              ' ignored (invalid location).')
                continue
            if ~book_path.suffix == '.json':
                warnings.warn('File '+str(author_dir) +
                              ' ignored (type should be JSON).')
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
            # internal separation of text - dict of dicts
            elif isinstance(book_raw_text, dict):
                tmp = [list(d.values()) for d in book_raw_text.values()]
                flattened_raw_lst = list(flatten(tmp))

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

            #convert to one-hot and aggregate
            preprocessed_data.extend( \
                [(str2onehot(sample, alphabet), author_label) for sample in samples] \
                    )
            
            #TODO: should author labels be one-hot vectors too? 




def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def str2onehot(sample, alphabet):  # idxs is list of integers
    #return numpy 2D array where each character is a one-hot column vector
    #convert to indexes
    idxs = [alphabet.index(c) for c in sample]
    #convert to one-hot
    idxs_arr = np.array(idxs)
    length = len(alphabet)
    b = np.zeros((idxs_arr.size, length))
    b[np.arange(idxs_arr.size), idxs_arr] = 1
    return b.T
