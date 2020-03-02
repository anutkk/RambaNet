# TODO: convert to TensorFlow Dataset class - see https://www.tensorflow.org/datasets/add_dataset

import pathlib
import warnings
import json
import collections
import re


def preprocess_data(dataset_directory, input_size=1024, alphabet="אבגדהוזחטיכךלמםנןסעפףצץקרשת "):
    """ 
    Gets dataset directory path (which has structure as detailed behind TODO), and returns:
        - data as a list of (sample, label). Each sample is a numpy array of one-hot vectors, each label is an integer.
        - a dict where dict[label]=author_name
    """
    preprocessed_data = []
    ds_path = pathlib.Path(dataset_directory)
    for author_dir in ds_path.iterdir():
        if ~author_dir.is_dir():
            warnings.warn('File '+str(author_dir) +
                          ' ignored (invalid location).')
            continue

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

            # TODO: handle double quotes and keep them!
            # TODO: handle special double quotes characters. maybe convert all to special double quotes characters so that no need to escape?

            # keep only letters in alphabet and remove multiple spaces
            filtered = re.sub('[^'+alphabet+']', ' ', flattened_raw_str)
            filtered = re.sub(' +', ' ', filtered)
            # TODO: is it always correct to replace out-of-alphabet characters by spaces?

            # split to samples
            #TODO: prevent cutting in the middle of words
            n = input_size
            chunks = [filtered[i:i+n] for i in range(0, len(filtered), n)]


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
