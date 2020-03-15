import json
import pathlib
import shutil
import re
import numpy as np
import tables
import tensorflow as tf
import os
from utils import flatten, str2onehot, text_example
import io


#Organize dataset
#TODO: consider write to TFRecord file
# https://towardsdatascience.com/working-with-tfrecords-and-tf-train-example-36d111b3ff4d
# https://medium.com/@prasad.pai/how-to-use-tfrecord-with-datasets-and-iterators-in-tensorflow-with-code-samples-ffee57d298af
# https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
# https://www.tensorflow.org/tutorials/load_data/tfrecord
# https://www.tensorflow.org/guide/data
# https://www.tensorflow.org/guide/data_performance
# https://www.tensorflow.org/tutorials/load_data/tfrecord
def organize_data(dataset_dirname = "./sample_dataset/"):
    """ 
    Organize dataset in convenient folder structure and keep only relevant files in convenient form. 
    Existing data is overwritten.
    """
    #TODO: explain folder structure of input and output
    # dataset_dirname = "./sample_dataset/"
    raw_subdirname = "raw/"
    raw_metadata_subdirame = "_schemas/"
    organized_subdirname = "organized/"

    #load all relevant metadata
    authors_dict = {}
    error_count=0
    metadata_dir = dataset_dirname + raw_subdirname + raw_metadata_subdirame
    for metadata_fn in os.listdir(metadata_dir):
        filename, file_extension = os.path.splitext(metadata_fn)
        if file_extension != '.json':
            continue
        with open(metadata_dir+metadata_fn, 'r', encoding="utf8") as metadata_file:
            try:
                metadata = json.load(metadata_file)
            except:
                continue
        bookname = filename.replace('_', ' ')
        # author = ""
        try:
            author = metadata['authors'][0]['en']
        except: #if author=='':
            idx = bookname.find(" on ")
            if idx>0:
                author = bookname[0:idx]
            else:
                author = bookname
            # print("Book '" + filename +"' has no valid author information")
            error_count+=1
        authors_dict[bookname] = author
    print(str(error_count) + ' books out of '+ str(len(os.listdir(metadata_dir))) +' without valid author information were corrected. ')

    #get list of all directories in raw folder
    books = os.listdir(dataset_dirname + raw_subdirname)
    #remove metadata directory from list
    books.remove(raw_metadata_subdirame.replace('/', ''))
    books.remove('_links')
    #organize books
    for book in books:
        book_path = pathlib.Path(dataset_dirname + raw_subdirname+book+'/Hebrew/merged.json')
        #validate
        if not book_path.is_file():
            print('Directory ' + str(book_path) + ' ignored (invalid location).')
            continue
        if book_path.suffix != '.json':
            print('File ' + str(book_path) + ' ignored (type should be JSON).')
            continue
        #get author
        if book in authors_dict.keys():
            curr_author = authors_dict[book]
            author_dirname = dataset_dirname+organized_subdirname+curr_author+'/'
            pathlib.Path(author_dirname).mkdir(parents=True, exist_ok=True)
            out_file = author_dirname+book+".txt"
            # shutil.copyfile(dataset_dirname + raw_subdirname+book+'/Hebrew/merged.json',)

            # Put content into simple TXT file
            # Load JSON data
            with book_path.open(mode='r', encoding='utf8') as book_file:
                try:
                    book_raw_text = json.load(book_file)['text']
                except:
                    print('File '+str(book_path) +' ignored (impossible to read JSON).')
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

            #write to file
            with io.open(out_file, 'w', encoding='utf8') as f:
                f.write(flattened_raw_str)

#generator function (including preprocessing -> NumPy arrays)
#TODO: consider making preprocessing after generation. For now most compatible
def get_sample(dataset_directory = "./sample_dataset/organized", input_size=1024, alphabet='אבגדהוזחטיכךלמםנןסעפףצץקרשת "'):
    ds_path = pathlib.Path(dataset_directory)
    authors = list(enumerate(ds_path.iterdir()))
    one_hot_matrix = np.eye(len(authors), dtype='int8')
    for author_id, author_dir in authors:
        for book_path in author_dir.iterdir():
            with book_path.open(mode    ='r', encoding='utf8') as book_file:
                flattened_raw_str = book_file.read()
                # keep only letters in alphabet and remove multiple spaces
            filtered = re.sub('[^'+alphabet+']', ' ', flattened_raw_str)
            filtered = re.sub(' +', ' ', filtered)
            # TODO: is it always correct to replace out-of-alphabet characters by spaces?

            # split to samples
            #TODO: prevent cutting in the middle of words
            n = input_size
            samples = [filtered[i:i+n] for i in range(0, len(filtered), n)]

            #convert to numerical one-hot
            #TODO: consider convert to sparse representation
            samples_onehot_minus1 = np.stack([str2onehot(sample, alphabet) for sample in samples[0:-1]], axis=0)
            #pad last sample and add it to 3d array
            lastsample_onehot = str2onehot(samples[-1], alphabet)
            lastsample_onehot_padded = np.zeros_like(samples_onehot_minus1[-1,:,:], dtype=np.int8)
            lastsample_onehot_padded[0:lastsample_onehot.shape[0], 0:lastsample_onehot.shape[1]] = lastsample_onehot
            samples_onehot = np.concatenate((samples_onehot_minus1, lastsample_onehot_padded[np.newaxis,:,:]))
            author_label = one_hot_matrix[:,author_id]
            for sample in samples_onehot:
                yield (sample, author_label)


# def preprocess_all_data(dataset_directory, input_size=1024, alphabet='אבגדהוזחטיכךלמםנןסעפףצץקרשת "', output_filename='./sample_dataset/sample_dataset'):
#     """ 
#     Gets dataset directory path (which has structure as detailed behind TODO), and writes to file data as numeric NumPy ndarray in HDF5 file and TFrecord file.

#     If the output files already exists the preprocessed data is *overwritten*.
#     """
#     #initialize variables
#     preprocessed_samples = np.array([], dtype=np.int8)
#     preprocessed_labels =  np.array([], dtype=np.int8)

#     h5_fn = output_filename+'.h5'
#     tfr_fn = output_filename+'.tfrecords'

#     #initialize files dataset will be stored in
#     with tables.open_file(h5_fn, mode='w') as h5file, tf.io.TFRecordWriter(tfr_fn) as tfwriter:
#         typeAtom = tables.Int8Atom()
#         print('Processing...')
#         #iterate over authors
#         ds_path = pathlib.Path(dataset_directory)
#         for author_label, author_dir in enumerate(ds_path.iterdir()):
#             #validate
#             print('Processing ' + str(author_dir) + '...')
#             if not author_dir.is_dir():
#                 print('File '+str(author_dir) +' ignored (invalid location).')
#                 continue

#             #create h5 group and table
#             gauthor = h5file.create_group(h5file.root, 'author'+str(author_label), author_dir.name)
#             array_c = h5file.create_earray(gauthor, 'samples', typeAtom, (0,len(alphabet), input_size), author_dir.name+" Samples")

#             # author_dict[author_label] = author_dir.name
#             for book_path in author_dir.iterdir():
#                 # validation check
#                 if not book_path.is_file():
#                     print('Directory ' + str(author_dir) + ' ignored (invalid location).')
#                     continue
#                 if book_path.suffix != '.json':
#                     print('File ' + str(author_dir) + ' ignored (type should be JSON).')
#                     continue

#                 # load JSON data
#                 with book_path.open(mode='r', encoding='utf8') as book_file:
#                     try:
#                         book_raw_text = json.load(book_file)['text']
#                         # book_raw_text = book_raw_data
#                     except:
#                         print('File '+str(author_dir) +
#                                     ' ignored (impossible to read JSON).')
#                         continue

#                 # flatten
#                 if isinstance(book_raw_text, list):  # no internal separation of text
#                     flattened_raw_lst = list(flatten(book_raw_text))
#                 elif isinstance(book_raw_text, dict):# internal separation of text - dict of dicts
#                     tmp = []
#                     for d in book_raw_text.values():
#                         if isinstance(d, dict):
#                             tmp.extend(list(d.values()))
#                         elif isinstance(d, list):
#                             tmp.extend(d)
#                     flattened_raw_lst = list(flatten(tmp))
#                 else:
#                     raise ValueError(str(book_path)+ ': Could not parse.')

#                 # ensure file does not have different structure from expected
#                 assert(all(isinstance(x, str) for x in flattened_raw_lst))
#                 # TODO: check manually all is well

#                 # concatenate
#                 flattened_raw_str = ''.join(flattened_raw_lst)

#                 # TODO: handle single quote characters

#                 # keep only letters in alphabet and remove multiple spaces
#                 filtered = re.sub('[^'+alphabet+']', ' ', flattened_raw_str)
#                 filtered = re.sub(' +', ' ', filtered)
#                 # TODO: is it always correct to replace out-of-alphabet characters by spaces?

#                 # split to samples
#                 #TODO: prevent cutting in the middle of words
#                 n = input_size
#                 samples = [filtered[i:i+n] for i in range(0, len(filtered), n)]

#                 #convert to numerical one-hot
#                 samples_onehot_minus1 = np.stack([str2onehot(sample, alphabet) for sample in samples[0:-1]], axis=0)
#                 #pad last sample and add it to 3d array
#                 lastsample_onehot = str2onehot(samples[-1], alphabet)
#                 lastsample_onehot_padded = np.zeros_like(samples_onehot_minus1[-1,:,:], dtype=np.int8)
#                 lastsample_onehot_padded[0:lastsample_onehot.shape[0], 0:lastsample_onehot.shape[1]] = lastsample_onehot
#                 samples_onehot = np.concatenate((samples_onehot_minus1, lastsample_onehot_padded[np.newaxis,:,:]))

#                 ## write to file
#                 #write to h5
#                 array_c.append(samples_onehot)
#                 #write to tfrecord
#                 for text_arr in samples_onehot:
#                     tf_example = text_example(text_arr, author_label)
#                     tfwriter.write(tf_example.SerializeToString())
#             h5file.flush()
#             tfwriter.flush()


# organize_data() #DEBUG

# preprocess_data('./sample_dataset/organized') #DEBUG!