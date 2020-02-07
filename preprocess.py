
# Preprocess dataset


import os
import json

#Sample_dataset version

#TODO: explain structure of raw folder

dataset_dirname = "./sample_dataset/"
raw_subdirname = "raw/"
raw_metadata_subdirame = "_schemas/"
organized_subdirname = "organized/"
preprocessed_subdirname = "preprocessed/"

#get list of all directories in raw folder
books = os.listdir(dataset_dirname + raw_subdirname)
#remove metadata directory from list
books.remove(raw_metadata_subdirame.replace('/', ''))

#load all relevant metadata
authors_dict = {}
error_count=0
metadata_dir = dataset_dirname + raw_subdirname + raw_metadata_subdirame
for metadata_fn in os.listdir(metadata_dir):
    filename, file_extension = os.path.splitext(metadata_dir+metadata_fn)
    if file_extension != '.json':
        continue
    with open(metadata_dir+metadata_fn, 'r', encoding="utf8") as metadata_file:
        try:
            metadata = json.load(metadata_file)
        except:
            continue
    bookname = filename.replace('_', ' ')
    author = ""
    try:
        author = metadata['authors'][0]['en']
    except: #if author=='':
        # print("Book '" + filename +"' has no valid author information")
        error_count+=1
    authors_dict[bookname] = author
print(str(error_count) + ' books out of '+ str(len(os.listdir(metadata_dir))) +' without valid author information.')

# for book in books:
    #get metadata
