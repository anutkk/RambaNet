
# Organize dataset
#TODO: transform to functions

import os
import json
import pathlib
import shutil
#Sample_dataset version

#TODO: explain structure of raw folder

dataset_dirname = "./sample_dataset/"
raw_subdirname = "raw/"
raw_metadata_subdirame = "_schemas/"
organized_subdirname = "organized/"
preprocessed_subdirname = "preprocessed/"


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
#organize books
for book in books:
    #get author
    if book in authors_dict.keys():
        curr_author = authors_dict[book]
        author_dirname = dataset_dirname+organized_subdirname+curr_author+'/'
        pathlib.Path(author_dirname).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset_dirname + raw_subdirname+book+'/Hebrew/merged.json',
                author_dirname+book+".json")




