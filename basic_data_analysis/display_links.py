import os
import json
import pathlib
import shutil
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

dataset_dirname = "./sample_dataset/"
raw_subdirname = "raw/"
raw_metadata_subdirame = "_schemas/"
organized_subdirname = "organized/"
preprocessed_subdirname = "preprocessed/"

masekhtot = ['Arakhin', 'Bekhorot', 'Chullin', 'Keritot', 'Meilah', 'Menachot', 'Tamid', 'Temurah', 'Zevachim', 'Beitzah', 'Chagigah', 'Eruvin', 'Megillah', 'Moed Katan', 'Pesachim', 'Rosh Hashanah', 'Shabbat', 'Sukkah', 'Taanit', 'Yoma', 'Gittin', 'Ketubot', 'Kiddushin', 'Nazir', 'Nedarim', 'Sotah', 'Yevamot', 'Avodah Zarah', 'Bava Batra', 'Bava Kamma', 'Bava Metzia', 'Horayot', 'Makkot', 'Sanhedrin', 'Shevuot', 'Niddah', 'Berakhot']

torah_books = ['Deuteronomy', 'Exodus', 'Genesis', 'Leviticus', 'Numbers']

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

#get links count list
links_count_fn = './raw_dataset/Sefaria-Export-master/links/links_by_book_without_commentary.csv'
all_links_counts = pd.read_csv(links_count_fn)
all_link_counts_filtered = all_links_counts[all_links_counts['Link Count']>=0]
# book_authors_df = pd.DataFrame(authors_dict.items(), columns=['Book', 'Author'])
#join with authors
book_authors_df = pd.DataFrame.from_dict(authors_dict, columns=['Author'], orient='index')
df1 = all_link_counts_filtered.join(book_authors_df, on='Text 1', rsuffix='_1')
df2 = df1.join(book_authors_df, on='Text 2', rsuffix='_2')
df2.rename(columns={'Author': 'Author_1'}, inplace=True)

#keep only authors links
authors_links_count = df2.loc[:,['Author_1', 'Author_2', 'Link Count']]
#make all masekhtot as Talmud
authors_links_count.replace(masekhtot, 'Talmud', inplace=True)
#make all torah books as torah
authors_links_count.replace(torah_books, 'Torah', inplace=True)
#remove Jastrow
idx2 = ~((authors_links_count['Author_2']=='Marcus Jastrow') | (authors_links_count['Author_1']=='Marcus Jastrow'))
authors_links_count = authors_links_count.loc[idx2]
#remove self references
idx = ~authors_links_count['Author_1'].eq(authors_links_count['Author_2'])
authors_links_count = authors_links_count.loc[idx]
#aggregate identical rows
authors_links_count_agg = authors_links_count.groupby(['Author_1', 'Author_2']).sum()
authors_links_count_agg = authors_links_count_agg.reset_index() #or not?
authors_links_count_agg_morethan = authors_links_count_agg.loc[authors_links_count_agg['Link Count']>500]

#write to file
G_weighted = nx.Graph()
for index, row in authors_links_count_agg.iterrows():
    G_weighted.add_edge(row['Author_1'], row['Author_2'], weight=row['Link Count'])
nx.write_graphml(G_weighted, './sample_scripts/sefarialinksgraph.graphml')

#Vizualize
G_weighted_morethan = nx.Graph()
for index, row in authors_links_count_agg_morethan.iterrows():
    G_weighted_morethan.add_edge(row['Author_1'], row['Author_2'], weight=row['Link Count'])
nx.write_graphml(G_weighted_morethan, './sample_scripts/sefarialinksgraphmorethan500.graphml')

# plt.fig()
G = G_weighted_morethan

#code inspired by https://qxf2.com/blog/drawing-weighted-graphs-with-networkx/
pos=nx.spring_layout(G) 
nx.draw_networkx_nodes(G,pos, node_color='r')
nx.draw_networkx_labels(G,pos, font_color='b')
# nx.draw_networkx(G_weighted)
all_weights = []
for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight']) #we'll use this when determining edge thickness
#4 b. Get unique weights
unique_weights = list(set(all_weights))

#4 c. Plot the edges - one by one!
for weight in unique_weights:
    #4 d. Form a filtered list with just the weight you want to draw
    weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
    #4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
    width = weight*2.0*G.number_of_nodes()/sum(all_weights)
    nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width)


plt.show()
