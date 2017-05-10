from __main__ import *
import time
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from bs4 import BeautifulSoup as Soup
import re
import pprint
import nltk
import os.path

def merge_sentences(s, l, n, min_num_words=6):
    result = []
    new_labels = []
    if n < 0:
        n = len(s)
    n = min(len(s), n)
    temp_data_list = []
    temp_labels_list = []
    for i in range(len(s)-n+1):
        if len(s[i].split()) < min_num_words:
            continue
        temp = ''
        for j in range(n):
            temp += s[i+j] + ' '
        result += [temp]
        new_labels += [l]
    return result, new_labels

def split_into_sentences(data, labels,n=-1):
    result = []
    new_labels = []
    for si,s in enumerate(data):
        split_sent, split_labels = merge_sentences(nltk.sent_tokenize(s),labels[si],n)
        result += split_sent
        new_labels += split_labels
    return result, new_labels


EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.30

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_SPLIT, random_state=42)

augs_max_sequence_length = 100
min_sequence_length = 30
min_aug_length = 10


base_path = '../data/Arxiv/'+department+'/cso/'

data_path = base_path+'data.npy'
labels_path = base_path+'labels.npy'


if os.path.isfile(data_path):
    data = np.load(data_path)
    labels = np.load(labels_path)
else:

	from nltk.corpus import stopwords
	stop_words = set(stopwords.words("english"))
	stop_words.update([chr(i) for i in range(97,123)])
	stop_words.update(['et','al', 'i.e.', 'ie', 'e.g.', 'eg', 'lt', 'gt'])

	import string
	#pattern = re.compile('[\W_]+')
	pattern = re.compile('[^a-zA-Z.]')
	print 'Reading dataset...'
	start = time.time()


	categories_path = '/home/ubuntu/MOUNT_POINT/JonaQ/Cybershare/data/Arxiv/' + department + '/categories.txt'
	text_path = '/home/ubuntu/MOUNT_POINT/JonaQ/Cybershare/data/Arxiv/' + department + '/text/'
	soup_tag = 'summary'
	#Read categories file
	categories = []
	with open(categories_path, 'r') as f:
	    for line in f:
		categories += [line.strip('\n')]

	data = []
	labels = []
	lbl = 0

	#	For each category
	for c, category in enumerate(categories):

		raw_data = open(text_path+category+'.txt', 'r').read()
		soup = Soup(raw_data, "lxml")
		text_data = soup.findAll(soup_tag)
		chunk = [pattern.sub(' ', str(message).strip('<'+soup_tag+'></'+soup_tag+'>').lower()) for message in text_data]
		chunk = [' '.join([word for word in message.split() if word not in stop_words]) for message in chunk]
		
		if len(chunk) >= 3000:
		    data += chunk
		    labels += [lbl for i in range(len(chunk))]
		    lbl+=1
		    print len(chunk),'abstracts for class',category

		    rnd_idx = np.random.randint(0, len(chunk), 5)
		    for ri in rnd_idx:
			print text_data[ri]
			print chunk[ri]
			print '\n'
        remove_idx = []
        for li, l in enumerate(data):
        #if example is padded, at least min length, and with csm aug
            if len(''.join([s for s in l])) < min_sequence_length:
                remove_idx += [li]
        data = np.delete(np.array(data), remove_idx, axis=0)
        labels = np.delete(np.array(labels), remove_idx, axis=0)

        np.save(data_path,data)
        np.save(labels_path,labels)

num_labels = len(set(labels))
print num_labels,'labels'

#   split into training and validation sets                                              
for train_idx, val_idx in sss.split(data, labels):     
    #Baseline data
    x_train = np.array(data)[train_idx]
    x_val = np.array(data)[val_idx]                              

    y_train = labels[train_idx]                        
    y_val = labels[val_idx]

if split_sentences:
    x_train, y_train = split_into_sentences(x_train,y_train,merge_size)

print x_train[0:5]
print y_train[0:5]

#Tokenize
tokenizer = Tokenizer(nb_words=num_words)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
rev_word_index = {}
for k, v in word_index.items():
    rev_word_index[v] = k
x_train = tokenizer.texts_to_sequences(x_train)

if split_sentences:
    unsplit_x_val = [s for s in x_val]
    unsplit_y_val = [l for l in y_val]
    x_val, y_val = split_into_sentences(x_val,y_val,merge_size)
    
x_val = tokenizer.texts_to_sequences(x_val)

#Create embedding matrix
#Remove top n words

#   Read embeddings
embeddings_path = '/home/ubuntu/MOUNT_POINT/JonaQ/Cybershare/data/glove.6B.100d.txt'
#   Load embeddings                                     
start = time.time()                                     
print 'Reading embeddings...'
embeddings_index = {}                                   
reverse_embeddings_index = {}
f = open(embeddings_path)                               
for line in f:                                          
    values = line.split()                               
    word = values[0]                                    
    coefs = np.asarray(values[1:], dtype='float32')     
    embeddings_index[word] = coefs                      
    reverse_embeddings_index[tuple(coefs)] = word
f.close()
print 'embeddings read after ', time.time()-start
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
	# words not found in embedding index will be all-zeros
	embedding_matrix[i] = embedding_vector

x_train = pad_sequences(x_train, maxlen=max_sequence_length)
x_val = pad_sequences(x_val, maxlen=max_sequence_length)

for n in range(n_common_words):
    embedding_matrix[n] = np.zeros(EMBEDDING_DIM)

y_train = to_categorical(np.asarray(y_train))
y_val = to_categorical(np.asarray(y_val))

print y_train[0:5]
