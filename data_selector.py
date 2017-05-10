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


EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.30

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_SPLIT, random_state=42)

augs_max_sequence_length = 100
min_sequence_length = 30
min_aug_length = 10


base_path = '../data/Arxiv/'+department+'/cso/'
x_train_path = base_path + 'x_train.npy'
x_val_path = base_path + 'x_val.npy'

mirror_pad_x_train_path = base_path + 'mirror_pad_x_train.npy'
mirror_pad_x_val_path = base_path + 'mirror_pad_x_val.npy'

cso_pad_x_train_path = base_path + 'cso_pad_x_train.npy'
cso_pad_x_val_path = base_path + 'cso_pad_x_val.npy'

lsh_pad_x_train_path = base_path + 'lsh_pad_x_train.npy'
lsh_pad_x_val_path = base_path + 'lsh_pad_x_val.npy'

cso_x_train_path = base_path + 'cso_x_train.npy'
cso_x_val_path = base_path + 'cso_x_val.npy'

lsh_x_train_path = base_path + 'lsh_x_train.npy'
lsh_x_val_path = base_path + 'lsh_x_val.npy'

y_train_path = base_path + 'y_train.npy'
y_val_path = base_path + 'y_val.npy'

tokenizer_path = base_path + 'tokenizer.npy'
embedding_matrix_path = base_path + 'embedding_matrix.npy'


#if False:
if os.path.isfile(x_train_path):

    x_train = np.load(x_train_path)
    x_val = np.load(x_val_path)

    mirror_pad_x_train = np.load(mirror_pad_x_train_path)
    mirror_pad_x_val = np.load(mirror_pad_x_val_path)

    cso_pad_x_train = np.load(cso_pad_x_train_path)
    cso_pad_x_val = np.load(cso_pad_x_val_path) 
    
    cso_x_train = np.load(cso_x_train_path)
    cso_x_val = np.load(cso_x_val_path)

    y_train = np.load(y_train_path)
    y_val = np.load(y_val_path)

    tokenizer = np.load(tokenizer_path).item()
    word_index = tokenizer.word_index
    embedding_matrix = np.load(embedding_matrix_path)

    num_labels = y_val.shape[1]
    print num_labels

    x_train = pad_sequences(x_train, maxlen=max_sequence_length)
    x_val = pad_sequences(x_val, maxlen=max_sequence_length)

    print x_train[0]
    print y_train[0]
    print '\n'
    print x_train[1]
    print y_train[1]
    print len(y_train)

    #mirror_pad_x_train = pad_sequences(mirror_pad_x_train, maxlen=max_sequence_length)
    #mirror_pad_x_val = pad_sequences(mirror_pad_x_val, maxlen=max_sequence_length)

    cso_pad_x_train = pad_sequences(cso_pad_x_train, maxlen=max_sequence_length)
    cso_pad_x_val = pad_sequences(cso_pad_x_val, maxlen=max_sequence_length)

    #cso_x_train = pad_sequences(cso_x_train, maxlen=max_sequence_length)
    #cso_x_val = pad_sequences(cso_x_val, maxlen=max_sequence_length)
    
else:

	from nltk.corpus import stopwords
	stop_words = set(stopwords.words("english"))
	stop_words.update([chr(i) for i in range(97,123)])
	stop_words.update(['et','al'])

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
	stop_words = set(stopwords.words("english"))
	stop_words.update([chr(i) for i in range(97,123)])
	stop_words.update(['et','al'])


	from CSM import CSM_2
	from nltk import tokenize
	tagger_path = 'Chunking/chunker2.pickle'
	with open(tagger_path, 'rb') as handle:
	    chunker = pickle.load(handle)
	tags = ['NN','NNS', 'JJ', 'RB', 'VBP', 'VB']
	csm = CSM_2(chunker, embeddings_index, tags, EMBEDDING_DIM)

	from textblob import Blobber
	from textblob_aptagger import PerceptronTagger
	tagger = Blobber(pos_tagger=PerceptronTagger())
	#pip install -U git+https://github.com/sloria/textblob-aptagger.git@dev

	import string
	pattern = re.compile('[\W_.,]+')
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
	cso_pad_data = []
	cso_data = []
	labels = []
	lbl = 0

	#	For each category
	for c, category in enumerate(categories):

		raw_data = open(text_path+category+'.txt', 'r').read()
		soup = Soup(raw_data, "lxml")
		text_data = soup.findAll(soup_tag)
		chunk = [pattern.sub(' ', str(message).strip('<'+soup_tag+'></'+soup_tag+'>').lower()) for message in text_data]

		cso_file = '../data/Arxiv/' + department + '/cso/augs_'+ category +'.npy'

		if os.path.isfile(cso_file):
		    print 'loading', cso_file
		    cso_keywords = np.load(cso_file)
		    cso_keywords = cso_keywords.tolist()
		else:
		    #CSO Augmentation
		    cso_keywords = [nltk.sent_tokenize(str(message.encode('utf-8')).strip('<'+soup_tag+'></'+soup_tag+'>').lower()) for message in text_data]
		    print 'sentence tokenization complete'
		    cso_keywords = [[re.sub('[^0-9a-zA-Z]+', ' ', sentence) for sentence in message] for message in cso_keywords]
		    cso_keywords = [[str(' '.join([word for word in sentence.split() if word not in stop_words])) for sentence in message] for message in cso_keywords]
		    print 'pattern removal complete'
		    cso_keywords = [[(tagger(sentence)).tags for sentence in message] for message in cso_keywords]
		    print 'PoS tagging complete'
		    cso_keywords = [[item for sublist in csm.select_keywords(message, 'optim') for item in sublist] for message in cso_keywords]
		    print 'CSO completed'
		    #cso_keywords = [str(' '.join([word for word in message if word not in stop_words])) for message in cso_keywords]
		    cso_keywords = [str(' '.join([word for word in message])) for message in cso_keywords]
		    np.save(cso_file, cso_keywords)

		
		chunk = [' '.join([word for word in message.split()]) for message in chunk]
		cso_pad_chunk = [chunk[i]+' '+cso_keywords[i] for i in range(len(chunk))]

		if len(chunk) >= 3000:
		    data += chunk
		    cso_pad_data += cso_pad_chunk
		    cso_data += cso_keywords
		    labels += [lbl for i in range(len(chunk))]
		    lbl+=1
		    print len(chunk),'abstracts for class',category

		    """rnd_idx = np.random.randint(0, len(chunk), 5)
		    for ri in rnd_idx:
			print text_data[ri]
			print cso_keywords[ri]
			print '\n'"""
        remove_idx = []
        for li, l in enumerate(data):
        #if example is padded, at least min length, and with csm aug
            if len(''.join([s for s in l])) < min_sequence_length:
                remove_idx += [li]
        data = np.delete(np.array(data), remove_idx, axis=0)
        labels = np.delete(np.array(labels), remove_idx, axis=0)
        mirror_pad_data = np.copy(data)

	print 'set(labels)',set(labels)
	num_labels = len(set(labels))
	print num_labels,'labels'



	#Neighbor keywords
	"""from sklearn.neighbors import LSHForest
	from sklearn.externals import joblib 
	lsh_file = '../data/Arxiv/' + department + '/cso/lsh_data.npy'
	lsh_pad_data_file = '../data/Arxiv/' + department + '/cso/lsh_pad_data.npy'
	if os.path.isfile(lsh_file) and os.path.isfile(lsh_pad_data_file):
	    lsh_data = np.load(lsh_file)
	    lsh_data = np.load(lsh_pad_data_file)
	else:
	    lshf = LSHForest(random_state=42)
	    lsh_map = list(set([rev_word_index[word] for row in cso_data for word in row if rev_word_index[word] in embeddings_index]))
	    lsh_mat = np.stack([embeddings_index.get(i) for i in lsh_map])
	    lsh_list = [[embeddings_index.get(rev_word_index[word]) for word in row if rev_word_index[word] in embeddings_index] for row in cso_data]
	    lsh_len = len(lsh_list)
	    print lsh_mat.shape
	    print lsh_mat[0]
	    lshf.fit(lsh_mat)

	    for r, row in enumerate(lsh_list):
		if len(row) < min_aug_length:
		    continue
		print type(row)
		print len(row)
		for ar in row:
		    print type(ar)
		    print ar.shape
		_, indices = lshf.kneighbors(np.stack(row), 2)
		
		lsh_addition = [word_index.get(lsh_map[i[1]]) for i in indices]
		lsh_data[r] = lsh_data[r] + lsh_addition
		lsh_pad_data[r] = lsh_pad_data[r] + lsh_addition

		if r%1000 == 0:
		    print r,'/',lsh_len
		    print row
		    print [lsh_map[j] for i in indices for j in i]
		
	    np.save(lsh_file, lsh_data)
	lsh_pad_data = np.delete(np.array(lsh_pad_data), remove_idx, 0)"""

	labels = to_categorical(np.asarray(labels))



	#   split into training and validation sets                                              
	for train_idx, val_idx in sss.split(data, labels):     
            #Baseline data
	    x_train = np.array(data)[train_idx]
	    x_val = np.array(data)[val_idx]                              
            #Wrapped padded data
	    mirror_pad_x_train = np.array(mirror_pad_data)[train_idx]
	    mirror_pad_x_val = np.array(mirror_pad_data)[val_idx]
            #CSO pad data
	    cso_pad_x_train = np.array(cso_pad_data)[train_idx]
	    cso_pad_x_val = np.array(cso_pad_data)[val_idx]

	    cso_x_train = np.array(cso_data)[train_idx]
	    cso_x_val = np.array(cso_data)[val_idx]

	    y_train = labels[train_idx]                        
	    y_val = labels[val_idx]

	#Tokenize
	tokenizer = Tokenizer(nb_words=num_words)
	tokenizer.fit_on_texts(x_train)
	word_index = tokenizer.word_index
	rev_word_index = {}
	for k, v in word_index.items():
	    rev_word_index[v] = k

	x_train = tokenizer.texts_to_sequences(x_train)
	x_val = tokenizer.texts_to_sequences(x_val)

	cso_pad_x_train = tokenizer.texts_to_sequences(cso_pad_x_train)
	cso_pad_x_val = tokenizer.texts_to_sequences(cso_pad_x_val)


	cso_pad_data = tokenizer.texts_to_sequences(cso_pad_data)
	lsh_pad_data = np.copy(cso_pad_data)
	cso_data = tokenizer.texts_to_sequences(cso_data)
	lsh_data = np.copy(cso_data)

	#    Compute length of texts and of augmentations (noun phrases)
	lengths = [len(i) for i in data]
	cso_pad_lengths = [len(i) for i in cso_pad_data]
	cso_lengths = [len(i) for i in cso_data]
	print 'Data length stats before padding:'
	print 'min:', min(lengths), 'max:', max(lengths), 'mean:', sum(lengths)/float(len(lengths))
	print 'Data length stats of cso extracted nps:'
	print 'min:', min(cso_lengths), 'max:', max(cso_lengths), 'mean:', int(sum(cso_lengths)/float(len(cso_lengths)))
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
		# words not found in embedding index will be all-zeros
		embedding_matrix[i] = embedding_vector

        np.save(x_train_path, x_train)
        np.save(x_val_path, x_val)

        np.save(mirror_pad_x_train_path, mirror_pad_x_train)
        np.save(mirror_pad_x_val_path, mirror_pad_x_val)

        np.save(cso_pad_x_train_path, cso_pad_x_train)
        np.save(cso_pad_x_val_path, cso_pad_x_val)

        np.save(cso_x_train_path, cso_x_train)
        np.save(cso_x_val_path, cso_x_val)

        np.save(y_train_path, y_train)
        np.save(y_val_path, y_val)

        np.save(tokenizer_path, tokenizer)
        np.save(embedding_matrix_path, embedding_matrix)
	
	x_train = pad_sequences(x_train, maxlen=max_sequence_length)
	x_val = pad_sequences(x_val, maxlen=max_sequence_length)

        """mirror_pad_x_train = pad_sequences(mirror_pad_x_train, maxlen=max_sequence_length)
	mirror_pad_x_val = pad_sequences(mirror_pad_x_val, maxlen=max_sequence_length)"""

	cso_pad_x_train = pad_sequences(cso_pad_x_train, maxlen=max_sequence_length)
	cso_pad_x_val = pad_sequences(cso_pad_x_val, maxlen=max_sequence_length)

        """cso_x_train = pad_sequences(cso_x_train, maxlen=max_sequence_length)
	cso_x_val = pad_sequences(cso_x_val, maxlen=max_sequence_length)"""

