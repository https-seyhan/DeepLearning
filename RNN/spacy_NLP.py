import glob
import os
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from random import shuffle

nlp = spacy.load('en_core_web_md') #load pre-trained model
tokenizer = Tokenizer(nlp.vocab) #generate tokenizer
traindata = '/home/saul/deeplearning/aclImdb/train'

def pre_process_data(filepath):
	positive_path = os.path.join(filepath, 'pos')
	negative_path = os.path.join(filepath, 'neg')
	pos_label = 1
	neg_label = 0
	dataset = []
	for filename in glob.glob(os.path.join(positive_path, '*.txt')):
		with open(filename, 'r') as f:
			dataset.append((pos_label, f.read()))
	for filename in glob.glob(os.path.join(negative_path, '*.txt')):
		with open(filename, 'r') as f:
			dataset.append((neg_label, f.read()))
	shuffle(dataset)
	return dataset

def tokenize_and_vectorized(dataset):
	vectorized_data = []
	for sample in dataset:
		tokens = tokenizer.tokenize(sample[1]) #tokize sentences 
		sample_vecs = []
		print(tokens)
		for token in tokens:
			try:
				sample_vecs.append(word_vectors[token])
			except KeyError:
				pass
		vectorized_data.append(sample_vecs)
	return vectorized_data

def tok_vec(dataset):
	vectorized_data = []
	for sample in dataset:	
		tokens = tokenizer(sample[1])
		sample_vecs = []

		for token in tokens:
			try:
				sample_vecs.append(nlp(str(token)).vector)
				print('A token :', token, 'Its vector :', len(nlp(str(token)).vector))
			except KeyError:
				pass
	#doc = nlp(str(dataset[1]))

if __name__ == '__main__':
	dataset = pre_process_data(traindata)
	tok_vec(dataset) # Tokenise and Vectorise the data
	#vectorized_data = tokenize_and_vectorized(dataset)
