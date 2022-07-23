'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
Jack Freeman
CS 251 Data Analysis Visualization, Spring 2020
'''
import re
import os
import numpy as np


def tokenize_words(text):
	'''Transforms an email into a list of words.'''
	
	# Define words as lowercase text with at least one alphabetic letter
	pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
	return pattern.findall(text.lower())


def count_words(email_path='data/enron'):
	'''Determine the count of each word in the entire dataset (across all emails)'''
	
	word_freq = {}
	num_emails = 0
	for root, dirs, files in os.walk(email_path):
		for file in files:
			filepath = root + os.sep + file
			if filepath.endswith('.txt'):
				num_emails += 1
				f = open(filepath,'r')
				tokenized = tokenize_words(f.read())
				for word in tokenized:
					if word_freq.get(word) == None:
						word_freq[word] = 1
					else:
						word_freq[word] += 1
	return word_freq, num_emails



def find_top_words(word_freq, num_features=200):
	'''Given the dictionary of the words that appear in the dataset and their respective counts,
	compile a list of the top `num_features` words and their respective counts.'''
	
	top_words = []
	counts = []
	sort = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
	for i in range(num_features):
		if (i == len(sort)):
			break
		top_words.append(sort[i][0])
		counts.append(sort[i][1])
	
	return top_words, counts


def make_feature_vectors(top_words, num_emails, email_path='data/enron'):
	'''Count the occurance of the top W (`num_features`) words in each individual email, turn into
	a feature vector of counts.'''
	
	
	feats = np.zeros((num_emails,len(top_words)))
	y = np.zeros(num_emails)
	email = 0
	for root, dirs, files in os.walk(email_path):
		for file in files:
			filepath = root + os.sep + file
			if (filepath.endswith('.txt')):
				y[email] = int(root.endswith('spam'))
				f = open(filepath,'r')
				tokenized = tokenize_words(f.read())
				for word in tokenized:
					if (word in top_words):
						feats[email,top_words.index(word)] += 1
				email += 1
	
	return feats, y



def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
	'''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
	of each split is determined by `test_prop`.'''
	
	
	inds = np.arange(y.size)
	if shuffle:
		features = features.copy()
		y = y.copy()

		inds = np.arange(y.size)
		np.random.shuffle(inds)
		features = features[inds]
		y = y[inds]

	# Your code here:
	inds_train = inds[:int(y.size*(1-test_prop))]
	x_train = features[inds_train]
	y_train = y[inds_train]

	inds_test = inds[int(y.size*(1-test_prop)):]
	x_test = features[inds_test]
	y_test = y[inds_test]

	return x_train, y_train, inds_train, x_test, y_test, inds_test


def retrieve_emails(inds, email_path='data/enron'):
	'''Obtain the text of emails at the indices `inds` in the dataset.'''
	emails = []
	root, spam, ham = os.walk('data/enron')
	for ind in inds:
		if (ind >= len(spam[2])):
			filepath = root[0] + os.sep + root[1][1] + os.sep + ham[2][ind-len(spam[2])]
		else:
			filepath = root[0] + os.sep + root[1][0] + os.sep + spam[2][ind]
		f = open(filepath, 'r')
		emails.append(f.read())
	
	return emails

