'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Jack Freeman
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np


class NaiveBayes:
	'''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
	 number of classes)'''
	def __init__(self, num_classes):
		'''Naive Bayes constructor	'''
		
		
		# class_priors: ndarray. shape=(num_classes,).
		#	Probability that a training example belongs to each of the classes
		#	For spam filter: prob training example is spam or ham
		self.class_priors = None
		# class_likelihoods: ndarray. shape=(num_classes, num_features).
		#	Probability that each word appears within class c
		self.class_likelihoods = None
		self.num_classes = num_classes

	def train(self, data, y):
		'''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
		class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
		class likelihoods (the probability of a word appearing in each class — spam or ham)'''
		
		
		num_samps, num_features = data.shape
		self.class_priors = np.zeros(self.num_classes)
		self.class_likelihoods = np.zeros((self.num_classes,num_features))

		for i in range(self.num_classes):
			self.class_priors[i] = y[y == i].shape[0]/num_samps
			for j in range(num_features):
				self.class_likelihoods[i,j] = (np.sum(data[y == i,j]) + 1)/(np.sum(data[y == i]) + num_features)

	def predict(self, data):
		'''Combine the class likelihoods and priors to compute the posterior distribution. The
		predicted class for a test sample from `data` is the class that yields the highest posterior
		probability.'''
		
		return np.argmax(np.log(self.class_priors) + data @ (np.log(self.class_likelihoods)).T,1)

	def accuracy(self, y, y_pred):
		'''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
		that match the true values `y`.'''
		
		return y_pred[y_pred == y].shape[0]/y_pred.shape[0]

	def confusion_matrix(self, y, y_pred):
		'''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
		by the classifier (`y_pred`).'''
		
		confusion_matrix = np.empty((self.num_classes, self.num_classes), dtype = np.int)
		for i in (range(self.num_classes)):
			for j in (range(self.num_classes)):
				confusion_matrix[i, j] = len(np.where((y == i)*(y_pred == j))[0])
		return confusion_matrix