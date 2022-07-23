'''knn.py
K-Nearest Neighbors algorithm for classification
Jack Freeman
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from palettable import cartocolors


class KNN:
	'''K-Nearest Neighbors supervised learning algorithm'''
	def __init__(self, num_classes):
		'''KNN constructor'''
		
		# exemplars: ndarray. shape=(num_train_samps, num_features).
		#	Memorized training examples
		self.exemplars = None
		# classes: ndarray. shape=(num_train_samps,).
		#	Classes of memorized training examples
		self.classes = None
		self.num_classes = num_classes

	def train(self, data, y):
		'''Train the KNN classifier on the data `data`, where training samples have corresponding
		class labels in `y`.'''
		
		
		self.exemplars = data
		self.classes = y.astype(int)

	def predict(self, data, k):
		'''Use the trained KNN classifier to predict the class label of each test sample in `data`.
		Determine class by voting: find the closest `k` training exemplars (training samples) and
		the class is the majority vote of the classes of these training exemplars.'''
				
		predicted = np.zeros((data.shape[0]))

		for i in range(data.shape[0]):
			d = data[i, :]
			dist = np.sqrt(np.sum((self.exemplars - d)**2, axis = 1))
			idx = np.argsort(dist)
			sorted_ = self.classes[idx]
			unique, counts = np.unique(sorted_[0:k], return_counts = True)
			predicted[i] = unique[ np.argmax(counts)]
		return predicted

	def accuracy(self, y, y_pred):
		'''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
		that match the true values `y`.	'''
		
		return y[y == y_pred].shape[0]/y.shape[0]

	def plot_predictions(self, k, n_sample_pts):
		'''Paints the data space in colors corresponding to which class the classifier would
		 hypothetically assign to data samples appearing in each region.'''
		
		
		bold4 = cartocolors.qualitative.Bold_4.mpl_colors
		cmap = ListedColormap(bold4)
		samp_vec = np.linspace(-40,40,n_sample_pts)
		x, y = np.meshgrid(samp_vec,samp_vec)
		x_samp = np.matrix.flatten(x)[:, np.newaxis]
		y_samp = np.matrix.flatten(y)[:, np.newaxis]
		coor = np.hstack((x_samp,y_samp))
		results = np.reshape(self.predict(coor, k), x.shape)
		

		plt.figure(figsize=(20,20))
		plt.pcolormesh(x, y, results, cmap=cmap, shading='auto')
		plt.colorbar()

		plt.xlabel("X")
		plt.ylabel("Y")
		title = "K = " + str(k)
		plt.title(title)
		
	def confusion_matrix(self, y, y_pred):
		'''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
		by the classifier (`y_pred`).'''
		
		confusion_matrix = np.empty((self.num_classes, self.num_classes), dtype = np.int)
		for i in (range(self.num_classes)):
			for j in (range(self.num_classes)):
				confusion_matrix[i, j] = len(np.where((y == i)*(y_pred == j))[0])
		return confusion_matrix
