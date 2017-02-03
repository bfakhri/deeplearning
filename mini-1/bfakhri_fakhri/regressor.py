"""
Author: Bijan Fakhri
Date: 2/3/17
CSE591 - Intro to Deep Learning

This file implements the regressor class which performs a linear regression given some data
"""

import numpy as np

class regressor(object):
	""" 
	Class that implements a regressor. After training, it will have weights that can be exported.  

	Args:
		train_data: data with which to train the regressor. 
		alpha (optional): sets the regularization parameter alpha		
	"""    
	def __init__(self, train_data, **kwargs):
		
		# Regularization Param
		if 'alpha' in kwargs.keys():
			self.alpha = kwargs['alpha']
		else:
			self.alpha = 1
	
		# Changes name of incoming data for clearer representation below	
		X = train_data[0]
		#X = np.append(X, np.ones(X.shape[1]), axis=0)
		ones_c = np.ones((X.shape[0], 1))
		X = np.concatenate((X, ones_c), axis=1)
		Y = train_data[1]

		# Begin Regression	
		m1_1 = np.dot(np.transpose(X), X) 
		m1_2 = np.multiply(self.alpha, np.identity(X.shape[1]))
		m1 = m1_1 + m1_2
		m2 = np.linalg.inv(m1)
		m3 = np.dot(m2, np.transpose(X))
		m4 = np.dot(m3, Y)
		self.w = m4
	
	def get_params(self):
		return self.w

	def get_predictions(self, test_data):
		ones_c = np.ones((test_data.shape[0], 1))
		test_data = np.concatenate((test_data, ones_c), axis=1)
		return np.transpose(np.dot(np.transpose(self.w), np.transpose(test_data)))

if __name__=='__main__':
	pass
