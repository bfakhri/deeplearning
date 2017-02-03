import numpy as np

class regressor(object):
	""" 
	Class that implements a regressor. After training, it will have weights that can be exported.  

	Args:
		
		dimensions: number of dimensions of dataset (optional, default randomly 15-30)
	"""    
	def __init__(self, train_data):
		X = train_data[0]
		Y = train_data[1]

		# Begin Regression
		#m1 = np.dot(np.transpose(X), X)
		#m2 = np.linalg.inv(m1)
		#m3 = np.dot(m2, np.transpose(X))
		#m4 = np.dot(m3, Y)
		#self.w = m4
	
		m1 = np.dot(np.transpose(X), X) + 21.521
		m2 = np.linalg.inv(m1)
		m3 = np.dot(m2, np.transpose(X))
		m4 = np.dot(m3, Y)
		self.w = m4
		self.b = np.zeros(train_data[0].shape[1])
	
	def get_params(self):
		return (self.w, self.b)

	def get_predictions(self, test_data):
		return np.transpose(np.dot(np.transpose(self.w), np.transpose(test_data)))
