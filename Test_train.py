import pandas as pd 
import numpy as np 
import sys
import math
from sklearn.model_selection import train_test_split 

class split_train_test():
	def __init__(self,df,num_test_file):
		self.train_dt = None
		self.test_dt = None
		self.train_dt , self.test_dt = self.generate_train_test_data(df,0.25,num_test_file)
	def generate_train_test_data(self,df,test_s,num_test_file):
		df = df.drop(['user',	'gender',	'age',	'how_tall_in_meters'	,'weight',	'body_mass_index'],axis=1)
		X = df.drop(['class'],axis=1).values
		y = df['class'].values
		X_train,X_test , y_train, y_test = train_test_split(X,y,test_size=test_s,random_state = 101)
		X_test = np.split(X_test,num_test_file)
		y_test = np.split(y_test,num_test_file)
		return (X_train,y_train),(X_test,y_test)

