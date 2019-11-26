import pandas as pd 
import numpy as np 
import sys
import math

class preprocess_data():
	def __init__(self,df):
		self.data = df
		self.data = self.remove_null(df)
		self.data = self.drop_wrong_value(df)
	def remove_null(self,df):
		col = df.columns
		for index,c in enumerate(col):
			if '?' in list(df[c].unique()):
				df = df[df[c] != '?']
		return df
	def drop_wrong_value(self,df):
		for i in list(df['z4'].unique()):
			try:
				val = int(i)
			except Exception :
				wrong_value = i
				df = df[df['z4'] != wrong_value]
		df = df.astype({'z4':'int64'})
		return df

