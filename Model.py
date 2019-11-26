import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from Preprocessing_data import preprocess_data
from Test_train import split_train_test
import pandas as pd
import pickle
class Model():
	def __init__(self):
		self.rfc = RandomForestClassifier()
		self.accuracy = 0
		self.train = None
		self.test = None

	def train_data(self,path_file):
		df = pd.read_csv(path_file,delimiter=';')
		df = preprocess_data(df).data
		stt = split_train_test(df,4)
		self.train = stt.train_dt
		self.test = stt.test_dt
		self.rfc.fit(self.train[0],self.train[1])
		joblib.dump(self.rfc,'rfc_model_trained.sav')

	def predict_value(self,X_test):
		rfc_y_predict = self.rfc.predict(X_test)
		return rfc_y_predict

	def calculate_accuracy(self,X_test,y_test):
		y_predict = self.predict_value(X_test)
		self.accuracy = accuracy_score(y_test,y_predict)

	def get_report(self,num_test):
		df = pd.read_csv('data.csv',delimiter=';')
		df = preprocess_data(df).data
		stt = split_train_test(df,num_test_file=num_test)
		self.train = stt.train_dt
		self.test = stt.test_dt
		test = self.test
		for i in range(num_test):
			with open('test_'+str(i),'wb') as f:
				pickle.dump((test[0][i],test[1][i]),f)
		report = ''
		for i in range(num_test):
			report += 'Test_file_ '+str(i)+'\n'
			y_pred = self.rfc.predict(test[0][i])
			report += classification_report(test[1][i],y_pred)
			report += '\n'
		with open('Evaluate_Model.txt','w') as f:
			f.write(report)

	def save_weight(self,path_file_save):
		joblib.dump(self.rfc,path_file_save)

	def load_weight(self,path_file):
		self.rfc = joblib.load(path_file)
	def predict_file(self,path_file):
		data = None
		with open(path_file,'r') as f:
			data = f.read()
		test = data.split('\n')
		n = len(test)
		with open('result_file.txt','w') as f:
			for i in range(n):
				tmp = test[i].split(' ')
				for i,vl in enumerate(tmp):
					tmp[i] = int(vl)
				tmp=[tmp]
				y_pred = self.predict_value(tmp)
				f.write(y_pred[0]+'\n')	