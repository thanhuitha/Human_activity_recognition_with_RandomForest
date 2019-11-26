from Model import Model
import sys


if __name__ == '__main__':
	option = int(sys.argv[1])
	if option == 1:
		print('--------Training_model-------')
		print('..............................')
		path_file = sys.argv[2]
		model = Model()
		model.train_data(path_file)
		print('--------Saved model -----------')
	if option == 2:
		print('------Classification --------')
		test = [[]]
		for i in sys.argv[2:]:
			test[0].append(i)
		model = Model()
		model.load_weight('rfc_model_trained.sav')
		with open('result_file.txt','w') as f:
			y_pred = model.predict_value(test)
			f.write(y_pred[0])
	if option == 3:
		print('------Classification multi test ----------')
		path_file = sys.argv[2]
		model = Model()
		model.load_weight('rfc_model_trained.sav')
		model.predict_file(path_file)

	if option == 4:
		print('---------Evaluate model-------------')
		num_test = int(sys.argv[2])
		model = Model()
		model.load_weight('rfc_model_trained.sav')
		model.get_report(num_test)