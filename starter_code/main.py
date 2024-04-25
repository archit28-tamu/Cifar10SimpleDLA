### YOUR CODE HERE
import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split
from DataLoader import load_testing_images
from Configure import model_configs, training_configs
from ImageUtils import visualize, transform_data, transform_val_data, transform_test_data


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict")
parser.add_argument("data_dir", help="path to the data")
parser.add_argument("--pred_data_dir", default="../data2024/private_test_images_2024.npy", help="path to private test data")
parser.add_argument("--result_dir", default="../data2024", help="path to predictions data")
args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(model_configs)

	if args.mode == 'train':
		
		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

		trainloader, valloader = transform_data(x_train, y_train, x_valid, y_valid)
		
		num_epochs = 50
		for epoch in range(1,num_epochs+1):
			model.train(epoch, trainloader = trainloader) 
			model.test(testloader = valloader) 


	elif args.mode == 'test':

		#model.load_state_dict(torch.load('../saved_models/model-50.ckpt'))
		model.load('../saved_models/model-50.ckpt')
		
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)

		testloader = transform_val_data(x_test, y_test)
		
		model.test(testloader = testloader) 

	elif args.mode == 'predict':
		
		model.load('../saved_models/model-50.ckpt')

		# Loading private testing dataset
		x_test = load_testing_images(args.pred_data_dir)
		testloader = transform_test_data(x_test)

		# visualizing the first testing image to check your image shape
		visualize(x_test[0], 'test.png')

		# Predicting and storing results on private testing dataset 
		predictions = model.predict_prob(testloader)

		os.makedirs(args.result_dir, exist_ok=True)
		np.save(os.path.join(args.result_dir, "predictions.npy"), predictions)
		

### END CODE HERE

