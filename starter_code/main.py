### YOUR CODE HERE
import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split
from DataLoader import load_testing_images, trainset_load, testset_load
from Configure import model_configs, training_configs
from ImageUtils import visualize, transform_data, transform_val_data, transform_test_data


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict")
parser.add_argument("data_dir", help="path to the data")
#parser.add_argument("--save_dir", help="path to save the results")
args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(model_configs)

	if args.mode == 'train':
		
		x_train, y_train, x_test, y_test = load_data(args.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

		trainloader, valloader = transform_data(x_train, y_train, x_valid, y_valid)

		# print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape)

		# dataiter = iter(trainloader)
		# images, labels = dataiter.next()
		# print(type(images))
		# print(images.shape)
		# print(labels.shape)
  
		# for i, (images, labels) in enumerate(valloader):
		# 	print(type(images))
		# 	print(images.shape)
		# 	print(labels.shape)

		# print("trainloader ", trainloader)
		# print("valloader ", valloader)

		# model.train(x_train, y_train, training_configs, x_valid, y_valid) #TODO
		# model.evaluate(x_test, y_test) #TODO
  
		#trainloader = trainset_load()
		#valloader = testset_load()

		num_epochs = 2
		for epoch in range(1,num_epochs+1):
			model.train(epoch, trainloader = trainloader) #TODO
			model.test(epoch, testloader = valloader) #TODO


	elif args.mode == 'test':
		
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)

		testloader = transform_val_data(x_test, y_test)
		
		# print(x_test.shape, y_test.shape)
		# print(testloader)
		num_epochs = 2
		for epoch in range(1,num_epochs+1):
			model.test(epoch, testloader = testloader) #TODO

	elif args.mode == 'predict':
		data_dir = "../data2024/private_test_images_2024.npy"
		# Loading private testing dataset
		x_test = load_testing_images(data_dir)
		testloader = transform_test_data(x_test)
		# visualizing the first testing image to check your image shape
		visualize(x_test[0], 'test.png')
		# Predicting and storing results on private testing dataset 
		#predictions = model.predict_prob(testloader)
		#np.save(args.result_dir, predictions)
		

### END CODE HERE

