### YOUR CODE HERE
import torch
import os, time
import numpy as np
from Network import SimpleDLA
import torch.nn as nn
from utils import progress_bar
import sys

"""This script defines the training, validation and testing process.
"""

class MyModel(nn.Module):

    def __init__(self, configs):
        super(MyModel, self).__init__()
        self.configs = configs
        self.network = SimpleDLA(configs).to('cuda')

        self.lr = self.configs.learning_rate

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)


    def train(self, epoch, trainloader):
        
        print('\nEpoch: %d' % epoch)
        self.network.train()
        train_loss = 0
        correct = 0
        total = 0
        val_labels = []
        val_preds = []
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            inputs = inputs.to('cuda')
            # targets = targets.to('cpu')
            targets = torch.tensor(targets, dtype = torch.int64).to('cuda')
            self.optimizer.zero_grad()
            outputs = self.network(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


        if (epoch) % self.configs.save_interval == 0:
            self.save(epoch)
            
    def test(self, testloader):

        self.network.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs = inputs.to('cuda')
                # targets = targets.to('cpu')
                targets = torch.tensor(targets, dtype = torch.int64).to('cuda')
                outputs = self.network(inputs)

                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def predict_prob(self, testloader):
            
        self.network.eval()
        all_probs = []

        with torch.no_grad():
            for inputs in testloader:
                inputs = inputs.to('cuda')
                outputs = self.network(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                all_probs.append(probabilities.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        #np.save("predictions.npy", all_probs)
        return all_probs
        

    def save(self, epoch):
        checkpoint_path = os.path.join(self.configs.save_dir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.configs.save_dir, exist_ok = True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location = 'cpu')
        self.network.load_state_dict(ckpt, strict = True)
        print("Restored model parameters from {}".format(checkpoint_name))


### END CODE HERE