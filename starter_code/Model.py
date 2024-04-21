### YOUR CODE HERE
import torch
import os, time
import numpy as np
from Network import MyNetwork, SimpleDLA
#from ImageUtils import parse_record
import torch.nn as nn
from utils import progress_bar
import sys

"""This script defines the training, validation and testing process.
"""

class MyModel(nn.Module):

    def __init__(self, configs):
        super(MyModel, self).__init__()
        self.configs = configs
        #self.network = MyNetwork(configs)
        self.network = SimpleDLA(configs).to('cpu')

        self.lr = self.configs.learning_rate

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    # def model_setup(self):
    #     pass

    # def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
    #     pass

    # def evaluate(self, x, y):
    #     pass

    # def predict_prob(self, x):
    #     pass

    def train(self, epoch, trainloader):

        # if device == 'cuda':
        #     net = torch.nn.DataParallel(net)
        #     cudnn.benchmark = True
        
        print('\nEpoch: %d' % epoch)
        self.network.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):

            inputs = inputs.to('cpu')
            # targets = targets.to('cpu')
            targets = torch.tensor(targets, dtype = torch.int64).to('cpu')
            self.optimizer.zero_grad()
            outputs = self.network(inputs)

            # print("targets: ", targets.shape)
            # print("outputs: ", outputs.shape)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    def test(self, epoch, testloader):
        global best_acc
        best_acc = 0
        self.network.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs = inputs.to('cpu')
                # targets = targets.to('cpu')
                targets = torch.tensor(targets, dtype = torch.int64).to('cpu')
                outputs = self.network(inputs)
                # print("targets: ", targets.shape)
                # print("outputs: ", outputs.shape)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
        # Save checkpoint.
        #acc = 100.*correct/total
        
        # if acc > best_acc:
        #     print('Saving..')
        #     state = {
        #         'net': self.network.state_dict(),
        #         'acc': acc,
        #         'epoch': epoch,
        #     }
        #     if not os.path.isdir('checkpoint'):
        #         os.mkdir('checkpoint')
        #     torch.save(state, './checkpoint/ckpt.pth')
        #     best_acc = acc

        if (epoch) % self.configs.save_interval == 0:
            self.save(epoch)

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