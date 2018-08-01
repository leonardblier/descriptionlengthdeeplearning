from __future__ import print_function
import argparse
import pyvarinf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--prior', type=str, default='gaussian', metavar='P',
                    help='prior used (default: gaussian)',
                    choices=['gaussian', 'mixtgauss', 'conjugate', 'conjugate_known_mean'])

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# setting up prior parameters
prior_parameters = {}
if args.prior != 'gaussian':
    prior_parameters['n_mc_samples'] = 1
    
if args.prior == 'mixtgauss':
    prior_parameters['sigma_1'] = 0.02
    prior_parameters['sigma_2'] = 0.2
    prior_parameters['pi'] = 0.5
if args.prior == 'conjugate':
    prior_parameters['mu_0'] = 0.
    prior_parameters['kappa_0'] = 3.
    prior_parameters['alpha_0'] = .5
    prior_parameters['beta_0'] = .5
if args.prior == 'conjugate_known_mean':
    prior_parameters['alpha_0'] = .5
    prior_parameters['beta_0'] = .5
    prior_parameters['mean'] = 0.

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('~/datasets', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('~/datasets', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

transform_train = transforms.Compose([
    #transforms.RandomCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    #transforms.RandomCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='~/datasets', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='~/datasets', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv11 = nn.Conv2d(3, 32, kernel_size=3)#, padding=2)
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3)#, padding=2)

        self.conv21 = nn.Conv2d(32, 64, kernel_size=3)#, padding=2)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3)#, padding=2)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.fc1 = nn.Linear(5*5*64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.max_pool2d(x, 2)
        
        #x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        #x = self.bn1(x)
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #x = self.bn2(x)

        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(F.relu(self.fc3(x)))

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 256)
        #self.fc1 = nn.Linear(1*28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        #x = self.bn1(x)
        x = F.relu(self.fc2(x))
        #x = self.bn2(x)
        x = self.fc3(x)
        return F.log_softmax(x)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.log_softmax(out)

model = Net()
var_model = pyvarinf.Variationalize(model)
var_model.set_prior(args.prior, **prior_parameters)
if args.cuda:
    var_model.cuda()

optimizer = optim.Adam(var_model.parameters(), lr=args.lr)
#optimizer = optim.SGD(var_model.parameters(), lr=args.lr)



def train(epoch):
    var_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = var_model(data)
        loss_error = F.nll_loss(output, target)
        loss_prior = var_model.prior_loss() / len(train_loader.dataset)
        loss = loss_error + loss_prior
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss error: {:.6f}\tLoss weights: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), loss_error.item(), loss_prior.item()))


def compressionscores():
    var_model.train()
    loss_prior = var_model.prior_loss().item()
    loss_error = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = var_model(data)
        loss_error += F.nll_loss(output, target, size_average=False).item()
        
    loss = loss_error + loss_prior
    print('Compression scores: DL: {:.0f}\tDL error: {:.0f}\tDL weights: {:.6f}\tCompRate: {:.4f}'.format(
        loss, loss_error, loss_prior, loss / (len(train_loader.dataset) * np.log(10))))
        
def test(epoch):
    var_model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = var_model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target).cpu().sum().item()

            test_loss = test_loss
            test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

compressionscores()
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    compressionscores()
