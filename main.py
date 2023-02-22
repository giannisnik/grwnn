import argparse
import json

import torch
import torch.nn.functional as F
from torch import optim

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree

from model import GRWNN

lamdas = {'MUTAG': 1.0/5, 'DD': 1.0/20, 'NCI1': 1.0/20, 'PROTEINS_full': 1.0/30, 'ENZYMES': 1.0/20, 'IMDB-BINARY': 1.0/200,'IMDB-MULTI': 1.0/300, 'REDDIT-BINARY': 1.0/500, 'REDDIT-MULTI-5K': 1.0/400, 'COLLAB': 1.0/2000}

unlabeled_datasets = ['IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB']

class Degree(object):
    def __call__(self, data):
        idx = data.edge_index[0]
        deg = degree(idx, data.num_nodes, dtype=torch.float)
        data.x = deg.unsqueeze(1)
        return data

# Argument parser
parser = argparse.ArgumentParser(description='GRWNN')
parser.add_argument('--dataset', default='ENZYMES', help='Dataset name')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=300, metavar='N', help='Number of epochs to train')
parser.add_argument('--hidden-graphs', type=int, default=64, metavar='N', help='Number of hidden graphs')
parser.add_argument('--hidden-nodes', type=int, default=5, metavar='N', help='Number of nodes of each hidden graph')
parser.add_argument('--hidden-dim', type=int, default=32, metavar='N', help='Size of hidden layer of NN')
args = parser.parse_args()

use_node_attr = False
if args.dataset == 'ENZYMES' or args.dataset == 'PROTEINS_full':
    use_node_attr = True

if args.dataset in unlabeled_datasets:
    dataset = TUDataset(root='./datasets/'+args.dataset, name=args.dataset, transform=Degree())
else:
    dataset = TUDataset(root='./datasets/'+args.dataset, name=args.dataset, use_node_attr=use_node_attr)

lamda = lamdas[args.dataset]

with open('data_splits/'+args.dataset+'_splits.json','rt') as f:
    for line in f:
        splits = json.loads(line)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(epoch, loader, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def val(loader):
    model.eval()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        loss_all += F.nll_loss(model(data), data.y, reduction='sum').item()
    return loss_all / len(loader.dataset)


def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

acc = []
for i in range(10):
    model = GRWNN(dataset.num_features, args.hidden_graphs, args.hidden_nodes, args.hidden_dim, lamda, dataset.num_classes, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_index = splits[i]['model_selection'][0]['train']
    val_index = splits[i]['model_selection'][0]['validation']
    test_index = splits[i]['test']

    test_dataset = dataset[val_index]
    val_dataset = dataset[val_index]
    train_dataset = dataset[train_index]

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print('---------------- Split {} ----------------'.format(i))

    best_val_loss, test_acc = 100, 0
    for epoch in range(1, args.epochs+1):
        train_loss = train(epoch, train_loader, optimizer)
        val_loss = val(val_loader)
        if best_val_loss >= val_loss:
            test_acc = test(test_loader)
            best_val_loss = val_loss
        if epoch % 20 == 0:
            print('Epoch: {:03d}, Train Loss: {:.7f}, '
                    'Val Loss: {:.7f}, Test Acc: {:.7f}'.format(
                    epoch, train_loss, val_loss, test_acc))
    acc.append(test_acc)
acc = torch.tensor(acc)
print('---------------- Final Result ----------------')
print('Mean: {:7f}, Std: {:7f}'.format(acc.mean(), acc.std()))
