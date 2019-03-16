from util import *
from models.graphcnn import *
from main import train, test
from collections import namedtuple
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

args = namedtuple('args', ['iters_per_epoch', 'batch_size'])
args.iters_per_epoch = 50
args.batch_size = 32

graphs, num_classes = load_data("MUTAG", False)
train_graphs, test_graphs = separate_data(graphs, 0, 0)
N = np.array([[0.8, 0.2], [0.2, 0.8]])
train_graphs = corrupt_label(train_graphs, N)
device = "cuda:0"
model = GraphCNN(5, 2, train_graphs[0].node_features.shape[1],
                 64, 2, 0, False, 'sum', 'sum', device).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
