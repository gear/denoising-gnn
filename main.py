import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, separate_data, corrupt_label
from models.graphcnn import GraphCNN
from models.loss import estimate_C, backward_correction, \
                        forward_correction_xentropy


def _parse_anchors(str_anchors, g_list):
    if len(str_anchors) == 0:
        return None
    anchors = dict()
    for class_idx, graph_idx in enumerate(str_anchors.split(' ')):
        anchors[class_idx] = g_list[int(graph_idx)]
    return anchors


def train(args, model, device, train_graphs, optimizer, epoch, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.train()
    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')
    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)
        labels = torch.LongTensor([graph.label for graph in batch_graph])\
                    .to(device)
        #compute loss
        loss = criterion(output, labels)
        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        loss = loss.detach().cpu().numpy()
        loss_accum += loss
        #report
        pbar.set_description('epoch: %d' % (epoch))
    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    return average_loss

###pass data to model with minibatch during testing to avoid memory 
###overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test

def main():
    # Training settings
    # Note: Check experiment scripts for hyperparameters
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional\
                                                  neural net for whole-graph \
                                                  classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=str, default="0",
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10\
                              (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. \
                              Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one \
                              (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one \
                              (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", 
                        choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", 
                        choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average\
                              or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon \
                                              weighting for the center nodes.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of \
                              nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    parser.add_argument('--bn', type=bool, default=True, help="Enable batchnorm\
                                                               for MLP")
    parser.add_argument('--gbn', type=bool, default=True, help="Enable \
                                                    batchnorm for graph")
    parser.add_argument('--corrupt_label', action="store_true",
                        help="Enable label corruption")
    parser.add_argument('--N', type=str, default="",
                        help="Label noise configuration N. \
                              Should be passed as a flattened\
                               string with row order or a single\
                                value for symmetrix noise config.")
    parser.add_argument('--denoise', type=str, default="",
                        choices=["estimate", "anchors", "exact"],
                        help="Method to recover the noise matrix C.")
    parser.add_argument('--correction', type=str, default="backward",
                        choices=["backward", "forward"],
                        help="Type of loss correction function.")
    parser.add_argument('--anchors', type=str, default="",
                        help="List of representative train data.")
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    if args.device != "cpu":
        device = torch.device("cuda:" + args.device)\
                 if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    # Corrupt data 
    if args.corrupt_label:
        assert len(args.N) != 0, "Need to pass noise matrix!"
        N = np.fromstring(args.N, sep=" ", dtype=float)
        if len(N) == 1:
            self_prob = N[0]
            N = np.ones((num_classes, num_classes)) * \
                ((1 - self_prob) / (num_classes-1))
            np.fill_diagonal(N, self_prob)
            # Note: this could potentially cause some numerical problem
        elif len(N) == num_classes ** 2:
            N = N.reshape(num_classes, -1)        
        else:
            raise ValueError("N needs to be a single value or square matrix.")
        print("Corrupting training label with N:")
        print(N)
        train_graphs = corrupt_label(train_graphs, N)

    if args.denoise != "exact":
        model = GraphCNN(args.num_layers, 
                         args.num_mlp_layers, 
                         train_graphs[0].node_features.shape[1], 
                         args.hidden_dim, 
                         num_classes, 
                         args.final_dropout, 
                         args.learn_eps, 
                         args.graph_pooling_type, 
                         args.neighbor_pooling_type, 
                         device, args.bn, args.gbn).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        for epoch in range(1, args.epochs + 1):
            scheduler.step()

            avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
            acc_train, acc_test = test(args, model, device, train_graphs, 
                                       test_graphs, epoch)

            if not args.filename == "":
                with open(args.filename, 'w') as f:
                    f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                    f.write("\n")
            print("")

            print(model.eps)
    else:
        model = None
    
    if args.denoise in ["estimate", "anchors", "exact"]:
        C = None
        anchors = None
        if args.denoise == "estimate" or args.denoise == "anchors":
            anchors = _parse_anchors(args.anchors, train_graphs)
            C = estimate_C(model, train_graphs, anchors)
        elif args.denoise == "exact": 
            C = estimate_C(model, train_graphs, anchors, N)

        criterion = None
        if args.correction == "backward":
            criterion = lambda x, y: backward_correction(x,
                                                         y,
                                                         C,
                                                         device,
                                                         model.num_classes)
        elif args.correction == "forward":
            criterion = lambda x, y: forward_correction_xentropy(x,
                                                          y,
                                                          C,
                                                          device,
                                                          model.num_classes)

        del model
        print("Training new denoising model")
        model = GraphCNN(args.num_layers, 
                         args.num_mlp_layers, 
                         train_graphs[0].node_features.shape[1], 
                         args.hidden_dim, 
                         num_classes, 
                         args.final_dropout, 
                         args.learn_eps, 
                         args.graph_pooling_type, 
                         args.neighbor_pooling_type, 
                         device, args.bn, args.gbn).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        for epoch in range(1, args.epochs + 1):
            scheduler.step()
            avg_loss = train(args, model, device, 
                             train_graphs, optimizer, epoch,
                             criterion)
            acc_train, acc_test = test(args, model, device, train_graphs, 
                                       test_graphs, epoch)
            if not args.filename == "":
                with open(args.denoise+'_'+args.correction+'_'+args.filename, 'w') as f:
                    f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                    f.write("\n")
            print("")
            print(model.eps)

if __name__ == '__main__':
    main()
