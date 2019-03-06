import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def backward_correction(output, labels, C, device, nclass):
    '''
        Backward loss correction.

        output: raw (logits) output from model
        labels: true labels
        C: correction matrix
    '''
    softmax = nn.Softmax(dim=1)
    C_inv = np.linalg.inv(C).astype(np.float32)
    C_inv = torch.from_numpy(C_inv).to(device)
    label_oh = torch.FloatTensor(len(labels), nclass).to(device)
    label_oh.zero_()
    label_oh.scatter_(1,labels.view(-1,1),1)
    output = softmax(output)
    #output /= torch.sum(output, dim=-1, keepdim=True)
    output = torch.clamp(output, min=1e-5, max=1.0-1e-5)
    return -torch.mean(torch.matmul(label_oh, C_inv) * torch.log(output))


def forward_correction_xentropy(output, labels, C, device, nclass):
    '''
        Forward loss correction. In cross-entropy, softmax is the inverse
        link function.

        output: raw (logits) output from model
        labels: true labels
        C: correction matrix
    '''
    C = C.astype(np.float32)
    C = torch.from_numpy(C).to(device)
    softmax = nn.Softmax(dim=1)
    label_oh = torch.FloatTensor(len(labels), nclass).to(device)
    label_oh.zero_()
    label_oh.scatter_(1,labels.view(-1,1),1)
    output = softmax(output)
    #output /= torch.sum(output, dim=-1, keepdim=True)
    output = torch.clamp(output, min=1e-5, max=1.0-1e-5)
    return -torch.mean(label_oh * torch.log(torch.matmul(output, C)))


def compound_correction(output, labels, C, device, nclass):
    return backward_correction(output, labels, C, device, nclass) + \
           forward_correction_xentropy(output, labels, C, device, nclass)


def _C(model, candidates):
    '''
        Internal function to return the corruption matrix.

        model: pretrained model on noisy data
        candidates: dictionary from label to a representative sample
                    (TODO extension: list of samples)
    '''
    softmax = nn.Softmax(dim=1)
    C = np.zeros((model.num_classes, model.num_classes), dtype=float)
    model.eval()
    for class_id, list_x in candidates.items():
        scores = model([list_x])
        probs = softmax(scores)
        C[class_id] = probs.detach().cpu().numpy()
    print("Estimated: \n", C)
    return C


def estimate_C(model, graphs, anchors=None, C=None, est_mode="max"):
    '''
        Estimate the class corruption matrix C.
        There are 3 schemes:
            1. Estimate C from pretrained model.
            2. Estimate C from anchor nodes.
            3. C is given.

        model: model pretrained on noisy data.
        graphs: training input graphs.
        anchors: list or dict of nodes with exact label.
        C: exact corruption matrix. 
        est_mode: mode of estimation
    '''
    # Scheme 2 and 3
    if C is not None:
        return C
    elif anchors is not None:
        if type(anchors) is list:
            assert len(anchors) == model.num_classes, "Not enough samples!"
            candidates = dict()
            for class_id, x in enumerate(anchors): 
                candidates[class_id] = x
        elif type(anchors) is dict:
            candidates = anchors
        else:
            raise TypeError("Anchors must be list or dict.")
    else:  # Scheme 1
        n_classes = model.num_classes 
        scores, idx = model(graphs).max(dim=1)
        candidates = dict()
        min_val = torch.min(scores)
        max_val = torch.max(scores)
        temp = torch.ones_like(scores)
        for class_id in range(n_classes):
            if est_mode == "max":
                cand_id = torch.argmax(torch.where(idx==class_id,\
                                                   scores,\
                                                   temp*min_val)) 
            elif est_mode == "min":
                cand_id = torch.argmin(torch.where(idx==class_id,\
                                                   scores,\
                                                   temp*max_val)) 
            else:
                raise NotImplementedError("Should there be a better mode?")
            candidates[class_id] = graphs[cand_id] 
    return _C(model, candidates)
