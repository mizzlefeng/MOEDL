'''
@author: feng
'''
import torch.nn as nn
import torch
import torch.nn.functional as F

def ce_loss_all(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    label=label.type(torch.float32)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return (A+B)

def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss_single(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    label = F.one_hot(p, num_classes=c)
    label=label.type(torch.float32)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    return A

class Classifier(nn.Module):
    def __init__(self, classifier_dims, classes, dropout):
        super(Classifier, self).__init__()
        self.num_layers = len(classifier_dims)
        self.fc = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i+1]))
        self.fc.append(nn.Dropout(dropout))
        self.fc.append(nn.Linear(classifier_dims[self.num_layers-1], 64))
        self.fc.append(nn.Sigmoid())
        self.fc.append(nn.Dropout(dropout))
        self.fc.append(nn.Linear(64, classes))
        self.fc.append(nn.Softplus())

    def forward(self, x):
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h

class TMC(nn.Module):
    def __init__(self, classes, views, classifier_dims, lambda_epochs=1,dropout=0.5):
        super(TMC, self).__init__()
        self.classes = classes
        self.views = views
        self.lambda_epochs = lambda_epochs
        self.dropout=dropout
        self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes, self.dropout) for i in range(self.views)])
        
    def DS_Combin(self, alpha):
        def DS_Combin_two(alpha1, alpha2):
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = self.classes / S[v]
                
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))
            S_a = self.classes / u_a
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a

    def forward(self, X, y, global_step):
        evidence = self.infer(X)
        loss = 0
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidence[v_num] + 1
            
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss_single(y, alpha_a, self.classes, global_step, self.lambda_epochs)
        loss = torch.mean(loss)
        return evidence, evidence_a, loss

    def infer(self, input):
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Classifiers[v_num](input[v_num])
        return evidence