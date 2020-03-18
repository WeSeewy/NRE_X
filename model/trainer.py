"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.aggcn import GCNClassifier
from utils import torch_utils


class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    if cuda: # batch：tuple12*50Tensor
        inputs = [Variable(b.cuda()) for b in batch[:10]] # (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type)
        labels = Variable(batch[10].cuda()) # rels
    else:
        inputs = [Variable(b) for b in batch[:10]]
        labels = Variable(batch[10])
    tokens = batch[0]  # 50 * 64
    head = batch[5] # 50 * 64
    subj_pos = batch[6]   # 50 * 64
    obj_pos = batch[7] # 50 * 64
    lens = batch[1].eq(0).long().sum(1).squeeze() # 每个句子的长度 Shape=50
    '''
    subj_pos:tensor([[-14, -13, -12,  ...,  46,  47,  48],
        [-50, -49, -48,  ...,   0,   0,   0],
        [-31, -30, -29,  ...,   0,   0,   0],
        ...,
        [-11, -10,  -9,  ...,   0,   0,   0],
        [-13, -12, -11,  ...,   0,   0,   0],
        [ -9,  -8,  -7,  ...,   0,   0,   0]])
        
    lens:tensor([64, 58, 56, 51, 50, 47, 46, 46, 45, 45, 44, 43, 43, 43, 43, 43, 42, 41,
        41, 38, 37, 36, 36, 35, 33, 33, 33, 32, 31, 31, 31, 31, 30, 30, 29, 28,
        28, 28, 27, 27, 26, 24, 23, 22, 22, 21, 21, 20, 16, 14])
    '''
    return inputs, labels, tokens, head, subj_pos, obj_pos, lens

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, pooling_output = self.model(inputs)
        loss = self.criterion(logits, labels)
        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.opt['conv_l2']
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            # L2-penalty for all pooling output.
            # 各个参数的平方值的和的开方值
            loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm']) # 梯度裁剪
        self.optimizer.step() # 更新参数
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[11]
        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.item()
