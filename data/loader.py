"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant
from torch.autograd import Variable

from torch._six import string_classes, int_classes, FileNotFoundError
import collections

from prefetch_generator import BackgroundGenerator
from model.tree import head_to_adj

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False, pin_memory=False):
        self.batch_size = batch_size # 50
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation # false or true
        self.label2id = constant.LABEL_TO_ID # {dict:42}{'no_relation':0,...}
        self.pin_memory = pin_memory

        with open(filename) as infile:
            data = json.load(infile)

        self.raw_data = data # {list:68124}-[{dict:17}-{'id':'','head':''}...]
        data = self.preprocess(data, vocab, opt)
        # data = list:68124, each list is a 10+tuple,

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()]) # {0:'no_relation',... *42}
        self.labels = [self.id2label[d[9]] for d in data] # 关系 label  ['per:title',... *68124]
        self.num_examples = len(data) # 68124

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        # self.data = data # [[(tokens,ner,pos,...*12)*50]*1363]
        self.data = [self.__getitem__(data[i]) for i in range(len(data))]

        print("{} batches created for {}".format(len(data), filename))
        del data

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data: # d:{dict:17}
            tokens = list(d['token']) # {list:tokens_len}
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1) # 实体的每个 token 被换成 type mask
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)
            tokens = map_to_ids(tokens, vocab.word2id) # {list_of_int:tokens_len}-tokens 换成了 id
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            # deprel = [0 for i in range(len(pos))]
            head = [int(x) for x in d['stanford_head']] # {list_of_int:tokens_len}
            head_berkeley = [int(x) for x in d['berkeley_head']]
            head_sequence = [x for x in range(len(head))]
            # deprel_berkeley = map_to_ids(d['berkeley_deprel'], constant.DEPREL_TO_ID)
            assert any([x == 0 for x in head])
            assert any([x == 0 for x in head_berkeley])
            l = len(tokens) # 当前tokens 的长度
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l) # 所有 token 以主语位置为原点的相对位置 {list_of_int:tokens_len}
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l) # 所有 token 以宾语位置为原点的相对位置
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]] # int
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            relation = self.label2id[d['relation']]
            processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type,
                           relation, head_berkeley, head_sequence)]
        return processed # {list_of_{tuple:12}:len(data)}

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, in_batch):
        """ Get a batch with index. """
        # if not isinstance(key, int):
        #     raise TypeError
        # if key < 0 or key >= len(self.data):
        #     raise IndexError
        # batch = data[key] # [(1,...,12)*50]
        batch = in_batch
        batch_size = len(batch) # 50
        batch = list(zip(*batch)) # zip 接受一系列可迭代的对象作为参数，将对象中对应的元素打包成一个个tuple（元组）
        # [[token_list]*50, [pos_list]*50,... *12]
        assert len(batch) == 10+2  # 相当于行列颠倒了

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]] # 50个句子各自的长度(排序前)
        batch, orig_idx = sort_all(batch, lens) # 按长度从大到小排序，还没<pad>

        # word dropout
        if not self.eval: # todo 为啥word也要dropout
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size) # train模式下是dropout之后的句子 50x64
        masks = torch.eq(words, 0)  # 比较元素相等性 50x64
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)
        head_berkeley = get_long_tensor(batch[10], batch_size)
        # deprel_berkeley = get_long_tensor(batch[11], batch_size)
        head_sequence = get_long_tensor(batch[11], batch_size)
        orig_idx = torch.IntTensor(orig_idx)

        rels = torch.LongTensor(batch[9]) # shape=50

        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)  # l 是各个句子的长度 [64,...,14 *50]
        l = torch.IntTensor(l)
        # def inputs_to_adj_reps(head, l):
        #     adj = [head_to_adj(head[i], l[i], maxlen, directed=False)for i in range(len(l))]
        #     adj = [adj[i].reshape(1, maxlen, maxlen) for i in range(len(l))]
        #     adj = np.concatenate(adj, axis=0)  # shape=(maxlen * adj_num, adj)
        #     adj = torch.from_numpy(adj)
        #     return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions,
                subj_type, obj_type, rels, orig_idx, head_berkeley, head_sequence, l)

    def __iter__(self):
        for i in range(self.__len__()):
            # yield self.__getitem__(i)
            batch = self.data[i]
            if self.pin_memory:
                batch = pin_memory_batch(batch)
            yield tuple(batch)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

class DataLoaderX(DataLoader):
    def __iter__(self):
        return  BackgroundGenerator(super().__iter__())

def pin_memory_batch(batch):
    if isinstance(batch, torch.Tensor):
        return batch.pin_memory()
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: pin_memory_batch(sample) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [pin_memory_batch(sample) for sample in batch]
    else:
        return batch

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = [Variable(b.cuda(non_blocking=True)) for b in self.batch[:]]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch