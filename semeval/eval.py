"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch

from data.loader import DataLoader,DataPrefetcher
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

import json

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='checkpoint_epoch_150.pt', help='Name of the model file.')
parser.add_argument('--config', type=str, default='/config.json')

parser.add_argument('--data_dir', type=str, default='dataset/semeval')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(0)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
# opt = torch_utils.load_config(model_file)
with open(args.model_dir + args.config,'r') as fconfig:
    opt = json.load(fconfig) #todo
trainer = GCNTrainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir']  + '/{}_std_ber.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
test_batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
all_probs = []
# batch_iter = tqdm(batch)
# for i, b in enumerate(batch_iter):
#     preds, probs, _ = trainer.predict(b)
#     predictions += preds
#     all_probs += probs
print('Start testing...')
prefetcher = DataPrefetcher(test_batch)
batch = prefetcher.next()
while batch is not None:
    preds, probs, _ = trainer.predict(batch)
    predictions += preds
    all_probs += probs
    batch = prefetcher.next()

predictions = [id2label[p] for p in predictions]
print(predictions)
p, r, f1 = scorer.score(test_batch.gold(), predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

print("Evaluation ended.")

