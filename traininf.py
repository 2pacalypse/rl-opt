from utils import split_train_test, featurize_queries, make_tensor
import pickle
import numpy as np
from Net import Net
import torch.nn as nn
import torch.optim as optim
import itertools
import torch
import numpy as np
from utils import iterate_minibatches, hint, get_hints, prep_hinttext, prepare_min_cost_dict, featurize_dict
import argparse
import os
from Query import Query

parser = argparse.ArgumentParser(description='Training of the queries')
parser.add_argument('eval_queries', type = str)
parser.add_argument('train_queries')
parser.add_argument('exp_output_dir', type= str)
parser.add_argument('model_output_dir', type = str)
parser.add_argument('exp_output_freq', type = int)
parser.add_argument('model_output_freq', type = int)

def train(train_qs, eval_qs):
    d = prepare_min_cost_dict(train_qs)
    fs, cs = featurize_dict(d)
    # an episode
    for e in itertools.count(start = 1):
        print("\ntotal size of the batch, total size of the queries, episode number: ", len(fs), len(train_qs), e)

        net = Net()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr = 0.001)

        last_epoch_loss = 0
        #epoch
        for i in range(10):
            epoch_loss = 0
            for minibatch_fv, minibatch_cs in iterate_minibatches(fs,cs,8):
                x, y = make_tensor(minibatch_fv), make_tensor(minibatch_cs)
                optimizer.zero_grad()
                loss = criterion(net(x), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print("loss, ", epoch_loss)


        hinted_qs = evaluate(eval_qs, net)
        train_qs += hinted_qs
        if e% args.exp_output_freq == 0:
            print("saving experience")
            exp_name = str(len(train_qs)) + '.pkl'
            with open(os.path.join(args.exp_output_dir, exp_name), 'wb') as f:
                pickle.dump(train_qs, f)

        d = prepare_min_cost_dict(hinted_qs, d)
        fs, cs = featurize_dict(d)

        if e % args.model_output_freq == 0:
            m_name = str(len(train_qs)) + '.pt'
            torch.save(net, os.path.join(args.model_output_dir, m_name ))
            print("saved model with name", str(len(train_qs)))




def evaluate(train_qs, net):
    hinted_qs = []
    for q in train_qs:
        hinttree = hint(q, net)
        hintlist = get_hints(hinttree)
        hinttext = prep_hinttext(hintlist)
        q_h = Query(q.querytext, hinttext, q.ast)
        print(q.cost, q_h.cost)
        hinted_qs.append(q_h)
    print('total log cost real, total log cost trained', sum([np.log(q.cost) for q in train_qs]), sum([np.log( q.cost) for q in hinted_qs]))

    return hinted_qs



args = parser.parse_args()

with open(args.eval_queries, 'rb') as f:
    eval_queries = pickle.load(f)

with open (args.train_queries, 'rb') as f:
    train_queries = pickle.load(f)


if args.model_output_dir and not os.path.exists(args.model_output_dir):
    os.makedirs(args.model_output_dir)

if args.exp_output_dir and not os.path.exists(args.exp_output_dir):
    os.makedirs(args.exp_output_dir)

train(train_queries, eval_queries)
