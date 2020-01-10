import pickle
import torch
from utils import split_train_test
from Query import Query
import pprint
from time import sleep
from utils import hint, featurize_queries, make_tensor, get_hints, prep_hinttext
from utils import construct_intermediate_tree
import numpy as np
from utils import find_predicates
from utils import get_n_joins
from Executor import Executor
import argparse

parser = argparse.ArgumentParser(description='Testing of the queries')
parser.add_argument('evaluated_queries', type=str)
parser.add_argument('model_inp_path', type = str)
args = parser.parse_args()


with open(args.evaluated_queries, 'rb') as f:
    job_queries = pickle.load(f)




#95670 is the one in the paper
net = torch.load(args.model_inp_path)


q_hs = []
for q in job_queries:
    hinttree = hint(q, net)
    hints = get_hints(hinttree)
    hinttext = prep_hinttext(hints)
    q_h = Query(q.querytext, hinttext, q.ast)
    q_hs.append(q_h)

    print("real & hinted", q.cost, q_h.cost)








