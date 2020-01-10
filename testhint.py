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



with open('job_queries.pkl', 'rb') as f:
    job_queries = pickle.load(f)

train_qs, test_qs = split_train_test(job_queries)



#95670 is the one in the paper
net = torch.load('models_per_episode/95670.pt')


q_hs = []
for q in train_qs:
    hinttree = hint(q, net)
    hints = get_hints(hinttree)
    hinttext = prep_hinttext(hints)
    q_h = Query(q.querytext, hinttext, q.ast)
    q_hs.append(q_h)

    print(q.cost, q_h.cost)







