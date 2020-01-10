from Expert import Expert
from Executor import Executor
from utils import get_ast, extract_table_aliases
from config import scan_types, join_types, relations
import numpy as np
import torch
from config import device
import pprint
from utils import construct_intermediate_tree, construct_final_tree, extract_join_info

class Query():
    def __init__(self, querytext, hinttext = '', ast=None):
        self.querytext = querytext
        self.hinttext = hinttext

        self.ast = get_ast(querytext) if ast is None else ast

        self.expert_plan = Expert().get_expert_plan(querytext, hinttext, timeout = 0)
        self.cost = self.expert_plan['Plan']['Total Cost']

        self.table_aliases = extract_table_aliases(self.ast)
        self.base_relations, self.join_pairs = extract_join_info(self.ast)


        #self.aux_list = [self.cost]






