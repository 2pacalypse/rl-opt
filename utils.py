import moz_sql_parser
import numpy as np
from config import scan_types, join_types, binary_ops, unary_ops, ternary_ops
from config import PGHOST, PGDATABASE
from config import relations, columns
from config import device
import torch
import psycopg2
from treeutils import *
import pprint
import heapq
import random




"""
Given a Query objct, this function is utilized in plan searching
to initialize the children with their features, hint trees, and other information.
"""
def init_children(q):
    participation_vector = prep_participation_vector(q)
    predicate_vector = extract_column_predicates(q.ast, q.table_aliases)

    features = []
    hint_trees = []
    joined_attrs = []
    joined_root_vectors = []

    #consider each join in every possible way
    for jp in q.join_pairs:
        for jt in join_types:
            for st0 in scan_types:
                for st1 in scan_types:
                    v_r0 = create_tree_scan_vector(jp[0], st0, q.table_aliases)
                    v_r1 = create_tree_scan_vector(jp[1], st1, q.table_aliases)
                    v_join = create_tree_join_vector(jt, v_r0, v_r1)

                    feature1, feature2 = v_join + v_r0 + v_r1 + participation_vector + predicate_vector, v_join + v_r1 + v_r0 + participation_vector + predicate_vector
                    hint_v_r0, hint_v_r1 = create_node((st0, jp[0])), create_node((st1, jp[1]))
                    h1,h2 = create_node(jt, hint_v_r0, hint_v_r1), create_node(jt, hint_v_r1, hint_v_r0)

                    hint_trees += [h1, h2]
                    joined_attrs += [jp, jp]
                    joined_root_vectors += [v_join, v_join]
                    features += [feature1, feature2]
    return tuple(zip(features, hint_trees, joined_attrs, joined_root_vectors))


"""
This function is iteratively used in the main loop of plan searching
such that we give it a Query object, and the popped child from the previous
iteration, and we end up with the resulting children.
"""
def get_children(q, popped):
    participation_vector = prep_participation_vector(q)
    predicate_vector = extract_column_predicates(q.ast, q.table_aliases)

    feature, hint_tree, joined_attrs, joined_root_vector = popped[0], popped[1], popped[2], popped[3]

    features = []
    hint_trees = []
    joined_attrs_list = []
    joined_root_vectors = []

    #left deep and right deep here
    for br in q.base_relations:
        if br not in joined_attrs:
            for br2 in joined_attrs:
                if (br,br2) in q.join_pairs or (br2, br) in q.join_pairs:
                    for jt in join_types:
                        for st in scan_types:
                            v_st = create_tree_scan_vector(br, st, q.table_aliases)
                            v_join = create_tree_join_vector(jt, v_st, joined_root_vector)
                            hint_v_st = create_node((st, br))
                
                            f1, f2 = v_join + v_st + joined_root_vector + participation_vector + predicate_vector, v_join + joined_root_vector + v_st + participation_vector + predicate_vector
                            h1, h2 = create_node(jt, hint_v_st,  hint_tree), create_node(jt, hint_tree, hint_v_st)

                            hint_trees += [h1, h2]
                            features += [f1, f2]
                            joined_root_vectors += [v_join, v_join]
                            joined_attrs_list += [joined_attrs + (br,), joined_attrs + (br,)]

    #bushy here
    for jp in q.join_pairs:
        if jp[0] not in joined_attrs and jp[1] not in joined_attrs:
            for grandparent_jt in join_types:
                for parent_jt in join_types:
                    for st0 in scan_types:
                        for st1 in scan_types:
                            v_r0 = create_tree_scan_vector(jp[0], st0, q.table_aliases)
                            v_r1 = create_tree_scan_vector(jp[1], st1, q.table_aliases)
                            v_join = create_tree_join_vector(parent_jt, v_r0, v_r1)

                            hint_v_r0, hint_v_r1 = create_node((st0, jp[0])), create_node((st1, jp[1]))
                            sub_h1,sub_h2 = create_node(parent_jt, hint_v_r0, hint_v_r1), create_node(parent_jt, hint_v_r1, hint_v_r0)
                            grandparent_v_join = create_tree_join_vector(grandparent_jt, v_join, joined_root_vector)
                            
                            f1 = grandparent_v_join + v_join + joined_root_vector + participation_vector + predicate_vector
                            f2 = grandparent_v_join + joined_root_vector + v_join + participation_vector + predicate_vector

                            h1 = create_node(grandparent_jt, sub_h1, hint_tree)
                            h2 = create_node(grandparent_jt, hint_tree, sub_h1)
                            h3 = create_node(grandparent_jt, hint_tree, sub_h2)
                            h4 = create_node(grandparent_jt, sub_h2, hint_tree)

                            hint_trees += [h1, h2, h3, h4]
                            features += [f1, f2, f2, f1]
                            joined_root_vectors += [grandparent_v_join] * 4
                            joined_attrs_list += [joined_attrs + jp] * 4
    
    return tuple(zip(features, hint_trees, joined_attrs_list, joined_root_vectors))



"""
This function is searching for plans given a query and a network trained on
Q values, and outputs a hinttree.
"""

def hint(q, net):
    children = init_children(q)
    children_plans, children_hints, joined_attrs, joined_root_vectors = zip(*children)
    children_q_values = net(make_tensor(children_plans)).data.cpu().numpy()

    min_idx = np.argmin(children_q_values)
    best_child = children[min_idx]

    while True:
        if len(best_child[2]) == len(q.base_relations):
           return best_child[1]
        children = get_children(q, best_child)
        children_plans, children_hints, joined_attrs, joined_root_vectors = zip(*children)
        children_q_values = net(make_tensor(children_plans)).data.cpu().numpy()
        min_idx = np.argmin(children_q_values)
        best_child = children[min_idx]



"""
Given the AST of the query, this function returns the relations involved in the query
and the join clauses only with their table names.
"""
def extract_join_info(ast):
    joins = []
    find_joins(ast['where'], joins)
    join_pairs = tuple(map(lambda l: tuple([x.split('.')[0] for x in l]), joins))
    base_relations = tuple(set([x.split('.')[0] for join_tuple in joins for x in join_tuple]))    
    return base_relations, join_pairs


"""
Creates a zero vector of size R*ST + JT
"""
def create_zero_vector():
    return tuple([0] * (len(relations) * len(scan_types) + len(join_types)))



"""
Used in training
"""
def featurize_dict(d):
    fs = []
    vs = []
    for f,v in d.items():
        fs.append(f), vs.append((np.log(v),))
    return tuple(fs), tuple(vs)

"""
Used in training
"""
def prepare_min_cost_dict(qs, d = {}):
    pairs = [(f, q.cost) for q in qs for f in featurize_query(q)]
    for f,c in pairs:
        v = d.get(f, c)
        d[f] = min(v,c)
    return d

"""
Featurizes a set of Query objects
"""
def featurize_queries(qs):
    pairs = [(f, q.cost) for q in qs for f in featurize_query(q)]
    d = {}
    for f,c in pairs:
        v = d.get(f, c)
        d[f] = min(v,c)

    fs = []
    vs = []
    for f,v in d.items():
        fs.append(f), vs.append((np.log(v),))

    return tuple(fs), tuple(vs)



"""
Participation vector from the paper
"""
def prep_participation_vector(q):
    v = [0] * len(relations)
    for br in q.base_relations:
        idx = get_table_id(br, q.table_aliases)
        v[idx] = 1
    return tuple(v)


"""
Featurize a query according to the description in the paper.
"""
def featurize_query(q):
    participation_vector = prep_participation_vector(q)
    predicate_vector = extract_column_predicates(q.ast, q.table_aliases)

    readable_plan_tree = construct_intermediate_tree(q.expert_plan['Plan'])
    encoded_plan_tree = construct_final_tree(readable_plan_tree, q.table_aliases)

    features= []
    stack = [encoded_plan_tree]
    while stack:
        n = stack.pop()
        left_vector =  create_zero_vector() if is_leaf(n) else get_datum(left(n))
        right_vector = create_zero_vector() if is_leaf(n) else get_datum(right(n))
        current_vector = get_datum(n)

        feature = current_vector + left_vector + right_vector + participation_vector + predicate_vector
        features.append(feature)
        if not is_leaf(n):
            stack.append(left(n))
            stack.append(right(n))

    return tuple(features)


"""
This takes the intermediate tree generated by construct_intermediate_tree
along with the table aliases belonging to queries.
"""
def construct_final_tree(readable_plan, table_aliases):
    if is_leaf(readable_plan):
        scan_type, scanned_table = get_datum(readable_plan)
        vector = create_tree_scan_vector(scanned_table, scan_type, table_aliases)
        return create_node(vector)
    else:
        join_type = get_datum(readable_plan)
        left_tree, right_tree = construct_final_tree(left(readable_plan), table_aliases), construct_final_tree(right(readable_plan), table_aliases)
        vector = create_tree_join_vector(join_type, get_datum(left_tree), get_datum(right_tree))
        return create_node(vector, left_tree, right_tree)


"""
Given an SQL query
return its abstract syntax tree
"""
def get_ast(query_text):
    return moz_sql_parser.parse(query_text)


"""
Take the upper triangle part of the (with offset 1) symmetric join graph,
concat with the predicates vector.
"""
def extract_query_level_vector(ast, table_aliases):
    join_graph = extract_join_graph(ast, table_aliases)
    column_predicates = extract_column_predicates(ast, table_aliases)

    iu1 = np.triu_indices(len(relations), 1)
    return tuple(np.concatenate((join_graph[iu1], column_predicates)))


"""
From the expert plan, construct the tree structured join and scan information.
Tree is of the form [data, [leftChild], [rightChild]]
data is Node Type or (Node Type, Relation Name)

Used internally for plan encoding
"""
def construct_intermediate_tree(node):
    if node['Node Type'] in scan_types: #leaf
        scan_type, relation_name = node['Node Type'], node['Relation Name']
        leaf = create_node((scan_type, relation_name))
        return leaf
    elif node['Node Type'] in join_types:
        left_tree = construct_intermediate_tree(node['Plans'][0])
        right_tree = construct_intermediate_tree(node['Plans'][1])
        treenode = create_node(node['Node Type'], left_tree, right_tree)
        return treenode
    else:
        return construct_intermediate_tree(node['Plans'][0])



"""
a scan vector needs the name of table and the type of scan along with the
table aliases belonging to the query
"""
def create_tree_scan_vector(scanned_table, scan_type, table_aliases):
    vector = [0] * ( len(join_types) + len(scan_types) * len(relations) )
    table_index = get_table_id(scanned_table, table_aliases)
    idx = len(join_types) + table_index * len(scan_types) + scan_types.index(scan_type)
    vector[idx] = 1
    return tuple(vector)


"""
a join vector is the union of its children plus the join type prepended.
"""
def create_tree_join_vector(join_type, left_tree_vector, right_tree_vector):
    vector = [0] * ( len(join_types) + len(scan_types) * len(relations) )
    vector[len(join_types):] = [ x or y for (x,y) in zip( left_tree_vector[len(join_types):], right_tree_vector[len(join_types):] )]
    join_idx = join_types.index(join_type)
    vector[join_idx] = 1
    return tuple(vector)



"""
Return a dict with (alias_name, table_name) as k,v for the query given its ast.
"""
def extract_table_aliases(ast):
    aliases = dict()
    for v in ast['from']:
        if(isinstance(v, dict)):
            aliases[v['name']]  = v['value']
    return aliases


"""
Given an alias or table name from the query, this returns its index from the self.relations.

Used internally for state encoding.
"""

def get_table_id(table_name, table_aliases):
    if(table_aliases.get(table_name) is None):
        return relations.index(table_name)
    return relations.index(table_aliases[table_name])



"""
Matrix of size len(relations)*len(relations)
graph[i][j] means t_i and t_j are joined.
This is a symmetric matrix.
"""

def extract_join_graph(ast, table_aliases):
    joins = []
    find_joins(ast['where'], joins)


    graph = np.zeros((len(relations), len(relations)))
    for j in joins:
            table_left = j[0].split(".")[0]
            table_right = j[1].split(".")[0]
            table_left_index = get_table_id(table_left, table_aliases)
            table_right_index = get_table_id(table_right, table_aliases)
            graph[table_left_index, table_right_index] = 1
            graph[table_right_index, table_left_index] = 1
    return graph


"""
Populates the given list with join cluases.
join_list[out] = [["t1.attr1", "t2.attr2"],  ...]
"""
def find_joins(root, join_list):
    if(isinstance(root, dict) and root.get('literal') is None):
        for k in root:
            if(k == 'eq' and isinstance(root[k][0], str) and isinstance(root[k][1], str)):
                join_list.append(root[k])
            for child in root[k]:
                find_joins(child, join_list)



"""
Given an alias or tablename, and also a column name from that table, this returns the
attribute's index according to the order in self.relations, and the order in the values of
self.attributes.

Used internally for state encoding.
"""
def get_pred_id(table_name, attr_name, table_aliases):
    table_id = get_table_id(table_name, table_aliases)

    idx = 0
    for i in range(table_id):
        idx += len(columns[relations[i]])

    attr_list = columns[relations[table_id]]
    for attr in attr_list:
        if(attr == attr_name):
            break
        idx += 1

    return idx


"""
Vector of column predicates.
"""
def extract_column_predicates(ast, table_aliases):
    predicates = []
    find_predicates(ast['where'], predicates)
    # pprint.pprint(predicates)

    size = sum(map(len, columns.values()))
    state = [0] * size

    for d in predicates:
        for v in d.values():
            v = [v] if isinstance(v, str) else v
            for operand in v:
                if isinstance(operand, str):
                    table, attr = operand.split(".")
                    idx = get_pred_id(table, attr, table_aliases)
                    state[idx] = 1
    return tuple(state)


"""
Returns a list of dictionaries with k,v where k is op and v is either a str
or a list depending on whether op is unary or binary/ternary.

Used internally for column predicates
"""
def find_predicates(root, predicate_list):
    if(isinstance(root, dict) and root.get('literal') is None):
        for k in root:
            if k in binary_ops and ( isinstance(root[k][0], str) ^ isinstance(root[k][1], str) ):
                predicate_list.append(root)
            elif k in unary_ops | ternary_ops  and isinstance(root[k][0], str):
                predicate_list.append(root)
            for child in root[k]:
                find_predicates(child, predicate_list)



"""
returns a connection object to the database
"""
def connect():
    try:
        conn_string = (
        "host="
        + PGHOST
        + " port="
        + "5432"
        + " dbname="
        + PGDATABASE
#        + " options='-c statement_timeout=60000'"
#                               + " user="
#                               + creds.PGUSER
#                               + " password="
#                               + creds.PGPASSWORD
        )
        conn = psycopg2.connect(conn_string)
        return conn
    except (Exception, psycopg2.Error) as error:
        print("Error connecting", error)


"""
split used in training
"""
def split_train_test(queries, test_percent = 0.2):
    np.random.seed(8)
    len_all_data = len(queries)
    k = int(len_all_data*(1- test_percent))
    np.random.shuffle(queries)
 
    train_queries, test_queries = queries[:k], queries[k:]

    return train_queries, test_queries





"""
takes the hint() method's output as the input.
this returns the hints ready for pg_hint_plan
"""
def get_hints(plan):
    def formatter(text):
        text = text.replace(' ', '')
        text = text.replace('BitmapHeapScan', 'BitmapScan')
        text = text.replace('NestedLoop', 'NestLoop')
        return text


    hints = []
    def scan_hints(plan):
        nonlocal  hints
        if plan[1] == None and plan[2] == None:
            scantype, table = plan[0]
            hints.append(formatter(scantype) + '(' + table + ')')
            return
        scan_hints(plan[1])
        scan_hints(plan[2])


    scan_hints(plan)


    def leading_hint(plan):
        if plan[1] == None and plan[2] == None:
            return plan[0][1]
        else:
            leftop, rightop = leading_hint(plan[1]), leading_hint(plan[2])
            currenthint  = '(' + leftop + ' ' + rightop + ')'
            return currenthint

    hints.append('Leading('+ leading_hint(plan) + ')')




    def join_hints(plan):
        nonlocal hints
        if plan[1] == None and plan[2] == None:
            scantype, table = plan[0]
            return table
        else:
             leftop = join_hints(plan[1])
             rightop = join_hints(plan[2])
             currenthint = formatter(plan[0]) + '(' + rightop + ' ' + leftop + ')'
             hints.append(currenthint)
             return (leftop + ' ' + rightop)

    join_hints(plan)

    return hints


"""
given the output of get_hints
this prepares the hinttext so that
it may be substituted in as a query comment
"""
def prep_hinttext(hintlist):
    hinttext = '/*+\n'
    for hint in hintlist:
        hinttext += hint + '\n'
    hinttext += '*/\n'

    return hinttext




"""
create a tensor on the device
"""
def make_tensor(x):
    return torch.Tensor(x).to(device)





"""
used to create minibatches for the training
"""
def iterate_minibatches(fvs, cs, batchsize):
    num_elements = len(fvs)
    if batchsize > num_elements:
        batchsize = num_elements
    for start_idx in range(0, num_elements - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield fvs[excerpt], cs[excerpt] 




"""
returns the number of join clauses in the ast
"""
def get_n_joins(ast):
    joins = []
    #this is a list of tuples (joins)
    find_joins(ast['where'], joins)
    #joins = list(map(lambda l: [x.split('.')[0] for x in l], joins))
    return len(joins)




