import os
import moz_sql_parser
import pprint
import json
from Query import Query
import pickle
from utils import featurize_query
query_set = "./join-order-benchmark" #query dir

cwd = os.getcwd()
querydir = os.path.join(cwd, query_set)
files = os.listdir(querydir)
files = list(filter(lambda f: f[0].isdigit(), files))



queries = []

i = 0
for file_name in files:
    file_path =  os.path.join(querydir, file_name)
    f = open(file_path, "r")
    querytext = f.read() #sql query
    q = Query(querytext)
    print('q' + str(i))
    queries.append(q)
    i += 1


with open('job_queries.pkl', 'wb') as f:
    pickle.dump(queries, f)
