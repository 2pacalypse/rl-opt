import pickle
from utils import split_train_test

with open('job_queries.pkl', 'rb') as f:
    qs = pickle.load(f)
    train_qs, test_qs = split_train_test(qs)


with open('job_train_qs.pkl', 'wb') as f:
    pickle.dump(train_qs, f)

with open('job_test_qs.pkl', 'wb') as f:
    pickle.dump(test_qs, f)


