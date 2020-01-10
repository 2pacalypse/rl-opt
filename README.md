# rl-opt

This is the code for the semester project I did at DIAS/EPFL. It is an implementation
of a deep reinforcement learning based query optimizer. The background and the general design
may be found in the report.

## Guides

- queries2pkl.py:
     assumes you have the join-order-benchmark in the current directory.
this file is responsible for generating a list of Query objects for 113 JOB queries.
The output generated by this file is job_queries.pkl

- split_job_qs.py:
     assumes you have the job_queries.pkl in the current directory.
The output is the split 2 files named job_train_qs.pkl and job_test_qs.pkl.

- traininf.py
    usage: traininf.py [-h]
                   eval_queries train_queries exp_output_dir model_output_dir
                   exp_output_freq model_output_freq

    this is the training of the deep network. for an infinite number of episodes, at each episode,
    we train 10 epochs based on train_queries and then we evaluate the performance on eval_queries
    and retrain from the scratch in the next episode with the feedback for the eval_queries.
    the last two numbers are the frequencies to output the experience and the model.
    the experience is nothing but a pickle file containing a list of Query objects.

    if you run it for the first time, it makes sense to start like this
        example run: `python3 traininf.py job_train_qs.pkl job_train_qs.pkl expout modelout 20 10`
    
    once you have the generated experiences under exp_output_dir then you can continue with them by
        `python3 traininf.py job_train_qs.pkl exp_output_dir/540.pkl expout modelout 20 10`

    so for example, the second example will train a model from 540 queries, and then when all the epochs are over
    it will evaluate the trained net's performance on the job_train_qs which is of size 90, and the train_queries
    will be updated with these fresh 90 experience, so it will continue with 630 queries, 700, and so on. Depending on
    the output frequency, it will output these cumulative experiences.

- testhint.py
 usage: testhint.py [-h] evaluated_queries model_inp_path

    it takes a pickle file containing queries and also a model and it outputs the costs of the queries and the resulting costs
    from the model.    

    example run: `python3 testhint.py job_train_qs.pkl 540.pt`
             `python3 testhint.py job_test_qs.pkl 9000.pt`

- config.py: contains the device (gpu/cpu) for Pytorch, and also database related credentials.

- Net.py: A Pytorch neural network.

-  Executor.py: A wrapper for useful operations like retrieving the table&column names, latencies, etc.

-  Expert.py: A class with the method get_expert_plan so that the plan can be used afterwards.

-  Query.py: A wrapper for the queries, it caches some of the query related information.

- utils.py: most of the work is here including the plan search, parsing, featurization.

- treeutils.py: we use python tuples to represent trees, there are a few helpers.
