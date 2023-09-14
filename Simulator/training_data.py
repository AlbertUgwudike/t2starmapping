import torch
import numpy as np
import pandas as pd
from time import time
from functools import reduce

from .numerical_simulator import numerical_simulator
from utility import pt_irange, hcat, identity, col_vectors, fst, snd


'''
generate_data - uses simulator to generate large quatities of data

parameters: 
    ranges_dict:    dict { <simulator parameter>: <range tuple> } e.g. { "R2_star" : (100, 150) }
    n_t:            number of timpoints in the range (0.005, 0.040) to sample
    batch_size:     number of samples to simulate per batch
    n_batches:      number of batches (n_samples == batch_size * n_batches == a ^ len(ranges_dict))
    transform:      function to apply to df before returning it. 

return:     df of simulated data transformed using the transform parameter. If transform == identity, 
            returns original df with headers == ranges_dict.keys() + ["r1", ... "r<n_t>"] + ["i1", ... "i<n_t>"]

example: generate_data({ "R2_star" : (100, 150), "B0_x" : (50, 100), "B0_y" : (50, 100) }, 2, 8, 1) = ...
        
           [[ "R2_star", "B0_x", "B0_y", "r1", "r2", "i1", "i2" ],
            [  100,       50,     50,     --,   --,   --,   --  ],
            [  100,       50,     100,    --,   --,   --,   --  ],
            [  100,       100,    50,     --,   --,   --,   --  ]
            ....                    ....                      ....
            [  150,       100,    100,    --,   --,   --,   --  ]]
'''
def generate_data(ranges_dict, n_t, batch_size, n_batches, transform=identity):

    n_samples = batch_size * n_batches 
    t = pt_irange(0.000, 0.04, n_t)  

    combos = parameter_combos(ranges_dict.values(), n_samples)
    batches = torch.split(combos, batch_size)

    # associate each param vec with argument_name, add t to dict here
    keys = list(ranges_dict.keys()) + ["t"]
    f = lambda batch: dict(zip(keys, col_vectors(batch) + [t]))
    batches = list(map(f, batches))

    print(f"Generating {n_batches} batches of {batch_size} simulated signals.")

    start = time()
    out_batches = list(map(simulate_batch, batches))
    out = np.concatenate(out_batches)
    elapsed = round((time() - start) * 1000)

    print(f"Training examples generated: {out.shape[0]}")
    print(f"Elapsed Time: {elapsed}ms")

    col_names = lambda c: [f"{c}{i + 1}" for i in range(n_t)]
    columns = list(ranges_dict.keys()) + col_names("r") + col_names("i")
    return transform(pd.DataFrame(hcat(combos, out), columns=columns))


'''
cross_join - compute the cartesian product (every combination) of the input vectors

parameters: 'list_of_tensors' = list of 1D tensors to be cross joined

return: 2D tensor, result of cartesian product

example: cross_join([[1, 2], [3, 4]]) = [[1, 3], [1, 4], [2, 3], [2, 4]]
'''
def cross_join(list_of_tensors):
    list_of_dfs = map(lambda t: pd.DataFrame(t.numpy(), columns=[0]), list_of_tensors)
    join = lambda acc, a: acc.merge(a, how="cross", suffixes=[None, len(acc)]) 
    return torch.tensor(reduce(join, list_of_dfs).values)


'''
parameter_combos - generate an even spread of parameter combos from the given ranges

parameters: 
    ranges:     list of tuples defining the range of each arbitrary parameter. 
                e.g. [(1, 2), (11, 110)] 
    n_samples:  the total number of combinations to produce. Note: to operate
                as intended, n_samples should be a ^ len(ranges), e.g. if 
                len(ranges) == 2, n_samples could be 4, 9, 16 ...

return:  2D (len(ranges) X n_samples) tensor of combinations

example: parameter_combos([(1, 3), (4, 6)], 9) = ...
         [ [1, 4], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6] ]
'''
def parameter_combos(ranges, n_samples):

    # pair each param range with sample count
    lower_bounds = map(fst, ranges)
    upper_bounds = map(snd, ranges)
    counts = [n_samples ** (1 / len(ranges))] * len(ranges)
    args_vec = zip(lower_bounds, upper_bounds, counts)

    # generate range of vals for each parameter
    param_vals = list(map(lambda args: pt_irange(*args), args_vec))

    # generate every combo (outer product)
    return cross_join(param_vals)

'''
generate_mlp_data - Creates training data for MLP model

parameters:
    filename:   file to write generated data to.
'''
def generate_mlp_data(filename):   
    transform = lambda raw_df: mlp_transform(raw_df, 3)

    ranges_dict_1 = {  "R2_star": (0, 200), "B0_x": (0, 150), "B0_y": (0, 150) }
    df1 = generate_data(ranges_dict_1, 18, 225, 15, transform)

    ranges_dict_2 = {  "R2_star": (201, 500), "B0_x": (0, 150), "B0_y": (0, 150) }
    df2 = generate_data(ranges_dict_2, 18, 100, 10, transform)

    ranges_dict_3 = {  "R2_star": (501, 800), "B0_x": (0, 150), "B0_y": (0, 150) }
    df3 = generate_data(ranges_dict_3, 18, 100, 10, transform)

    pd.concat([df1, df2, df3]).to_csv(filename)

'''
simulate_batch - perform simulation using each of a batch of function parameters

parameters:
    batch_dict:    dict { <parameter>: <values> }, e.g { R2_star: [50, 60], B0_x: [20, 35] }

return: 2D array (batch_size X n_t) of simulation results
'''
def simulate_batch(batch_dict):
    print("Executing Batch")
    sig = numerical_simulator(**batch_dict, n_isochromats=100_000)
    return torch.cat((sig["real"], sig["imag"]), 1)


'''
mlp_transform - convert raw simulation output df to MLP format

parameters:
    raw_df:     raw output from simulation using a range of params
    n_params:   number of parameters used (permuted) in simulation

return:   df of data in new format

example : ["R2_star", "B0", "t1", "t2", "t3"]     ["R2_star", "B0", "t",    "signal"]
          [ 50,        100,  0.9,  0.5,  0.3] --> [ 50,        100,  0.005,  0.9    ]
          [ 60,        120,  0.8,  0.2,  0.1]     [ 50,        100,  0.023,  0.5    ]
                                                  [ 50,        100,  0.040,  0.3    ]
                                                  [ 60,        120,  0.005,  0.8    ]
                                                  [ 60,        120,  0.023,  0.2    ]
                                                  [ 60,        120,  0.040,  0.1    ]
'''
def mlp_transform(raw_df, n_params):
    raw_pt = torch.tensor(raw_df.values)
    n_t = (raw_pt.shape[1] - n_params) // 2
    real = raw_pt[:, n_params:n_params + n_t].flatten().unsqueeze(1)
    imag = raw_pt[:, n_params + n_t:n_params + 2*n_t].flatten().unsqueeze(1)
    t = pt_irange(0.00, 0.04, n_t).unsqueeze(1)
    t_repeat = torch.tile(t, (raw_pt.shape[0], 1))
    param_repeat = raw_pt[:, :n_params].repeat_interleave(n_t, dim=0)
    cat = torch.cat((param_repeat, t_repeat, real, imag), 1)
    columns = list(raw_df.columns)[:n_params] + ["t", "real", "imag"]
    return pd.DataFrame(cat, columns=columns)





