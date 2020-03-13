import pandas as pd
import numpy as np

error = pd.read_csv('errors_per_sample_size.csv')
average_error = error.mean(axis = 0, skipna = True)
#print(average_error)

sample_sizes = list(range(50, 500, 50)) + [486]

probability = {}
true_hypothesis = {}

for sample in sample_sizes:
    col = error[str(sample)]
    good = 0
    hypothesis = 0
    for i in range(len(col)):
        if col[i] <= 0.05:
            good += 1
        if col[i] == 0.00:
            hypothesis += 1

    prob = good / len(col)
    hyp = hypothesis / len(col)
    
    if sample in probability:
        probability[sample].append(prob)
    else:
        probability[sample] = [prob]

    if sample in true_hypothesis:
        true_hypothesis[sample].append(hyp)
    else:
        true_hypothesis[sample] = [hyp]    

#print(pd.DataFrame.from_dict(probability)) 
#print(pd.DataFrame.from_dict(true_hypothesis)) 
