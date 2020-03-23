import pandas as pd
import numpy as np

error = pd.read_csv('errors.csv')

sample_sizes = list(range(50, 500, 50)) + [486]

probability_error = {}
true_hypothesis = {}

for sample in sample_sizes:
    col = error[str(sample)]
    no_error = 0
    hypothesis = 0

    for i in range(len(col)):
        if col[i] <= 0.05:
            no_error += 1
        if col[i] == 0.00:
            hypothesis += 1

    prob = no_error / len(col)
    hyp = hypothesis / len(col)
    
    if sample in probability_error:
        probability_error[sample].append(prob)
    else:
        probability_error[sample] = [prob]

    if sample in true_hypothesis:
        true_hypothesis[sample].append(hyp)
    else:
        true_hypothesis[sample] = [hyp]    

print('\n')
print('First column')
print('\n')
print(pd.DataFrame.from_dict(probability_error)) 
print('\n')
print('Second column')
print('\n')
print(pd.DataFrame.from_dict(true_hypothesis)) 
print('\n')
print('Third column')
print('\n')
average_error = error.mean(axis = 0, skipna = True)
print(average_error)
