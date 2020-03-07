import pandas as pd
import numpy as np

error = pd.read_csv('errors_per_sample_size.csv')
average_error = error.mean(axis = 0, skipna = True)
print(average_error)