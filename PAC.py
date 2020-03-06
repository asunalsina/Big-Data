import numpy as np
from random import sample
import pandas as pd


TRUE_FUNCTION = [(1, 1), (3, 0), (5, 1), (7, 1)]


def classify_example(example, hypothesis):
    return all([example[index] == label for index, label in hypothesis])


def classify(features, hypothesis):
    return np.array([classify_example(example, hypothesis) for example in features])


def learning(data):
    positive_examples = data[data[:, -1] == 1]
    hypothesis = []
    for i in range(positive_examples.shape[-1] - 1):
        values = np.unique(positive_examples[:, i])
        if len(values) == 1:
            hypothesis.append((i, values[0]))
    return hypothesis


if __name__ == "__main__":
    np.random.seed(23)
    N = 20

    FEATURES = np.random.choice(2, (10000, 20))
    TRUE_LABELS = classify(FEATURES, TRUE_FUNCTION).reshape((-1, 1))
    TRUE_DATA = np.hstack((FEATURES, TRUE_LABELS))


    sample_sizes = list(range(0, 500, 50)) + [486]

    for i in range(10000):
        errors_per_m = pd.DataFrame(columns=sample_sizes)
        for m in sample_sizes:
            current_sample = sample(FEATURES, m)
            labels = classify(current_sample, TRUE_FUNCTION).reshape((-1, 1))
            data = np.hstack((current_sample, labels))
            hypothesis = learning(data)

            POSITIVE_FEATURES = FEATURES[TRUE_DATA[:, -1] == 1]
            prediction = classify(POSITIVE_FEATURES, hypothesis)
            import pdb; pdb.set_trace()
            correct = np.count_nonzero(prediction)
            error = 1 - (correct/ len(prediction))
            

    # Testing
    # fake_data = np.random.choice(2, (10, 10))
