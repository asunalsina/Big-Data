import numpy as np
from random import sample
import pandas as pd
from tqdm import tqdm


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
    POSITIVE_FEATURES = FEATURES[TRUE_DATA[:, -1] == 1]
    sample_sizes = list(range(50, 500, 50)) + [486]

    true_loss = {}
    real_hypothesis = {}

    for m in sample_sizes:
        i = 0
        i_s = []

        while i < 10000:
            i_s.append(i)
            idx = np.random.randint(FEATURES.shape[0], size=m)
            current_sample = FEATURES[idx, :]
            labels = classify(current_sample, TRUE_FUNCTION).reshape((-1, 1))
            previous_i = i
            if not np.any(labels):
                continue
            data = np.hstack((current_sample, labels))
            hypothesis = learning(data)
            prediction_true_loss = classify(data, hypothesis)
            prediction_hypothesis = classify(POSITIVE_FEATURES, hypothesis)
            correct_true_loss = (np.count_nonzero(prediction_true_loss))/len(prediction_true_loss)
            correct_hypothesis = (np.count_nonzero(prediction_hypothesis))/len(prediction_hypothesis)


            if m in true_loss:
                true_loss[m].append(correct_true_loss)
            else:
                true_loss[m] = [correct_true_loss]
            if m in real_hypothesis:
                real_hypothesis[m].append(correct_hypothesis)
            else:
                real_hypothesis[m] = [correct_hypothesis]
            i += 1
        print(f'Total iterations: {len(i_s)}')

    results_tl = pd.DataFrame.from_dict(true_loss)
    results_tl.to_csv('true_loss.csv')

    results_rh = pd.DataFrame.from_dict(real_hypothesis)
    results_rh.to_csv('real_hypothesis.csv')

