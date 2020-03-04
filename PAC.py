import numpy as np

np.random.seed(23)
n = 20

features = np.random.choice(2, (10000, 20))
true_labels = label_data(features)
data = np.hstack((features, true_labels))
#Testing
# fake_data = np.random.choice(2, (10, 10))


true_function = [(1, 1), (3, 0), (5, 1), (7, 1)]


def classify_example(example):
    return all([example[index] == label for index, label in true_function])

def classify(features):
    return np.array([classify_example(example) for example in features])

def learning(data):
    hypo = [1 for i in range(n)]
    for example in data:
        pass