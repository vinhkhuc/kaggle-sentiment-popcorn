import os
import numpy as np
import pandas as pd

def normalize(data):
    norm = np.sqrt((data ** 2).sum())
    return data / norm

def load(e, classifiers, path="scores"):
    """ returns an array containing the scores of the specified classifiers.
    Note that we center the scores to 0."""
    assert e in ["TEST", "VALID"]
    probas = []
    for c in classifiers:
        data = np.loadtxt(os.path.join(path, "-".join([c, e])))
        probas += [data[:, 1]] # probabilities of positive labels
    x = np.vstack(probas).T
    y = np.vstack([np.ones((x.shape[0] // 2, 1)), np.zeros((x.shape[0] // 2, 1))])
    return x, y

def accuracy(k, d):
    """ d is an array of shape nsamples, nclassifiers
    k is an array of size nclassifiers
    this function return the accuracy of the linear combination
    with k coefficients
    """
    x, y = d
    output = [k[i] * x[:, i] for i in range(len(k))]
    pred = np.vstack(output).sum(0)
    cnt = ((pred < 0) == y.T).mean()
    return cnt * 100.

def predict_proba(k, d):
    """
    Same as accuracy(k, d) but it returns prediction with probabilities instead of accuracy
    """
    x, y = d
    output = [k[i] * x[:, i] for i in range(len(k))]
    return np.vstack(output).sum(0)

def ensemble(d, classifiers):
    """ computes the weigths of each ensemble
    according to the contribution of each model
    on the valid set """
    output = []
    x, y = d
    for i, c in enumerate(classifiers):
        k = np.zeros(len(classifiers))
        k[i] = 1
        acc = accuracy(k, d)
        output += [acc]
    k = np.array(output)
    k /= k.sum()
    best = accuracy(k, d)
    return k, best

if __name__ == "__main__":

    # Ensemble all classifiers
    all = ["NBSVM", "PARAGRAPH", "PASSAGE_GR"]
    valid, test = load("VALID", all), load("TEST", all)
    k, _ = ensemble(valid, all)
    results = predict_proba(k, test)

    pd.DataFrame(results.T).to_csv("ensemble.csv", index=False, header=["pos"])
