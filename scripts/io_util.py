"""
IO Utilities
"""

def read_file(fname, target_label):
    with open(fname) as f:
        X = [line.lower() for line in f.readlines()]
        Y = [target_label] * len(X)
        return X, Y

def load_data(pfile, nfile):
    """
    Load data from files containing positive and negative reviews
    """
    posX, posY = read_file(pfile, target_label=1)
    negX, negY = read_file(nfile, target_label=0)
    return posX + negX, posY + negY