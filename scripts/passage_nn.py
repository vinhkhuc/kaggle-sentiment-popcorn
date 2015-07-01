"""
Adapted from Passage's sentiment.py at
https://github.com/IndicoDataSolutions/Passage/blob/master/examples/sentiment.py

License: MIT
"""

import argparse
import numpy as np

from passage.models import RNN
from passage.updates import Adadelta
from passage.layers import Embedding, GatedRecurrent, LstmRecurrent, Dense
from passage.preprocessing import Tokenizer

from io_util import load_data

def main(ptrain, ntrain, ptest, ntest, out, modeltype):
    assert modeltype in ["gated_recurrent", "lstm_recurrent"]

    print("Using the %s model ..." % modeltype)
    print("Loading data ...")
    trX, trY = load_data(ptrain, ntrain)
    teX, teY = load_data(ptest, ntest)

    tokenizer = Tokenizer(min_df=10, max_features=100000)
    trX = tokenizer.fit_transform(trX)
    teX = tokenizer.transform(teX)

    print("Training ...")
    if modeltype == "gated_recurrent":
        layers = [
            Embedding(size=256, n_features=tokenizer.n_features),
            GatedRecurrent(size=512, activation='tanh', gate_activation='steeper_sigmoid',
                           init='orthogonal', seq_output=False, p_drop=0.75),
            Dense(size=1, activation='sigmoid', init='orthogonal')
        ]
    else:
        layers = [
            Embedding(size=256, n_features=tokenizer.n_features),
            LstmRecurrent(size=512, activation='tanh', gate_activation='steeper_sigmoid',
                          init='orthogonal', seq_output=False, p_drop=0.75),
            Dense(size=1, activation='sigmoid', init='orthogonal')
        ]

    model = RNN(layers=layers, cost='bce', updater=Adadelta(lr=0.5))
    model.fit(trX, trY, n_epochs=10)

    # Predicting the probabilities of positive labels
    print("Predicting ...")
    pr_teX = model.predict(teX).flatten()

    predY = np.ones(len(teY))
    predY[pr_teX < 0.5] = -1

    with open(out, "w") as f:
        for lab, pos_pr, neg_pr in zip(predY, pr_teX, 1 - pr_teX):
            f.write("%d %f %f\n" % (lab, pos_pr, neg_pr))

if __name__ == "__main__":
    """
    Usage :

    python passage_nn.py\
        --ptrain /PATH/data/full-train-pos.txt\
        --ntrain /PATH/data/full-train-neg.txt\
        --ptest /PATH/data/test-pos.txt\
        --ntest /PATH/data/test-neg.txt\
         --modeltype model_type\
         --out TEST-SCORE
    """

    parser = argparse.ArgumentParser(description='Use Passage for sentiment analysis.')
    parser.add_argument('--ptrain', help='path of the text file TRAIN POSITIVE')
    parser.add_argument('--ntrain', help='path of the text file TRAIN NEGATIVE')
    parser.add_argument('--ptest', help='path of the text file TEST POSITIVE')
    parser.add_argument('--ntest', help='path of the text file TEST NEGATIVE')
    parser.add_argument('--modeltype', help='Passage\'s model type: gated_recurrent or lstm_recurrent')
    parser.add_argument('--out', help='path and filename for score output')
    args = vars(parser.parse_args())

    main(**args)
