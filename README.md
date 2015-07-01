### What is it? ###
This is the source code of my submission for the Kaggle competition ["Bag of Words Meets Bags of Popcorn"]
(https://www.kaggle.com/c/word2vec-nlp-tutorial). The leader board score is 0.9766.

The model is an ensemble of NBSVM, Paragraph Vector and Gated Recurrent Neural Network. It is based on the code 
from the paper ["Ensemble of Generative and Discriminative Techniques for Sentiment Analysis of Movie Reviews"]
(http://arxiv.org/abs/1412.5335) by Grégoire Mesnil and others.

### How to run ###
* Data should be put into the directory [data](https://github.com/vinhkhuc/kaggle-sentiment-popcorn/tree/master/data).

* The code requires numpy, pandas, Theano and passage which can be installed by running: 
```
sudo pip install -r requirements.txt
```

* After that, run the following command to generate the submission file "ensemble-submission.csv":
```
chmod +x run.sh && time ./run.sh
```

* The code was tested on Ubuntu 14.04. It took about 1h16' on an i7 quad-core with a GTX Titan. The submission should give 
the score of ~0.976 on the leader board (it can be slightly different due to data shuffling during training).

### License ###
Since the code is based on Grégoire Mesnil's work at [iclr15](https://github.com/mesnilgr/iclr15), all files except 
[scripts/passage_nn.py](https://github.com/vinhkhuc/kaggle-sentiment-popcorn/blob/master/scripts/passage_nn.py)
are released under [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).
