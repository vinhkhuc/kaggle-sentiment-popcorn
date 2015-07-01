#### Gated recurrent ###
python ../scripts/passage_nn.py --ptrain data/small-train-pos.txt --ntrain data/small-train-neg.txt --ptest data/valid-pos.txt --ntest data/valid-neg.txt --modeltype gated_recurrent --out scores/PASSAGE_GR-VALID
tail -n 5000 scores/PASSAGE_GR-VALID > tmp; mv tmp scores/PASSAGE_GR-VALID
python ../scripts/passage_nn.py --ptrain data/full-train-pos.txt --ntrain data/full-train-neg.txt --ptest data/test-pos.txt --ntest data/test-neg.txt --modeltype gated_recurrent --out scores/PASSAGE_GR-TEST
tail -n 25000 scores/PASSAGE_GR-TEST > tmp; mv tmp scores/PASSAGE_GR-TEST

## LSTM ###
#python ../scripts/passage_nn.py --ptrain data/small-train-pos.txt --ntrain data/small-train-neg.txt --ptest data/valid-pos.txt --ntest data/valid-neg.txt --modeltype lstm_recurrent --out scores/PASSAGE_LSTM-VALID
#tail -n 5000 scores/PASSAGE_LSTM-VALID > tmp; mv tmp scores/PASSAGE_LSTM-VALID
#python ../scripts/passage_nn.py --ptrain data/full-train-pos.txt --ntrain data/full-train-neg.txt --ptest data/test-pos.txt --ntest data/test-neg.txt --modeltype lstm_recurrent --out scores/PASSAGE_LSTM-TEST
#tail -n 25000 scores/PASSAGE_LSTM-TEST > tmp; mv tmp scores/PASSAGE_LSTM-TEST

