#!/usr/bin/env bash

######################################################
# NOTE: This script is adapted from iclr15's data.sh
######################################################

# This function will convert text to lowercase and will disconnect punctuation and special symbols from words
function normalize_text {
  awk '{print tolower($0);}' < $1 | sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
  -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
  -e 's/\;/ \; /g' -e 's/\:/ \: /g' > $1-norm
}

mkdir data

# Normalize training data
python ../scripts/extract_reviews.py ../data/labeledTrainData.tsv positive > data/full-train-pos.txt
python ../scripts/extract_reviews.py ../data/labeledTrainData.tsv negative > data/full-train-neg.txt
python ../scripts/extract_reviews.py ../data/unlabeledTrainData.tsv > data/train-unsup.txt

# Normalize test data and assume that put normalized reviews into positive test file
# and create an empty negative test file since Kaggle's test data doesn't have labels.
python ../scripts/extract_reviews.py ../data/testData.tsv > data/test-pos.txt
touch data/test-neg.txt

for f in data/full-train-pos.txt data/full-train-neg.txt data/train-unsup.txt data/test-pos.txt data/test-neg.txt; do
  normalize_text $f
  mv $f-norm $f
done

cd data
head -n 10000 full-train-pos.txt > small-train-pos.txt
head -n 10000 full-train-neg.txt > small-train-neg.txt
tail -n 2500 full-train-pos.txt > valid-pos.txt
tail -n 2500 full-train-neg.txt > valid-neg.txt
cd ..
