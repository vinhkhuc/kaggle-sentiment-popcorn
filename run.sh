#!/usr/bin/env bash
chmod +x scripts/*.sh
chmod +x iclr15/scripts/*.sh

cd kaggle_run
mkdir scores

../scripts/prepare_data.sh
../scripts/install_liblinear.sh
../scripts/run_nbsvm.sh
../iclr15/scripts/paragraph.sh
../scripts/run_passage.sh

python ../scripts/ensemble.py
python ../scripts/make_submission.py