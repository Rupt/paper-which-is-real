#!/bin/bash
if [[ ! -d env ]]
then
    python3 -m venv env
	source env/bin/activate
	python3 -m pip install \
        numpy==1.20.0 \
        numba==0.54.0 \
        lightgbm==3.2.1 \
        xgboost==1.5.0 \
        matplotlib==3.4.3 \
        black==21.9b0
else
    source env/bin/activate
fi
