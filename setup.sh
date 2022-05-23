#!/bin/bash
if [[ ! -d env ]]
then
    python3 -m venv env
	source env/bin/activate
	python3 -m pip install --upgrade pip
	python3 -m pip install \
        numpy==1.21.6 \
        numba==0.55.1 \
        lightgbm==3.2.1 \
        matplotlib==3.5.2 \
        black==22.3.0 \
        flake8==4.0.1 \
        isort==5.10.1
else
    source env/bin/activate
fi
