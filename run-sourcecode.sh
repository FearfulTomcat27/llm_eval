#!/bin/bash

python gen_withcode.py && \
python eval_multiple.py && \
docker run --rm --network none -v ./tutorial:/tutorial:rw multipl-e-eval --dir /tutorial --output-dir /tutorial --recursive && \
python MultiPL-E/pass_k.py tutorial/*
