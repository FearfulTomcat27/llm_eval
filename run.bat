@echo off

python gen_mermaid.py && ^
python gen_multiple.py && ^
python eval_multiple.py && ^
docker run --rm --network none -v ".\tutorial:/tutorial:rw" multipl-e-eval --dir /tutorial --output-dir /tutorial --recursive && ^
python MultiPL-E\pass_k.py tutorial\*

pause