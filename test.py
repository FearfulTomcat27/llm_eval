import gzip
import json
import os
import time
from typing import Iterable, Dict

from data.utils import stream_jsonl
from gen_multiple import get_config


def read_json(file) -> Dict or list:
    with open(file, "r") as fp:
        return json.load(fp)


def generator_to_list(generator: Iterable[Dict]) -> list:
    return list(generator)


def get_multiple_result_file(dir) -> list:
    return [file for file in os.listdir(dir) if file.endswith("results.json.gz")]


def read_multiple_result(dir: str) -> list:
    for file in get_multiple_result_file(dir):
        with gzip.open(os.path.join(dir, file), "r") as fp:
            yield json.load(fp)


def main():
    config = get_config()
    humaneval_result = generator_to_list(
        stream_jsonl(config.get("analyze", "humaneval_result"))
    )
    multiple_result = generator_to_list(
        read_multiple_result(config.get("analyze", "multiple_result"))
    )
    time.sleep(1)


if __name__ == "__main__":
    main()
