import gzip
import json
import os
import re
import textwrap

from tqdm import tqdm

from data.utils import get_entry_point_from_multiple
from gen_multiple import read_jsonl, get_config


def __count_function__(text):
    # 去掉 main 函数
    function_pattern = (
        r"\b(public|protected|private|static|\s)\s+.*?\s+.*?\s*\([^)]*\)\s*\{"
    )
    functions = re.findall(function_pattern, text, re.DOTALL)
    return len(functions)


def extract_java_code(text, entry_point):
    entry_point = get_entry_point_from_multiple(entry_point)
    main_block_pattern = r"public static void main\(String\[\] args\) {.*?    \}"
    text = re.sub(main_block_pattern, "", text, flags=re.DOTALL)

    function_num = __count_function__(text)

    if function_num == 1:
        code_pattern = re.compile(
            rf"```(?:[Jj]ava\n)?.*?public static.*?{entry_point}.*?{{\n(.*?)\n    }}\n.*?```",
            re.DOTALL,
        )
        code_block = code_pattern.search(text)
    else:
        code_pattern = re.compile(
            rf"```(?:[Jj]ava\n)?.*?public static.*?{entry_point}.*?{{\n(.*)\n    }}\n.*?```",
            re.DOTALL,
        )
        code_block = code_pattern.search(text)

    if code_block is None:
        code_pattern = re.compile(
            r"```[Jj]ava\n(.*?)(\n    }\n|\n    }\n}\n)?```", re.DOTALL
        )
        code_block = code_pattern.search(text)

    if code_block is not None:
        return code_block.group(1)

    return textwrap.indent(text, " " * 4)


def write_jsonl_gz(task, config):
    with gzip.open(
        os.path.join(config.get("eval", "output_folder"), f"{task['name']}.json.gz"),
        "wb",
    ) as fp:
        fp.write((json.dumps(task) + "\n").encode("utf-8"))


def extract_cpp_code(text, entry_point):
    entry_point = re.sub(r"HumanEval_\d+_", "", entry_point)
    code_pattern = re.compile(
        rf"(?:```cpp\n)?.*?{entry_point}\(.*?\) {{\n(.*?)\n}}\n(?:```)?", re.DOTALL
    )
    code_block = code_pattern.search(text)

    if code_block is None:
        code_pattern = re.compile(r"(?:```cpp\n)?(.*?)\n\}\n(?:```)?", re.DOTALL)
        code_block = code_pattern.search(text)

    if code_block is not None:
        return code_block.group(1)

    raise ValueError(f"Code block not found:{text}")


def get_extractor(language):
    extractor = {
        "java": extract_java_code,
        "cpp": extract_cpp_code,
    }
    return extractor[language]


def main():
    config = get_config()
    os.makedirs(config.get("eval", "output_folder"), exist_ok=True)
    extractor = get_extractor(config.get("eval", "language"))
    tasks = read_jsonl(config.get("eval", "input_file"))
    for task in tqdm(tasks):
        completions = []
        for completion in task["completions"]:
            completions.append(extractor(completion, task["name"]))
        task["completions"] = completions
        write_jsonl_gz(task, config)


if __name__ == "__main__":
    main()
