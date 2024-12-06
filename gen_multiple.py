from openai import OpenAI
from tqdm import tqdm

from data.utils import (
    extract_description,
    get_entry_point_from_multiple,
    get_config,
    extract_function_declare,
    load_datasets,
)
from data.utils import read_jsonl, get_humaneval_task_id
from human_eval.data import write_jsonl


def construct_prompt(problem, flowchart, language: str):
    requirement = extract_description(problem["prompt"])
    function_name = get_entry_point_from_multiple(problem["name"])
    function_declare = extract_function_declare(problem)
    user_content = (
        f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n### Instruction:\nGenerate a static {function_name} function using {language}.Require the following requests.\n{requirement}\n\nThe function is declared as follows.\n{function_declare}\n\n### Response:"""
        if flowchart == {}
        else f"""Below is a plantuml syntax and an instruction that describes a task. Write a response that appropriately completes the request.\n\n### PlantUML:\n{flowchart["completion"]}\n\n### Instruction:\nGenerate a static {function_name} function using {language}. Require the following requests.\n{requirement}\n\nThe function is declared as follows.\n{function_declare}\n\n### Response:\n"""
    )
    messages = [
        {
            "role": "system",
            "content": f"You're a very experienced {language} programmer.",
        },
        {"role": "user", "content": user_content},
    ]
    return messages


def generate_one_completion(client, messages, config):
    response = client.chat.completions.create(
        model=config.get("generate", "model"),
        messages=messages,
        temperature=config.getfloat("generate", "temperature"),
        top_p=config.getfloat("generate", "top_p"),
        max_tokens=config.getint("generate", "max_tokens"),
    )

    return response.choices[0].message.content


def get_flowcharts(file):
    data = read_jsonl(file)
    return {item["task_id"]: item for item in data}


def main():
    config = get_config()
    client = OpenAI(
        api_key=config.get("generate", "api_key"),
        base_url=config.get("generate", "base_url"),
    )
    problems = load_datasets(config.get("generate", "language"))
    flowcharts = get_flowcharts(f"plantuml-{config.get('generate', 'model')}.jsonl")

    output_file = f"{config.get('generate','model')}-{config.get('generate','language')}-{'optimize' if config.getboolean('generate','optimize') else 'baseline'}.jsonl"
    data = []
    print(f"Generating {config.get('generate','language')} completions...")
    for problem in tqdm(problems):
        flowchart = (
            flowcharts[get_humaneval_task_id(problem["name"])]
            if config.getboolean("generate", "optimize")
            else {}
        )
        completions = [
            generate_one_completion(
                client,
                construct_prompt(
                    problem, flowchart, config.get("generate", "language")
                ),
                config,
            )
            for _ in range(config.getint("generate", "k"))
        ]
        task = {
            "name": problem["name"],
            "language": config.get("generate", "language"),
            "temperature": config.getfloat("generate", "temperature"),
            "top_p": config.getfloat("generate", "top_p"),
            "max_tokens": config.getint("generate", "max_tokens"),
            "prompt": problem["prompt"],
            "tests": problem["tests"],
            "completions": completions,
            "stop_tokens": problem["stop_tokens"],
        }
        data.append(task)
    write_jsonl(output_file, data)


if __name__ == "__main__":
    main()
