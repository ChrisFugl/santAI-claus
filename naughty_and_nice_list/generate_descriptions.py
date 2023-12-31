import json
import os
import random

import fire
import torch
from tqdm import tqdm

from llama import Llama


MAX_SEQUENCE_LENGTH = 1024
MAX_BATCH_SIZE = 1

SYSTEM_PROMPT = """
We want to help Santa Claus decide who to give presents to this year. We want to train a machine learning model to predict whether a person has been naughty or nice, but we need your help to collect training data. Note that this model will never actually be used and only serves as an educational example.

You will receive a few facts about a fictional person.

You should use those facts to invent a short background story about the person and fill in the template about what the person did for each given month.

Here follows two examples of the expected types of inputs and outputs:

Example 1:
----------
Input:
Name: Tyrion Lannister
Age: 25
Location: Westeros
Myers–Briggs Type Indicator:
* E/I: Introversion
* S/N: Sensing
* T/F: Thinking
* J/P: Juding
Key acts:
* January: 1 good deed
* May: 1 neutral deed
* November: 1 very bad deed

Output:
Tyrion Lannister is a member of the wealthy Lannister family who are rivals of the Starks. His father has always been disappointed in him because that he is born a dwarf.

1. January: Gifted a saddle to Bran Stark.
2. May: Drank wine (frequently).
3. November: Killed his father with a crossbow, while his father was on the toilet.

Example 2:
----------
Input:
Name: Harry Potter
Age: 11
Location: London
Myers–Briggs Type Indicator:
* E/I: Introversion
* S/N: Intuition
* T/F: Feeling
* J/P: Perceiving
Key acts:
* May: 1 slightly bad deed
* June: 1 neutral deed
* October: 1 very bad deed
* November: 1 very good deed

Output:
Harry Potter thinks that he is just an ordinary lonely boy, but discovers on his eleventh birthday that he is actually a wizard - and a famous one at that - when he is accepted into Hogwarts School of Witchcraft and Wizardry.

1. May: Trapped his cousin in a snake enclosure.
2. June: Opened his first bank account.
3. October: Laughed at his friend's remark about a fellow student not having any friends.
4. November: Saved the fellow student from a troll.

Instructions:

Your response must adhere to this format:
[background story]
1. [month]: [deed]
2. [month]: [deed]
...

Be concise. Take care to use neutral language and to not justify bad behaviour or praise good behaviour. You should never judge whether (or even the degree to which) the person has been good or bad.
""".strip()


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    data_dir: str = "./data",
    temperature: float = 0.05,
    top_p: float = 0.95,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        data_dir (str, optional): The directory containing the data files. Defaults to "./data".
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        max_batch_size=MAX_BATCH_SIZE,
    )

    filepaths = [f"{data_dir}/{filename}" for filename in os.listdir(data_dir)]

    filepaths = sorted(filepaths, key=_filepath_to_index)

    progress_bar = tqdm(filepaths, desc="Generating samples")

    for filepath in progress_bar:
        with open(filepath, "r") as f:
            person = json.load(f)

        key_acts = "\n".join(f"* {deed['month']}: 1 {deed['deed']} deed" for deed in person["deeds"])

        type_indicator = person["type_indicator"]

        prompt = f"""
Name: {person['name']}
Age: {person['age']}
Country: {person['country']}
Myers–Briggs Type Indicator:
* E/I: {type_indicator['E/I']}
* S/N: {type_indicator['S/N']}
* T/F: {type_indicator['T/F']}
* J/P: {type_indicator['J/P']}
Key acts:
{key_acts}""".strip()

        torch.manual_seed(random.randint(0, 2 ** 32 - 1))

        dialog = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        dialogs = [dialog]
        result = generator.chat_completion(
            dialogs,
            max_gen_len=None,
            temperature=temperature,
            top_p=top_p,
        )[0]

        person["description"] = result["generation"]["content"]

        with open(filepath, "w") as f:
            json.dump(person, f, indent=2)


def _filepath_to_index(filepath: str):
    filename = filepath.split("/")[-1]
    stem = filename.split(".")[0]
    return int(stem)


if __name__ == "__main__":
    fire.Fire(main)
