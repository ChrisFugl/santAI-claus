# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from textwrap import dedent

import fire

from llama import Llama


MAX_SEQUENCE_LENGTH = 512
MAX_BATCH_SIZE = 1

SYSTEM_PROMPT = """
We want to help Santa Claus decide who to give presents to this year. We want to train a machine learning model to predict whether a person has been naughty or nice, but we need your help to collect training data.

You will receive a few facts about a fictional person. Your job is to use those facts to invent a list of key events that the person has done in the past year. Your response should only include this list.

Your response must adhere to this format:
1. [month]: [event]
2. [month]: [event]
...

You should only describe what happened. Take particular care to use neutral language and to not justify bad behaviour or praise good behaviour. You should never judge whether (or even the degree to which) the person has been naughty or nice.
""".strip()


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
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

    num_total = 3
    num_naughty = 2
    num_nice = num_total - num_naughty

    nice = f"{num_nice} nice thing"
    if num_nice != 1:
        nice += "s"

    naughty = f"{num_naughty} naughty thing"
    if num_naughty != 1:
        naughty += "s"

    people = [
        dedent("""
            Name: John
            Age: 10
            Location: Oslo
            Occupation: Student
            Key events:
                * January: good
                * February: bad
                * May: slightly good
                * August: neutral
                * November: very bad
        """)
    ]

    for person in people:
        dialog = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": person},
        ]
        dialogs = [dialog]
        result = generator.chat_completion(
            dialogs,
            max_gen_len=None,
            temperature=temperature,
            top_p=top_p,
        )[0]

        print(result["generation"]["content"])
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
