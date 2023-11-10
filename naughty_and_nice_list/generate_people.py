import json
import os
import random

import fire
from faker import Faker


MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
]

DEED_SCORES = {
    "very bad": -2,
    "bad": -1,
    "neutral": 0,
    "good": 1,
    "very good": 2,
}

DEEDS = list(DEED_SCORES.keys())


def main(
    num_samples: int,
    output_dir: str = "./data",
    min_age: int = 3,
    max_age: int = 17,
    min_deeds: int = 2,
    max_deeds: int = 6,
):
    faker = Faker()

    os.makedirs(output_dir, exist_ok=True)

    for sample_index in range(num_samples):
        name = faker.name()
        age = random.randint(min_age, max_age)
        country = faker.country()

        num_deeds = random.randint(min_deeds, max_deeds)

        months = random.sample(MONTH_NAMES, num_deeds)
        months.sort(key=MONTH_NAMES.index)

        deeds = []
        total_score = 0
        for deed_index in range(num_deeds):
            deed = random.choice(DEEDS)
            score = DEED_SCORES[deed]
            total_score += score
            month = months[deed_index]
            deeds.append({
                "month": month,
                "deed": deed,
            })

        type_indicator = {
            "E/I": random.choice(["Extraversion", "Introversion"]),
            "S/N": random.choice(["Sensing", "Intuition"]),
            "T/F": random.choice(["Thinking", "Feeling"]),
            "J/P": random.choice(["Judging", "Perceiving"]),
        }

        person = {
            "name": name,
            "age": age,
            "country": country,
            "type_indicator": type_indicator,
            "deeds": deeds,
            "score": total_score,
        }

        output_path = f"{output_dir}/{sample_index}.json"
        with open(output_path, "w") as f:
            json.dump(person, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
