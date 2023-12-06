# SantAI Claus
Santa Claus needs our help. There are too many children for him to check whether they have been naughty and nice before Christmas. Let's check his list twice with an ensemble deep learning model.

# Usage
The project contains two separate subprojects:

1. **naughty_and_nice_list**: Generate a dataset of naughty and nice people.
2. **santas_ai_helper**: Train and test the ensemble model.

## Naughty and nice list
This subproject is cloned from [the official Llama 2 repository](https://github.com/facebookresearch/llama).

How to use it:

1. Download Llama-2 Chat weights by visitng the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and requesting a link for download.
2. Use the README in the subproject to install dependencies. We recommend to install inside of this Docker image to ensure that you have all NVIDIA/Cuda dependencies: nvcr.io/nvidia/pytorch:23.05-py3
3. Generate fake people like this:\
\
```python generate_people.py 1000```
\
\
This will save 1000 JSON files in `naughty_and_nice_list/data`, where each file represents a fake person, including name, age, country, personality type, naughty/nice score, and mentions of when key events happened to the person in the past year.
4. Generate descriptions of each person using `generate_descriptions.py`. This will use the metadata of each fake person from step 3 to prompt LLama-2 to generate a description of what the person has done in the past year. You will use the Llama-2 weights and tokenizer from step 1 here - we have downloaded the 70B parameter version in this example:\
\
```torchrun --nproc_per_node 8 generate_description.py --ckpt_dir llama-2-70b-chat --tokenizer_path tokenizer.model```
\
\
NB. `nproc_per_node` is the number of GPUs that will be used for generation. It must be set to 8 for the 70B parameter version, but 2 and 1 for the 13B and 7B parameter versions, respectively. You will need a machine with the required amount of GPUs available in order to use this.

## Santa's AI Helper
Follow the instructions in _santas_ai_helper/README.md_.
