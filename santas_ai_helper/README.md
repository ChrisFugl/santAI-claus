# Santa's AI Helper

_He is making a list and checking it twice, he is going to find out who is naughty or nice!_

This of course means that he is using an ensemble of two deep learning models to classify the children of the world into two categories: naughty or nice.

Santa's AI Helper uses two BERT models and averages their logits to make a final prediction.

## Installation
We recommend to use the attached devcontainer (for VS Code). This will ensure that you have the required CUDA dependencies.

Run the following command to install the required Python packages:

``` sh
pip install -r requirements.txt
```

## Usage
There are three scripts to train and test the models:

* train.py  -  train a single model
* test.py   -  evaluate one or more models in an ensemble
* speed.py  -  measure inference time for an ensemble model


Use `--help` to get more information about how to use each script. For example, `python train.py --help` will give you this:

``` txt
usage: train.py [-h] --data-dir DATA_DIR --batch-size BATCH_SIZE --learning-rate LEARNING_RATE --epochs EPOCHS [--min-age MIN_AGE] [--max-age MAX_AGE] [--drop-person-prop DROP_PERSON_PROP] --name NAME

Train the model.

options:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   Path to the directory containing the data.
  --batch-size BATCH_SIZE
                        The batch size to use for training, validation, and testing.
  --learning-rate LEARNING_RATE
                        The learning rate to use for training.
  --epochs EPOCHS       The number of epochs to use for training.
  --min-age MIN_AGE     The minimum age of people included in the dataset.
  --max-age MAX_AGE     The maximum age of people included in the dataset.
  --drop-person-prop DROP_PERSON_PROP
  --name NAME           The name of the model.
```
