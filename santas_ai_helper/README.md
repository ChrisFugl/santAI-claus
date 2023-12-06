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
