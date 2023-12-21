# Geometric Random Walk Graph Neural Networks via Implicit Layers

This repository is the official implementation of [Geometric Random Walk Graph Neural Networks via Implicit Layers](https://proceedings.mlr.press/v206/nikolentzos23c/nikolentzos23c.pdf).

## Requirements

Code is written in Python 3.9 and requires:
* PyTorch 1.12
* PyTorch Geometric 2.1


## Datasets

The datasets are automatically downloaded when the user executes the script


## Training and Evaluation

To train and evaluate the model, specify the dataset and the hyperparameters in the main.py file and then execute:

```
python main.py
```

## Cite

Please cite our paper if you use this code:
```
@inproceedings{nikolentzos2023graph,
  title={Geometric Random Walk Graph Neural Networks via Implicit Layers},
  author={Nikolentzos, Giannis and Vazirgiannis, Michalis},
  booktitle={Proceedings of the 26th International Conference on Artificial Intelligence and Statistics},
  pages={2035--2053},
  year={2023}
}
```
