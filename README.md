# Denoising Graph Neural Networks

This repository is the implementation of [Denoising Graph Neural Networks](https://openreview.net/forum?id=r1xOmNmxuN). This
is original a fork of powerful-gnns (How Powerful are Graph Neural Networks.

## Installation
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 0.4.1 and 1.0.0 versions.

Then install the other dependencies.
```
pip install -r requirements.txt
```

## Test run
Unzip the dataset file
```
unzip dataset.zip
```

and run

```
python main.py
```

Default parameters are not the best performing-hyper-parameters. Hyper-parameters need to be specified through the commandline arguments. 

Type

```
python main.py --help
```

to learn hyper-parameters to be specified.

