[![Build](https://github.com/feldberlin/wavenet/workflows/CI/badge.svg)](https://github.com/feldberlin/wavenet/actions)

# Wavenet

An unconditioned Wavenet implementation with fast generation.


## Installation

Requires python 3.7.2 or greater, will install into a virtualenv.

```bash
bin/install
```

## Training

Training is done by running notebooks from the command line. Here's an
example call, which will run `notebooks/train/maestro.ipynb`:

```bash
bin/train maestro -p batch_size 24 -p learning_rate 0.001
```
