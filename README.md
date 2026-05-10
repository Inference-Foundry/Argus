# Argus

Argus is an AI-generated image detection research project focused on reproducible binary classification (Real vs. AI-generated) with PyTorch baselines and scaffolding for probing frozen JEPA representations.

## Setup

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Data layout

Point training to a dataset directory with this layout:

```text
dataset/
  real/
    image_1.jpg
    ...
  fake/
    image_1.jpg
    ...
```

## Train baseline

```bash
python train.py --data-dir /absolute/path/to/dataset
```
