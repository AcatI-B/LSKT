# LSKT

Code for _Learning states enhanced knowledge tracing: Simulating the diversity in real-world learning process_

## Installation

```bash
poetry install
```

## Usage

### Train

Train LSKT :

```bash
python scripts/train.py -m LSKT -d assist09 -bs 16 -tbs 16 -p -emb 3pl [-o output/assist09_result] [--device cuda] 
```

For more options, run:

```bash
python scripts/train.py -h
```

### Evaluate

Evaluate DTransformer:

```bash
python scripts/test.py -m DTransformer -d [assist09,assist17,algebra05,statics] -bs 32 -p -f [output/best_model.pt] [--device cuda]
```
