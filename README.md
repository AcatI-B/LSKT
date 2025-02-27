# LSKT

Code for _Learning states enhanced knowledge tracing: Simulating the diversity in real-world learning process_ 

## Installation

```bash
poetry install
```

## Usage

### Train

Train LSKT:

```bash
python scripts/train.py -m LSKT -d assist09 -bs 16 -tbs 16 -p -emb 3pl [-o output/assist09_result] [--device cuda]
```


### Evaluate

Evaluate LSKT:

```bash
python scripts/test.py -m LSKT -d assist09 -bs 16 -tbs 16 -p -emb 3pl -f assist09_result/**.pt [--device cuda]
```
