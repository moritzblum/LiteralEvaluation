# LiteralE
Knowledge Graph Embeddings learned from the structure and literals of knowledge graphs.

ArXiv link for the paper: [Incorporating Literals into Knowledge Graph Embeddings](https://arxiv.org/abs/1802.00934)

### Credits

This work is built on top of Tim Dettmers' ConvE codes: <https://github.com/TimDettmers/ConvE>.

### Getting Started

**Note:** Python 3.6+ is required.

Note that we only support computation on GPU (CUDA). We have tested our code with Nvidia Titan Xp (12GB) and RTX 2080Ti (11GB). 6 or 8GB of memory should also be enough though we couldn't test them.

1. Install PyTorch. We have verified that version 1.2.0 works.
2. Install other requirements: `pip install -r requirements.txt`
3. Run `chmod +x preprocess.sh && ./preprocess.sh`
4. Install spacy model: `python -m spacy download en && python -m spacy download en_core_web_md`
5. Preprocess datasets (do these steps for each dataset in `{FB15k, FB15k-237, YAGO3-10}`):
    1. `python main_literal.py dataset {FB15k, FB15k-237, YAGO3-10} epochs 0 process True`
    2. Numerical literals: `python preprocess_num_lit.py --dataset {FB15k, FB15k-237, YAGO3-10}`
    3. Text literals: `python preprocess_txt_lit.py --dataset {FB15k, FB15k-237, YAGO3-10}`


### Reproducing Paper's Experiments

For DistMult+LiteralE and ComplEx+LiteralE:
```
python main_literal.py dataset {FB15k, FB15k-237, YAGO3-10} model {DistMult, ComplEx} input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True
```

For ConvE+LiteralE:
```
python main_literal.py dataset {FB15k, FB15k-237, YAGO3-10} model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 embedding_dim 200 batch_size 128 epochs 100 lr 0.001 process True
```

For DistMult+LiteralE with numerical and textual literals:
```
python main_literal.py dataset {FB15k, FB15k-237, YAGO3-10} model DistMult_text input_drop 0.2 embedding_dim 100 batch_size 128 epochs 100 lr 0.001 process True
```

NB: For base models, replace `main_literal.py` with `main.py`.


# Variations

## FB15k
* 1-1: 0.24237918215613383
* 1-n: 0.22899628252788104
* n-1: 0.28847583643122676
* n-n: 0.24014869888475837

## FB15k-237
* 1-1: 0.07172995780590717
* 1-n: 0.10970464135021098
* n-1: 0.34177215189873417
* n-n: 0.4767932489451477

## YAGO3-10
* 1-1: 0.05405405405405406
* 1-n: 0.13513513513513514
* n-1: 0.2702702702702703
* n-n: 0.5405405405405406

## WN18RR
* 1-1: 0.18181818181818182
* 1-n: 0.36363636363636365
* n-1: 0.2727272727272727
* n-n: 0.18181818181818182
