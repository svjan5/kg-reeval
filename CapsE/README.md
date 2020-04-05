# Re-Evaluating: CapsE

### Requirements

- Python 3.x and Tensorflow >= 1.6

### Training

```shell
# FB15k-237
$ python CapsE.py --embedding_dim 100 --num_epochs 31 --num_filters 50 --learning_rate 0.0001 --name FB15k-237 --useConstantInit --savedEpochs 30 --model_name fb15k237

# WN18RR
$ python CapsE.py --embedding_dim 100 --num_epochs 31 --num_filters 400 --learning_rate 0.00001 --name WN18RR --savedEpochs 30 --model_name wn18rr
```

### Original Evaluation:

Depending on the memory resources, you should change the values of `--num_splits` to a suitable value to get a faster process. To get the results (supposing `num_splits = 8`):

```shell
# FB15k-237
$ python evalCapsE.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_index 30 --model_name fb15k237 --num_splits 8 --decode

# WN18RR
$ python evalCapsE.py --embedding_dim 100 --num_filters 400 --name WN18RR --model_index 30 --model_name wn18rr --num_splits 8 --decode
```

### New Evaluation:

```shell
# FB15k-237
$ python eval_new.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_index 30 --model_name fb15k237 --num_splits 8 --decode

# WN18RR
$ python eval_new.py --embedding_dim 100 --num_filters 400 --name WN18RR --model_index 30 --model_name wn18rr --num_splits 8 --decode
```

`eval_type`  indicates evaluation protocol to use. It can take the values: `top`, `bottom` or `random`.