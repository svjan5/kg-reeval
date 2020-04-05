# Re-Evaluating: ConvKB

### Requirements
- Python 3.x and Tensorflow >= 1.6

### Training
```shell
 # FB15k-237
 $ python train.py --embedding_dim 100 --num_filters 50 --learning_rate 0.000005 --name FB15k-237 --useConstantInit --model_name fb15k237
 
 # WN18-RR
 $ python train.py --embedding_dim 50 --num_filters 500 --learning_rate 0.0001 --name WN18RR --model_name wn18rr --saveStep 50
```

### Original Evaluation:

```shell
# FB15k-237
$ python train.py --embedding_dim 100 --num_filters 50 --learning_rate 0.000005 --name FB15k-237 --useConstantInit --model_name fb15k237

# WN18RR
$ python train.py --embedding_dim 50 --num_filters 500 --learning_rate 0.0001 --name WN18RR --model_name wn18rr --saveStep 50
```

### New Evaluation:

```shell
# FB15k-237
$ python eval_new.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_name fb15k237 --num_splits 8 --decode --eval_type random
    
# WN18RR
$ python eval_new.py --embedding_dim 50 --num_filters 500 --name WN18RR --model_name wn18rr --num_splits 8 --decode --eval_type random
```

`eval_type`  indicates evaluation protocol to use. It can take the values: `top`, `bottom` or `random`.