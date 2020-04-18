# Re-Evaluating: RotatE


### Training

```shell
# FB15k-237
$ CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/FB15k \
 --model RotatE \
 -n 256 -b 1024 -d 1000 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0001 --max_steps 150000 \
 -save models/RotatE_FB15k_0 --test_batch_size 16 -de

# WN18RR
$ CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/wn18rr \
 --model RotatE \
 -n 512 -b 1024 -d 500 \
 -g 6.0 -a 0.6 -adv \
 -lr 0.00005 --max_steps 80000 \
 -save models/RotatE_FB15k_0 --test_batch_size 8 -de

```

### Original Evaluation:

```shell
$ CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u codes/run.py --do_test --cuda -init $SAVE
```

### New Evaluation:

```shell
$ CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u codes/run.py --do_test --cuda -init $SAVE --eval_type $EVAL_TYPE
```

`eval_type`  indicates evaluation protocol to use. It can take the values: `top`, `bottom` or `random`.