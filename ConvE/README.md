# Re-Evaluating: ConvE


### Training and Evaluation

```shell
# FB15k-237
$ python conve.py --name reprod_fb15k_237 --data FB15k-237 --gpu 0 --eval_type random

# WN18RR
$ python conve.py --name reprod_fb15k_237 --data FB15k-237 --gpu 0 --eval_type random
```

`eval_type`  indicates evaluation protocol to use. It can take the values: `top`, `bottom` or `random`.