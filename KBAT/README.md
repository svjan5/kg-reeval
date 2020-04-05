# Re-Evaluating: Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs
### Requirements
- Python 3.x and Pytorch 1.x

### Reproducing results

To reproduce the results published in the paper:      
When running for first time, run preparation script with:

```shell
$ sh prepare.sh
```

* **Wordnet**

    ```shell
    $ python3 main.py --get_2hop True
    ```

* **Freebase**

    ```shell
    $ python3 main.py --data ./data/FB15k-237/ --epochs_gat 3000 --epochs_conv 150 --weight_decay_gat 0.00001 --get_2hop True --partial_2hop True --batch_size_gat 272115 --margin 1 --out_channels 50 --drop_conv 0.3 --output_folder ./checkpoints/fb/out/
    ```

