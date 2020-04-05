python CapsE.py --embedding_dim 100 --num_epochs 31 --num_filters 50 --learning_rate 0.0001 --name FB15k-237 --useConstantInit --savedEpochs 30 --model_name fb15k237

nohup python evalCapsE.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_index 30 --model_name fb15k237 --num_splits 8 --testIdx 0 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_index 30 --model_name fb15k237 --num_splits 8 --testIdx 1 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_index 30 --model_name fb15k237 --num_splits 8 --testIdx 2 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_index 30 --model_name fb15k237 --num_splits 8 --testIdx 3 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_index 30 --model_name fb15k237 --num_splits 8 --testIdx 4 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_index 30 --model_name fb15k237 --num_splits 8 --testIdx 5 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_index 30 --model_name fb15k237 --num_splits 8 --testIdx 6 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 50 --name FB15k-237 --useConstantInit --model_index 30 --model_name fb15k237 --num_splits 8 --testIdx 7 &

python CapsE.py --embedding_dim 100 --num_epochs 31 --num_filters 400 --learning_rate 0.00001 --name WN18RR --savedEpochs 30 --model_name wn18rr

nohup python evalCapsE.py --embedding_dim 100 --num_filters 400 --name WN18RR --model_index 30 --model_name wn18rr --num_splits 8 --testIdx 0 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 400 --name WN18RR --model_index 30 --model_name wn18rr --num_splits 8 --testIdx 1 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 400 --name WN18RR --model_index 30 --model_name wn18rr --num_splits 8 --testIdx 2 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 400 --name WN18RR --model_index 30 --model_name wn18rr --num_splits 8 --testIdx 3 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 400 --name WN18RR --model_index 30 --model_name wn18rr --num_splits 8 --testIdx 4 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 400 --name WN18RR --model_index 30 --model_name wn18rr --num_splits 8 --testIdx 5 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 400 --name WN18RR --model_index 30 --model_name wn18rr --num_splits 8 --testIdx 6 &
nohup python evalCapsE.py --embedding_dim 100 --num_filters 400 --name WN18RR --model_index 30 --model_name wn18rr --num_splits 8 --testIdx 7 &
