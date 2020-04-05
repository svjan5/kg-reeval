import os, sys, pdb, numpy as np, random, argparse, codecs, pickle, time, json
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
from scipy.stats import describe 
from copy import deepcopy

parser = argparse.ArgumentParser(description='')
parser.add_argument('--num_splits', default=32, type=int, help='Number of splits used during evaluation')
parser.add_argument('--model_name', default='fb15k237_seed4', help='Name of the model')
parser.add_argument("--run_folder", default="../", help="Data sources.")
parser.add_argument("--eval_type", default='org', help="")
parser.add_argument("--model_index", default='200', help="")

args = parser.parse_args()

out_dir = os.path.abspath(os.path.join(args.run_folder, "runs", args.model_name))
print("Evaluating {}\n".format(out_dir))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")

_file   = checkpoint_prefix + "-" + args.model_index
all_res_org  = []
all_res_rand = []
all_res_last = []

for i in range(args.num_splits):
	all_res_org.extend(pickle.load(open(_file + '.eval_{}.'.format('org') + str(i) + '.pkl', 'rb')))
	print('Completed {}, {}'.format(i, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))
	

# Generates all_res_last from all_res_org
for x in tqdm(all_res_org):
	x = deepcopy(x)
	rand_pos = x['rand_pos']
	results = x['results']
	score   = results[0]
	results = results[1:]
	results = np.insert(results, results.shape[0], score)

	all_res_last.append({
		'rand_pos': results.shape[0]-1,
		'results': results
	})

# Generates all_res_last from all_res_org
for x in tqdm(all_res_org):
	x = deepcopy(x)
	rand_pos = x['rand_pos']
	results = x['results']
	score   = results[0]
	results = results[1:]
	rand 	= np.random.randint(results.shape[0])
	results = np.insert(results, rand, score)

	all_res_rand.append({
		'rand_pos': rand,
		'results': results
	})

cnt = []
for i, org in enumerate(all_res_org):
	cnt.append(np.sum(org['results'] == org['results'][0]))	

mrr = []
for i, last in enumerate(all_res_last):
	rank = np.where(np.argsort(last['results'], kind='stable') == last['rand_pos'])[0][0] + 1
	mrr.append(1.0/rank)
print('LAST Score', np.mean(mrr))

mrr = []
for i, org in enumerate(all_res_org):
	rank = np.where(np.argsort(org['results'], kind='stable') == 0)[0][0] + 1
	mrr.append(1.0/rank)
print('Original Score', np.mean(mrr))

mrr = []
for i, rand in enumerate(all_res_rand):
	rank = np.where(np.argsort(rand['results'], kind='stable') == rand['rand_pos'])[0][0] + 1
	mrr.append(1.0/rank)
print('Random Position Score', np.mean(mrr))

import pdb; pdb.set_trace()

# mrr = []
# for i, org in enumerate(all_res_org):
# 	rank = np.sum(org['results'] <= org['results'][0])
# 	mrr.append(1.0/rank)
# print(np.mean(mrr))


for i in range(len(all_res_rand)):
	rand = np.sort(all_res_rand[i]['results'])
	org  = np.sort(all_res_org[i]['results'])
	
	if np.allclose(rand, org) is False:
		import pdb; pdb.set_trace()

import pdb; pdb.set_trace()
# python comp_prediction.py --model_name fb15k237_seed2 --num_splits 7 --eval_type random