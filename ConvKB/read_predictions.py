import os, sys, pdb, numpy as np, random, argparse, codecs, pickle, time, json
from pprint import pprint
from collections import defaultdict as ddict

parser = argparse.ArgumentParser(description='')
parser.add_argument('--num_splits', default=7, type=int, help='Number of splits used during evaluation')
parser.add_argument('--model_name', default='fb15k237', help='Name of the model')
parser.add_argument("--run_folder", default="../", help="Data sources.")
parser.add_argument("--eval_type", default='rotate', help="")
parser.add_argument("--model_index", default='200', help="")

args = parser.parse_args()

out_dir = os.path.abspath(os.path.join(args.run_folder, "runs", args.model_name))
print("Evaluating {}\n".format(out_dir))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")

_file   = checkpoint_prefix + "-" + args.model_index
all_res = []
for i in range(args.num_splits):
	all_res.extend(pickle.load(open(_file + '.eval_{}.'.format(args.eval_type) + str(i) + '.pkl', 'rb')))

min_len = np.min([len(x['results']) for x in all_res])
results = np.float32([x['results'][:min_len] for x in all_res])
import pdb; pdb.set_trace()

# python read_predictions.py --model_name fb15k237_seed2 --num_splits 7 --eval_type random