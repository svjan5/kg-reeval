from helper import *
from data_loader  import *
from model 	  import *


class Main(object):

	def load_data(self):

		ent_set, rel_set = OrderedSet(), OrderedSet()
		for split in ['train', 'test', 'valid']:
			for line in open('../data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)

		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
		self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

		self.data = ddict(list)
		sr2o = ddict(set)

		for split in ['train', 'test', 'valid']:
			for line in open('../data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
				self.data[split].append((sub, rel, obj))

				if split == 'train': 
					sr2o[(sub, rel)].add(obj)
					sr2o[(obj, rel+self.p.num_rel)].add(sub)
		self.data = dict(self.data)

		self.sr2o = {k: list(v) for k, v in sr2o.items()}
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, rel+self.p.num_rel)].add(sub)

		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
		self.triples  = ddict(list)

		for (sub, rel), obj in self.sr2o.items():
			self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				rel_inv = rel + self.p.num_rel
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		self.triples = dict(self.triples)

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		self.data_iter = {
			'train'		:   get_data_loader(TrainDataset, 'train', 	self.p.batch_size),
			'valid_head'	:   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			'valid_tail'	:   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_head'	:   get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail'	:   get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}

	def __init__(self, params):
		self.p = params
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data()
		self.model        = self.add_model()
		self.optimizer    = self.add_optimizer(self.model.parameters())


	def add_model(self):
		model = ConvE(self.p)
		model.to(self.device)
		return model

	def add_optimizer(self, parameters):
		if self.p.opt == 'adam' : return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)
		else                    : return torch.optim.SGD(parameters,  lr=self.p.lr, weight_decay=self.p.l2)

	def read_batch(self, batch, split):
		if split == 'train':
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def save_model(self, save_path):
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_val'	: self.best_val,
			'best_test'	: self.best_test,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		torch.save(state, save_path)

	def load_model(self, load_path):
		state				= torch.load(load_path)
		state_dict			= state['state_dict']
		self.best_val_mrr 		= state['best_val']['mrr']
		self.best_test_mrr 		= state['best_test']['mrr']
		self.best_test 			= state['best_test']
		self.best_val 			= state['best_val']

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])

	def evaluate(self, split, epoch):
		left_results, left_scores   = self.predict(split=split, mode='tail_batch')
		right_results, right_scores = self.predict(split=split, mode='head_batch')

		# pickle.dump(left_scores + right_scores, open('neuron_cnt.pkl', 'wb'))
		results       = get_combined_results(left_results, right_results)
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
		return results

	def predict(self, split='valid', mode='tail_batch'):
		self.model.eval()

		temp = []

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)
				pred, zero_cnt		= self.model.forward(sub, rel, None, zero_cnt=True)
				temp.append(zero_cnt)

				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				pred 			= torch.where(label.byte(), torch.zeros_like(pred), pred)	# Filtering
				pred[b_range, obj] 	= target_pred

				pred = pred.cpu().numpy()
				obj  = obj.cpu().numpy()

				for i in range(pred.shape[0]):
					# import pdb; pdb.set_trace()
					scores	= pred[i]
					target	= obj[i]
					tar_scr	= scores[target]
					scores  = np.delete(scores, target)

					if self.p.eval_type == 'top':
						scores = np.insert(scores, 0, tar_scr)

					elif self.p.eval_type == 'bottom':
						scores = np.concatenate([scores, [tar_scr]], 0)

					elif self.p.eval_type == 'random':
						rand = np.random.randint(scores.shape[0])
						scores = np.insert(scores, rand, tar_scr)

					else:
						raise NotImplementedError


					sorted_indices = np.argsort(-scores, kind='stable')

					if 	self.p.eval_type == 'top':	_filter = np.where(sorted_indices == 0)[0][0]
					elif 	self.p.eval_type == 'bottom':	_filter = np.where(sorted_indices == scores.shape[0]-1)[0][0]
					elif 	self.p.eval_type == 'random': 	_filter = np.where(sorted_indices == rand)[0][0]
					else:	raise NotImplementedError


					results['count']	= 1 + results.get('count', 0.0)
					results['mr']		= (_filter + 1) + results.get('mr', 0.0)
					results['mrr']		= (1.0 / (_filter + 1)) + results.get('mrr', 0.0)

					for k in range(10):
						if _filter <= k: 
							results['hits@{}'.format(k+1)] = 1 + results.get('hits@{}'.format(k+1), 0.0)

				if step % 100 == 0:
					self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

		return results, temp

	def run_epoch(self, epoch, val_mrr = 0):
		self.model.train()
		losses = []
		train_iter = iter(self.data_iter['train'])

		for step, batch in enumerate(train_iter):
			self.optimizer.zero_grad()

			sub, rel, obj, label = self.read_batch(batch, 'train')

			pred	= self.model.forward(sub, rel)
			loss	= self.model.loss(pred, label)

			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())

			if step % 100 == 0:
				self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

		loss = np.mean(losses)
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
		return loss

	def fit(self):
		self.best_val_mrr, self.best_val, self.best_test, self.best_test_mrr, self.best_epoch = 0., {}, {}, 0., 0.
		val_mrr = 0
		save_path = os.path.join('./models', self.p.name)

		if self.p.restore:
			self.load_model(save_path)
			self.logger.info('Successfully Loaded previous model')
			test_results = self.evaluate('test', -1)
			pprint(test_results)

		kill_cnt = 0
		for epoch in range(self.p.max_epochs):
			train_loss = self.run_epoch(epoch, val_mrr)

			val_results = self.evaluate('valid', epoch)

			if val_results['mrr'] > self.best_val_mrr:
				test_results		= self.evaluate('test', epoch)
				self.best_val		= val_results
				self.best_val_mrr	= val_results['mrr']
				self.best_test		= test_results
				self.best_test_mrr	= self.best_test['mrr']
				self.best_epoch		= epoch
				kill_cnt = 0
			else:
				kill_cnt += 1
				if kill_cnt > 20: exit(0)

			self.logger.info('[Epoch {}]:  Training Loss: {:.5},  Valid MRR: {:.5},  Test MRR: {:.5}\n\n\n'.format(epoch, train_loss, self.best_val_mrr, self.best_test_mrr))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('-data',            dest="dataset",         default='FB15k-237',            help='Dataset to use')
	parser.add_argument("-name",            default='testrun_'+str(uuid.uuid4())[:8],		help="Set filename for saving or restoring models")

	parser.add_argument('-batch',           dest="batch_size",      default=128,    type=int,       help='Batch size')

	parser.add_argument("-gpu",		type=str,               default='0',			help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0, For Multiple GPU = 0,1")
	parser.add_argument("-l2",		type=float,             default=0.0,			help="L2 Regularization for Optimizer")
	parser.add_argument("-lr",		type=float,             default=0.001,			help="For Standard/Adaptive: Starting Learning Rate, CLR : Minimum LR")
	parser.add_argument('-loss',            dest="loss",		default='bce',			help='GPU to use')

	parser.add_argument("-lbl_smooth",      dest='lbl_smooth',	type=float,    default=0.1,	help="Label Smoothing enable or disable, reqd True")
	parser.add_argument('-no_rev',         	dest="reverse",         action='store_false',           help='Use uniform weighting')
	parser.add_argument('-nosave',       	dest="save",       	action='store_false',           help='Negative adversarial sampling')

	parser.add_argument("-epoch",		dest='max_epochs', 	type=int,         default=300,  help="Number of epochs")
	parser.add_argument("-num_workers",	type=int,               default=5,                      help="For CPU:0, For GPU Serial:1, For GPU PS and COLLECTIVE_ALL_REDUCE: 1+")
	parser.add_argument("-embed_dim",	type=int,              	default=None,                   help="For CPU:0, For GPU Serial:1, For GPU PS and COLLECTIVE_ALL_REDUCE: 1+")

	parser.add_argument('-opt',             dest="opt",             default='adam',                 help='GPU to use')
	parser.add_argument('-restore',         dest="restore",         action='store_true',            help='Restore from the previously saved model')
	parser.add_argument('-seed',            dest="seed",            default=42,   type=int,       	help='Seed for randomization')
	parser.add_argument('-bias',            dest="bias",            action='store_true',            help='Restore from the previously saved model')

	parser.add_argument('-eval_type',       dest="eval_type",       default='random',           	help='Evaluation Protocol to use. Options: top/bottom/random')

	parser.add_argument('-form',		type=str,               default='alternate',            help='Input concat form')
	parser.add_argument('-k_w',	  	dest="k_w", 		default=10,   	type=int, 	help='k_w')
	parser.add_argument('-k_h',	  	dest="k_h", 		default=20,   	type=int, 	help='k_h')

	parser.add_argument('-inp_drop',  	dest="inp_drop", 	default=0.2,  	type=float,	help='Dropout for full connected layer')
	parser.add_argument('-hid_drop',  	dest="hid_drop", 	default=0.3,  	type=float,	help='Dropout for full connected layer')
	parser.add_argument('-feat_drop', 	dest="feat_drop", 	default=0.3,  	type=float,	help='Dropout for full connected layer')

	parser.add_argument('-num_filt',  	dest="num_filt", 	default=32,   	type=int, 	help='Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest="ker_sz", 		default=3,   	type=int, 	help='ker_sz')

	parser.add_argument('-logdir',          dest="log_dir",         default='./log/',               help='Log directory')
	parser.add_argument('-config',          dest="config_dir",      default='./config/',           help='Config directory')

	args = parser.parse_args()

	if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

	set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	model = Main(args)
	model.fit()