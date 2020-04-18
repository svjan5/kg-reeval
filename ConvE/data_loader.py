from helper import *
from torch.utils.data import Dataset

class TrainDataset(Dataset):
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params
		self.entities	= np.arange(self.p.num_ent, dtype=np.int32)

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele			= self.triples[idx]
		triple, label, sub_samp	= torch.LongTensor(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
		trp_label		= self.get_label(label)

		if self.p.lbl_smooth != 0.0:
			trp_label = (1.0 - self.p.lbl_smooth)*trp_label + (1.0/self.p.num_ent)

		return triple, trp_label, None, None

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)

		return triple, trp_label
	
	def get_label(self, label):
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)


class TestDataset(Dataset):
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele		= self.triples[idx]
		triple, label	= torch.LongTensor(ele['triple']), np.int32(ele['label'])
		label		= self.get_label(label)

		return triple, label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		label		= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, label
	
	def get_label(self, label):
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)