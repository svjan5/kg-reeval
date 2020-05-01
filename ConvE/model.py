from helper import *

class BaseModel(torch.nn.Module):
	def __init__(self,  params):
		super(BaseModel, self).__init__()
		self.p		= params
		self.ent_embed	= torch.nn.Embedding(self.p.num_ent,   self.p.embed_dim, padding_idx=None); xavier_normal_(self.ent_embed.weight)
		self.rel_embed	= torch.nn.Embedding(self.p.num_rel*2, self.p.embed_dim, padding_idx=None); xavier_normal_(self.rel_embed.weight)
		self.bceloss	= torch.nn.BCELoss()

	def concat(self, e1_embed, rel_embed, form='plain'):
		if form == 'plain':
			e1_embed	= e1_embed. view(-1, 1, self.p.k_w, self.p.k_h)
			rel_embed	= rel_embed.view(-1, 1, self.p.k_w, self.p.k_h)
			stack_inp	= torch.cat([e1_embed, rel_embed], 2)

		elif form == 'alternate':
			e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
			rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
			stack_inp	= torch.cat([e1_embed, rel_embed], 1)
			stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))

		else: raise NotImplementedError
		return stack_inp

	def loss(self, pred, true_label=None):
		pos_scr		= pred[:,0]; 	  
		neg_scr		= pred[:,1:]
		label_pos	= true_label[0]; 
		label_neg	= true_label[1:]
		loss		= self.bceloss(pred, true_label)
		return loss

class ConvE(BaseModel):
	def __init__(self,  params):
		super(ConvE, self).__init__(params)
		self.input_drop		= torch.nn.Dropout(self.p.inp_drop)
		self.feature_drop	= torch.nn.Dropout2d(self.p.feat_drop)
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)

		self.conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)
		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward(self, sub, rel):
		sub_emb	= self.ent_embed(sub)
		rel_emb	= self.rel_embed(rel)
		stk_inp	= self.concat(sub_emb, rel_emb, self.p.form)
		x	= self.bn0(stk_inp)
		x	= self.input_drop(x)
		x	= self.conv1(x)
		x	= self.bn1(x)
		x	= F.relu(x)
		x	= self.feature_drop(x)
		x	= x.view(-1, self.flat_sz)
		x	= self.fc(x)
		x	= self.hidden_drop(x)
		x	= self.bn2(x)
		x	= F.relu(x)
		
		x = torch.mm(x, self.ent_embed.weight.transpose(1,0))
		x += self.bias.expand_as(x)

		pred	= torch.sigmoid(x)

		return pred