from model.tools import *
from model.GPKG_conv_upper import GPKGConvUpper
from model.GPKG_conv_base import GPKGConvBase
from model.Attention_conv import Attention_conv
from model.SpecialSpmmFinal import SpecialSpmmFinal

class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)

class GPKG_EMBEDD(BaseModel):
	def __init__(self, edge_index_base, edge_type_base, edge_index_upper, edge_type_upper, num_rel, params=None):
		super(GPKG_EMBEDD, self).__init__(params)

		self.edge_index_base = edge_index_base
		self.edge_type_base = edge_type_base
		self.edge_index_upper = edge_index_upper
		self.edge_type_upper = edge_type_upper

		self.p.gcn_dim		= self.p.gcn_dim
		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
		#self.device		= self.edge_index.device

		if self.p.num_bases > 0:
			self.init_rel  = get_param((self.p.num_bases,   self.p.init_dim))
		else:
			self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if self.p.num_bases > 0:
			self.conv_base_1 = GPKGConvBase(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
			self.conv_base_2 = GPKGConvUpper(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) 
			self.conv_upper_1 = Attention_conv(self.edge_index_upper, self.edge_type_upper, self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act, params=self.p, head_num=1)
			self.conv_upper_2 = Attention_conv(self.edge_index_upper, self.edge_type_upper, self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act, params=self.p, head_num=1) 

		else:
			self.conv_base_1 = GPKGConvUpper(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			self.conv_base_2 = GPKGConvUpper(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) 
			self.conv_upper_1 = Attention_conv(self.edge_index_upper, self.edge_type_upper, self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act, params=self.p, head_num=1)
			self.conv_upper_2 = Attention_conv(self.edge_index_upper, self.edge_type_upper, self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act, params=self.p, head_num=1) 


		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward_embedd(self, label, sub, rel, drop_base1, drop_base2, drop_upper1, drop_upper2):

		if label == 'base' :
			r	= self.init_rel 
			x, r	= self.conv_base_1(self.init_embed, self.edge_index_base, self.edge_type_base, rel_embed=r)
			x	= drop_base1(x)
			x, r	= self.conv_base_2(x, self.edge_index_base, self.edge_type_base, rel_embed=r)
			x	= drop_base2(x) 	


		if label == 'upper' :
			r	= self.init_rel 
			x, r	= self.conv_base_1(self.init_embed, self.edge_index_base, self.edge_type_base, rel_embed=r)
			x	= drop_base1(x)
			x, r	= self.conv_base_2(x, self.edge_index_base, self.edge_type_base, rel_embed=r)
			x	= drop_base2(x) 	

			x, r	= self.conv_upper_1(x, rel_embed=r) 	
			x	= drop_upper1(x) 	
			x, r	= self.conv_upper_2(x, rel_embed=r) 	
			x	= drop_upper2(x) 							

		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x

class GPKG_PREDICT(GPKG_EMBEDD):
	def __init__(self, edge_index_base, edge_type_base, edge_index_upper, edge_type_upper, chequer_perm, params=None):
		super(self.__class__, self).__init__(edge_index_base, edge_type_base, edge_index_upper, edge_type_upper, params.num_rel, params)

		self.inp_drop		= torch.nn.Dropout(self.p.inp_drop)
		self.hidden_drop_base	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop_upper	= torch.nn.Dropout(self.p.hid_drop)

		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)

		self.feature_map_drop	= torch.nn.Dropout2d(self.p.feat_drop)
		self.feature_map_drop_base	= torch.nn.Dropout2d(self.p.feat_drop)
		self.feature_map_drop_upper	= torch.nn.Dropout2d(self.p.feat_drop)
		self.bn0		= torch.nn.BatchNorm2d(self.p.perm)

		flat_sz_h 		= self.p.k_h
		flat_sz_w 		= 2*self.p.k_w
		self.padding 		= 0

		self.bn1 		= torch.nn.BatchNorm2d(self.p.num_filt*self.p.perm)
		self.flat_sz 		= flat_sz_h * flat_sz_w * self.p.num_filt*self.p.perm

		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		self.fc 		= torch.nn.Linear(self.flat_sz, self.p.embed_dim)
		self.chequer_perm	= chequer_perm

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
		self.register_parameter('conv_filt', Parameter(torch.zeros(self.p.num_filt, 1, self.p.ker_sz,  self.p.ker_sz))); xavier_normal_(self.conv_filt)

	def loss(self, pred, true_label=None, sub_samp=None):
		label_pos	= true_label[0]; 
		label_neg	= true_label[1:]
		loss 		= self.bceloss(pred, true_label)
		return loss

	def circular_padding_chw(self, batch, padding):
		upper_pad	= batch[..., -padding:, :]
		lower_pad	= batch[..., :padding, :]
		temp		= torch.cat([upper_pad, batch, lower_pad], dim=2)

		left_pad	= temp[..., -padding:]
		right_pad	= temp[..., :padding]
		padded		= torch.cat([left_pad, temp, right_pad], dim=3)
		return padded

	def forward(self, label, sub, rel, neg_ents):
		sub_emb, rel_emb, all_ent	= self.forward_embedd(label, sub, rel, self.hidden_drop_base, self.feature_map_drop_base, self.hidden_drop_upper, self.feature_map_drop_upper)
				
		comb_emb	= torch.cat([sub_emb, rel_emb], dim=1)
		chequer_perm	= comb_emb[:, self.chequer_perm]
		stack_inp	= chequer_perm.reshape((-1, self.p.perm, 2*self.p.k_w, self.p.k_h))

		x = stack_inp	= self.bn0(stack_inp)
		x		= self.circular_padding_chw(x, self.p.ker_sz//2)
		x		= F.conv2d(x, self.conv_filt.repeat(self.p.perm, 1, 1, 1), padding=self.padding, groups=self.p.perm)
		x		= self.bn1(x)
		x		= F.relu(x)
		x		= self.feature_map_drop(x)
		x		= x.view(-1, self.flat_sz)
		x		= self.fc(x)
		x		= self.hidden_drop2(x)
		x		= self.bn2(x)
		x		= F.relu(x)

		x = torch.mm(x, all_ent.transpose(1,0))
		x += self.bias.expand_as(x)

		pred	= torch.sigmoid(x)

		return pred
