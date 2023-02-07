import math
import sys
import argparse
import torch
import numpy as np
from utils.data_processing import get_data
from utils.utils import get_data_settings

def update_VR(sources_batch, destinations_batch, V, R, P):
	for idx in range(sources_batch.shape[0]):
		u, v = sources_batch[idx], destinations_batch[idx]
		Rprime = R.clone()
		for i in V[u].nonzero():
			R[i, v, :] = (P @ Rprime[i, u, :].T).T + Rprime[i, v, :]
		for i in V[v].nonzero():
			R[i, u] = (P @ Rprime[i, v, :].T).T + Rprime[i, u, :]
		V[u, :] = V[u, :] + V[v, :] - V[u, :] * V[v, :]
		V[v, :] = V[u, :]
	return V, R


torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser('PINT - positional features')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)', default='uci')
parser.add_argument('-ds', '--data_split', type=str, help='train, test_ind, test_trans, val_ind, val_trans, join', default='train')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--r_dim', type=int, default=4, help='dim for R')

try:
	args = parser.parse_args()
except:
	parser.print_help()
	sys.exit(0)

BATCH_SIZE = args.bs
GPU = args.gpu
DATA = args.data
SPLIT = args.data_split

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
	new_node_test_data = get_data(DATA)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

num_instance = len(train_data.sources)
num_batch = math.ceil(num_instance / BATCH_SIZE)

nextV, nextR = [], []

partition_size, last = get_data_settings(args.data)

r_dim = args.r_dim
R = torch.zeros((node_features.shape[0], node_features.shape[0], r_dim), requires_grad=False)
P = torch.zeros((r_dim, r_dim), requires_grad=False)
P[1:, :-1] = torch.eye(r_dim - 1, requires_grad=False)
for i in range(node_features.shape[0]):
	R[i, i, 0] = 1.0
V = torch.eye(node_features.shape[0], requires_grad=False)

if SPLIT == 'train':
	for k in range(0, num_batch):
		batch_idx = k

		start_idx = batch_idx * BATCH_SIZE
		end_idx = min(num_instance, start_idx + BATCH_SIZE)
		sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
			train_data.destinations[start_idx:end_idx]
		edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
		timestamps_batch = train_data.timestamps[start_idx:end_idx]

		prevV, prevR = V.clone(), R.clone()
		V, R = update_VR(sources_batch, destinations_batch, V, R, P)

		nextV.append((V - prevV).to('cpu').to_sparse())
		nextR.append((R - prevR).to('cpu').to_sparse())
		if ((k + 1) % partition_size == 0) or ((k + 1) == num_batch):  # savepoint
			prt = k // partition_size
			torch.save([nextV, nextR], 'pos_features/' + args.data + '_nextVR_part_' + str(prt) + '_bs_' + str(args.bs) + '_rdim_' + str(args.r_dim))
			nextV, nextR = [], []
else:
	nV, nR = torch.load('pos_features/' + args.data + '_nextVR_part_' + str(last) + '_bs_' + str(args.bs) + '_rdim_'+ str(args.r_dim))
	V, R = nV[-1].to_dense().clone(), nR[-1].to_dense().clone()  # save state at end of training
	TEST_BATCH_SIZE = args.bs
	if SPLIT == 'test_ind':
		num_test_instance = len(new_node_test_data.sources)
		num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

		ind_test_V, ind_test_R = [], []

		for k in range(num_test_batch):
			prevV, prevR = V.clone(), R.clone()
			s_idx = k * TEST_BATCH_SIZE
			e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
			sources_batch = new_node_test_data.sources[s_idx:e_idx]
			destinations_batch = new_node_test_data.destinations[s_idx:e_idx]

			V, R = update_VR(sources_batch, destinations_batch, V, R, P)

			ind_test_V.append((V - prevV).to('cpu').to_sparse())
			ind_test_R.append((R - prevR).to('cpu').to_sparse())
		torch.save([ind_test_V, ind_test_R], 'pos_features/' + args.data + '_VR_test_bs_' + str(args.bs) + '_rdim_' + str(args.r_dim) + '_inductive')
	elif SPLIT == 'test_trans':
		num_test_instance = len(test_data.sources)
		num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

		test_V, test_R = [], []

		for k in range(num_test_batch):
			prevV, prevR = V.clone(), R.clone()
			s_idx = k * TEST_BATCH_SIZE
			e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
			sources_batch = test_data.sources[s_idx:e_idx]
			destinations_batch = test_data.destinations[s_idx:e_idx]

			V, R = update_VR(sources_batch, destinations_batch, V, R, P)

			test_V.append((V - prevV).to('cpu').to_sparse())
			test_R.append((R - prevR).to('cpu').to_sparse())

		torch.save([test_V, test_R], 'pos_features/' + args.data + '_VR_test_bs_' + str(args.bs) + '_rdim_' + str(args.r_dim) + '_transductive')
	elif SPLIT == 'val_ind':
		num_test_instance = len(new_node_val_data.sources)
		num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

		ind_val_V, ind_val_R = [], []

		for k in range(num_test_batch):
			prevV, prevR = V.clone(), R.clone()
			s_idx = k * TEST_BATCH_SIZE
			e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
			sources_batch = new_node_val_data.sources[s_idx:e_idx]
			destinations_batch = new_node_val_data.destinations[s_idx:e_idx]

			V, R = update_VR(sources_batch, destinations_batch, V, R, P)

			ind_val_V.append((V - prevV).to('cpu').to_sparse())
			ind_val_R.append((R - prevR).to('cpu').to_sparse())

		# save validation stuff
		torch.save([ind_val_V, ind_val_R], 'pos_features/' + args.data + '_VR_val_bs_' + str(args.bs) + '_rdim_' + str(args.r_dim) + '_inductive')
	elif SPLIT == 'val_trans':
		num_test_instance = len(val_data.sources)
		num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

		val_V, val_R = [], []

		for k in range(num_test_batch):
			prevV, prevR = V.clone(), R.clone()
			s_idx = k * TEST_BATCH_SIZE
			e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
			sources_batch = val_data.sources[s_idx:e_idx]
			destinations_batch = val_data.destinations[s_idx:e_idx]

			V, R = update_VR(sources_batch, destinations_batch, V, R, P)

			val_V.append((V - prevV).to('cpu').to_sparse())
			val_R.append((R - prevR).to('cpu').to_sparse())

		torch.save([val_V, val_R], 'pos_features/' + args.data + '_VR_val_bs_' + str(args.bs) + '_rdim_' + str(args.r_dim) + '_transductive')
	else: # Join files
		ind_test_V, ind_test_R = torch.load('pos_features/' + args.data + '_VR_test_bs_' + str(args.bs) + '_rdim_' + str(args.r_dim) + '_inductive')
		test_V, test_R = torch.load('pos_features/' + args.data + '_VR_test_bs_' + str(args.bs) + '_rdim_' + str(args.r_dim) + '_transductive')
		val_V, val_R = torch.load('pos_features/' + args.data + '_VR_val_bs_' + str(args.bs) + '_rdim_' + str(args.r_dim) + '_transductive')
		ind_val_V, ind_val_R = torch.load('pos_features/' + args.data + '_VR_val_bs_' + str(args.bs) + '_rdim_' + str(args.r_dim) + '_inductive')
		torch.save([ind_val_V, ind_val_R, val_V, val_R], 'pos_features/' + args.data + '_VR_val_bs_' + str(args.bs) + '_rdim_' + str(args.r_dim))
		torch.save([ind_test_V, ind_test_R, test_V, test_R], 'pos_features/' + args.data + '_VR_test_bs_' + str(args.bs) + '_rdim_' + str(args.r_dim))

