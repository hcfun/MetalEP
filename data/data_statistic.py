import os
import random
import numpy as np
from collections import defaultdict as ddict
from tqdm import tqdm
import pickle
import dgl
import torch
import copy


ds = 'icews05-15' # icews14 icews05-15 gdelt
data_gra = 'day' #month day
data_path = './{}/{}'.format(ds, data_gra)
load_data = pickle.load(open(os.path.join(data_path, '{}_ext.pkl'.format(ds)), 'rb'))
valid_num_new_ent = np.sum(np.array(load_data['valid']['ent_map_list']) == -1)
valid_num_new_rel = np.sum(np.array(load_data['valid']['rel_map_list']) == -1)
valid_num_new_time = np.sum(np.array(load_data['valid']['time_map_list']) == -1)
test_num_new_ent = np.sum(np.array(load_data['test']['ent_map_list']) == -1)
test_num_new_rel = np.sum(np.array(load_data['test']['rel_map_list']) == -1)
test_num_new_time = np.sum(np.array(load_data['test']['time_map_list']) == -1)


print('train:')
print(f"num_ent: {len(load_data['train']['ent2id'])}")
print(f"num_rel: {len(load_data['train']['rel2id'])}")
print(f"num_time: {len(load_data['train']['new_time2id'])}")
print(f"num_tri: {len(load_data['train']['triples'])}")

print('valid:')
print(f"num_ent: {len(load_data['valid']['ent2id'])}(new: {valid_num_new_ent}, {valid_num_new_ent/len(load_data['valid']['ent2id']):.2})")
print(f"num_rel: {len(load_data['valid']['rel2id'])}(new: {valid_num_new_rel}, {valid_num_new_rel/len(load_data['valid']['rel2id']):.2})")
print(f"num_time: {len(load_data['valid']['new_time2id'])}(new: {valid_num_new_time}, {valid_num_new_time/len(load_data['valid']['new_time2id']):.2})")
print(f"num_sup: {len(load_data['valid']['support'])}")
print(f"num_que: {len(load_data['valid']['query'])}")

print('test:')
print(f"num_ent: {len(load_data['test']['ent2id'])}(new: {test_num_new_ent}, {test_num_new_ent/len(load_data['test']['ent2id']):.2})")
print(f"num_rel: {len(load_data['test']['rel2id'])}(new: {test_num_new_rel}, {test_num_new_rel/len(load_data['test']['rel2id']):.2})")
print(f"num_time: {len(load_data['test']['new_time2id'])}(new: {test_num_new_time}, {test_num_new_time/len(load_data['test']['new_time2id']):.2})")
print(f"num_sup: {len(load_data['test']['support'])}")
print(f"num_que_uent: {len(load_data['test']['query_uent'])}")
print(f"num_que_urel: {len(load_data['test']['query_urel'])}")
print(f"num_que_utime: {len(load_data['test']['query_utime'])}")
print(f"num_que_uentrel: {len(load_data['test']['query_uentrel'])}")
print(f"num_que_uenttime: {len(load_data['test']['query_uenttime'])}")
print(f"num_que_ureltime: {len(load_data['test']['query_ureltime'])}")
print(f"num_que_uall: {len(load_data['test']['query_uall'])}")