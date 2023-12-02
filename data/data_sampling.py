import os
import random
import numpy as np
from collections import defaultdict as ddict
from tqdm import tqdm
import pickle
import dgl
import torch
import copy

dgl.seed(1314)
dgl.random.seed(1314)
random.seed(1314)
np.random.seed(1314)

ds = 'icews14' # icews05-15 icews14
data_gra = 'day'
"""
1.load dataset in different granularity (day, month, and year)
"""
data_path = './{}/{}'.format(ds, data_gra)

def read_dict_openke(dict_path):
    """
    Read entity / relation dict.
    Format: dict({id: entity / relation})
    """

    element_dict = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        element, id_ = line.strip().split('\t')
        element_dict[element] = int(id_)
    return element_dict

def read_data_openke(data_path):
    """
    Read train / valid / test data.
    """
    triples = set()
    for line in open(data_path, 'r'):
        head, relation, tail, time = line.strip().split('\t')
        h = int(head)
        r = int(relation)
        t = int(tail)
        T = int(time)
        triples.add((h, r, t, T))
    return list(triples)

#ele2id
entity_dict = read_dict_openke(os.path.join(data_path, 'entity2id.txt'))
relation_dict = read_dict_openke(os.path.join(data_path, 'relation2id.txt'))
time_dict = read_dict_openke(os.path.join(data_path, 'time2id.txt'))

# id2ele
entity_dict_inv = {v: k for k, v in entity_dict.items()}
relation_dict_inv = {v: k for k, v in relation_dict.items()}
time_dict_inv = {v: k for k, v in time_dict.items()}

#reading id facts
train_triples = read_data_openke(os.path.join(data_path, 'train2id.txt'))
valid_triples = read_data_openke(os.path.join(data_path, 'valid2id.txt'))
test_triples = read_data_openke(os.path.join(data_path, 'test2id.txt'))

# integrate train, valid, and test triples as one tensor
triples = train_triples + valid_triples + test_triples
triples = torch.tensor(triples) # shape:(num_edges,4)

"""
2.Sample test facts from all facts(a.md whole graph that is made up of train, valid, and test facts)
"""
g_undir = dgl.graph((torch.cat([triples[:, 0], triples[:, 2]]),
                     torch.cat([triples[:, 2], triples[:, 0]])))
g = dgl.graph((triples[:, 0], triples[:, 2]))
g.edata['rel'] = triples[:, 1]
g.edata['time'] = triples[:, 3]

num_root_ent = 100
rw_len = 15
new_ratio = 0.1 # The ratio of unseen entities and relations in test and valid facts

root_ent = np.random.choice(g_undir.num_nodes(), num_root_ent, replace=False)
random_ent = torch.unique(dgl.sampling.random_walk(g_undir, root_ent, length=rw_len)[0])
if -1 in random_ent:
    random_ent = random_ent[1:]

test_g = dgl.node_subgraph(g, random_ent)  # induce test triples from sampled entities
test_ent = test_g.ndata[dgl.NID]  # entity in test triples. original node IDs
test_rel = torch.unique(test_g.edata['rel'])  # relations in test triples
test_time = torch.unique(test_g.edata['time'])

#unseen entities and relations in test data (only appear in test facts)
test_new_ent = np.random.choice(test_ent, int(len(test_ent) * new_ratio), replace=False)  # entities that only appear in test triples
test_new_rel = np.random.choice(test_rel, int(len(test_rel) * new_ratio), replace=False)  # relations that only appear in test triples
test_new_time = np.random.choice(test_time, int(len(test_time) * new_ratio), replace=False)

##np.setdiff1d(a.md,b): return the unique elements in a.md but not in b
#test_g.edata[dgl.EID]: return the ids of the sampled edges in the original graph
test_remain_edge = np.setdiff1d(np.arange(g.num_edges()), test_g.edata[dgl.EID]) # ids of remain edges in the original graph
test_remain_g = dgl.edge_subgraph(g, test_remain_edge) # remain edges in the original graph

test_remain_tri = torch.stack([test_remain_g.ndata[dgl.NID][test_remain_g.edges()[0]],
                               test_remain_g.edata['rel'],
                               test_remain_g.ndata[dgl.NID][test_remain_g.edges()[1]],
                               test_remain_g.edata['time']]).T.tolist()

#delete the facts that contain unseen elements
test_remain_tri_delnew = []
for tri in tqdm(test_remain_tri):
    h, r, t, T = tri
    if h not in test_new_ent and t not in test_new_ent and r not in test_new_rel and T not in test_new_time:
        test_remain_tri_delnew.append(tri)


"""
3.Sample valid facts
"""
triples_new = torch.tensor(test_remain_tri_delnew)
g_undir = dgl.graph((torch.cat([triples_new[:, 0], triples_new[:, 2]]),
                     torch.cat([triples_new[:, 2], triples_new[:, 0]])))

g = dgl.graph((triples_new[:, 0], triples_new[:, 2]))
g.edata['rel'] = triples_new[:, 1]
g.edata['time'] = triples_new[:, 3]

root_ent = np.random.choice(g_undir.num_nodes(), num_root_ent, replace=False)
random_ent = torch.unique(dgl.sampling.random_walk(g_undir, root_ent, length=rw_len)[0])
if -1 in random_ent:
    random_ent = random_ent[1:]

valid_g = dgl.node_subgraph(g, random_ent)

valid_ent = valid_g.ndata[dgl.NID]
valid_rel = torch.unique(valid_g.edata['rel'])
valid_time = torch.unique(valid_g.edata['time'])

valid_new_ent = np.random.choice(valid_ent, int(len(valid_ent) * new_ratio), replace=False)
valid_new_rel = np.random.choice(valid_rel, int(len(valid_rel) * new_ratio), replace=False)
valid_new_time = np.random.choice(valid_time, int(len(valid_time) * new_ratio), replace=False)

valid_remain_edge = np.setdiff1d(np.arange(g.num_edges()), valid_g.edata[dgl.EID])
valid_remain_g = dgl.edge_subgraph(g, valid_remain_edge)

valid_remain_tri = torch.stack([valid_remain_g.ndata[dgl.NID][valid_remain_g.edges()[0]],
                                valid_remain_g.edata['rel'],
                                valid_remain_g.ndata[dgl.NID][valid_remain_g.edges()[1]],
                                valid_remain_g.edata['time']]).T.tolist()

valid_remain_tri_delnew = []
for tri in tqdm(valid_remain_tri):
    h, r, t, T = tri
    if h not in valid_new_ent and t not in valid_new_ent and r not in valid_new_rel and T not in valid_new_time:
        valid_remain_tri_delnew.append(tri)

"""
4.Sample train triples
"""
triples_new = torch.tensor(valid_remain_tri_delnew)
g_undir = dgl.graph((torch.cat([triples_new[:, 0], triples_new[:, 2]]),
                     torch.cat([triples_new[:, 2], triples_new[:, 0]])))

g = dgl.graph((triples_new[:, 0], triples_new[:, 2]))
g.edata['rel'] = triples_new[:, 1]
g.edata['time'] = triples_new[:, 3]

num_train_root_ent = 100
train_rw_len = 20

root_ent = np.random.choice(g_undir.num_nodes(), num_train_root_ent, replace=False)
random_ent = torch.unique(dgl.sampling.random_walk(g_undir, root_ent, length=train_rw_len)[0])
if -1 in random_ent:
    random_ent = random_ent[1:]

train_g = dgl.node_subgraph(g, random_ent)

"""
5.re-index triples in train/valid/test
re-index ent_id
"""
train_triples = torch.stack([train_g.ndata[dgl.NID][train_g.edges()[0]],
                               train_g.edata['rel'],
                               train_g.ndata[dgl.NID][train_g.edges()[1]],
                               train_g.edata['time']]).T.tolist()

test_triples = torch.stack([test_g.ndata[dgl.NID][test_g.edges()[0]],
                               test_g.edata['rel'],
                               test_g.ndata[dgl.NID][test_g.edges()[1]],
                               test_g.edata['time']]).T.tolist()

valid_triples = torch.stack([valid_g.ndata[dgl.NID][valid_g.edges()[0]],
                               valid_g.edata['rel'],
                               valid_g.ndata[dgl.NID][valid_g.edges()[1]],
                               valid_g.edata['time']]).T.tolist()

"""
6.re-index train triples, entities, relations, and times
"""
def reidx_train(triples):
    ent_reidx = dict()
    rel_reidx = dict()
    time_reidx = dict()

    entidx = 0
    relidx = 0
    timeidx  = 0

    reidx_triples = []
    for tri in triples:
        h, r, t, T = tri
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        if r not in rel_reidx.keys():
            rel_reidx[r] = relidx
            relidx += 1
        if T not in time_reidx.keys():
            time_reidx[T] = timeidx
            timeidx += 1

        reidx_triples.append((ent_reidx[h], rel_reidx[r], ent_reidx[t], time_reidx[T]))

    return reidx_triples, ent_reidx, rel_reidx, time_reidx

#new reidx_triples, and the id mapping(entities and relation in train facts) from original ids to new ids
train_triples, train_ent_reidx, train_rel_reidx, train_time_reidx = reidx_train(train_triples)
# new ent2id and rel2id |ent/rel/time_name mapping to new id
train_ent2id = {entity_dict_inv[k]: v for k, v in train_ent_reidx.items()}
train_rel2id = {relation_dict_inv[k]: v for k, v in train_rel_reidx.items()}
train_time2id = {time_dict_inv[k]: v for k, v in train_time_reidx.items()}

"""
7.re-index valid/test triples
"""
def reidx_eval(triples, train_ent_reidx, train_rel_reidx, train_time_reidx):

    ent_reidx = dict()
    rel_reidx = dict()
    time_reidx = dict()

    entidx = 0
    relidx = 0
    timeidx = 0

    ent_freq = ddict(int)
    rel_freq = ddict(int)
    time_freq = ddict(int)

    reidx_triples = []
    for tri in triples:
        h, r, t, T = tri
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        if r not in rel_reidx.keys():
            rel_reidx[r] = relidx
            relidx += 1
        if T not in time_reidx.keys():
            time_reidx[T] = timeidx
            timeidx += 1

        ent_freq[ent_reidx[h]] += 1
        ent_freq[ent_reidx[t]] += 1
        rel_freq[rel_reidx[r]] += 1
        time_freq[time_reidx[T]] += 1

        reidx_triples.append((ent_reidx[h], rel_reidx[r], ent_reidx[t], time_reidx[T]))

    #the id mapping(entities and relation in train facts) from new ids to original ones
    ent_reidx_inv = {v: k for k, v in ent_reidx.items()}
    rel_reidx_inv = {v: k for k, v in rel_reidx.items()}
    time_reidx_inv = {v: k for k, v in time_reidx.items()}

    # mapping the ent ids in valid/test into the ent ids in train (seen entities have a.md mapping and the unseen ones equal to '-1' )
    ent_map_list = [train_ent_reidx[ent_reidx_inv[i]] if ent_reidx_inv[i] in train_ent_reidx.keys() else -1
                    for i in range(len(ent_reidx))]
    # mapping the rel ids in valid/test into the rel ids in train (seen relations have a.md mapping and the unseen ones equal to '-1' )
    rel_map_list = [train_rel_reidx[rel_reidx_inv[i]] if rel_reidx_inv[i] in train_rel_reidx.keys() else -1
                    for i in range(len(rel_reidx))]
    time_map_list = [train_time_reidx[time_reidx_inv[i]] if time_reidx_inv[i] in train_time_reidx.keys() else -1
                    for i in range(len(time_reidx))]

    return reidx_triples, ent_freq, rel_freq, time_freq, ent_reidx, rel_reidx, time_reidx, ent_map_list, rel_map_list, time_map_list

valid_triples, valid_ent_freq, valid_rel_freq, valid_time_freq, valid_ent_reidx, valid_rel_reidx, valid_time_reidx,\
    valid_ent_map_list, valid_rel_map_list, valid_time_map_list = reidx_eval(valid_triples, train_ent_reidx, train_rel_reidx, train_time_reidx)
valid_ent2id = {entity_dict_inv[k]: v for k, v in valid_ent_reidx.items()}
valid_rel2id = {relation_dict_inv[k]: v for k, v in valid_rel_reidx.items()}
valid_time2id = {time_dict_inv[k]: v for k, v in valid_time_reidx.items()}

test_triples, test_ent_freq, test_rel_freq, test_time_freq, test_ent_reidx, test_rel_reidx, test_time_reidx,\
    test_ent_map_list, test_rel_map_list, test_time_map_list = reidx_eval(test_triples, train_ent_reidx, train_rel_reidx, train_time_reidx)
test_ent2id = {entity_dict_inv[k]: v for k, v in test_ent_reidx.items()}
test_rel2id = {relation_dict_inv[k]: v for k, v in test_rel_reidx.items()}
test_time2id = {time_dict_inv[k]: v for k, v in test_time_reidx.items()}

"""
8.Split triples in valid/test into support and query
"""
def split_triples(triples, ent_freq, rel_freq, time_freq, ent_map_list, rel_map_list, time_map_list):
    ent_freq = copy.deepcopy(ent_freq)
    rel_freq = copy.deepcopy(rel_freq)

    support_triples = []

    query_uent = []
    query_urel = []
    query_utime = []
    query_uentrel = []
    query_uenttime = []
    query_ureltime = []
    query_uall = []


    random.shuffle(triples)
    for idx, tri in enumerate(triples):
        h, r, t, T = tri
        test_flag = (ent_map_list[h] == -1 or ent_map_list[t] == -1 or rel_map_list[r] == -1 or time_map_list[T] == -1)

        if (ent_freq[h] > 2 and ent_freq[t] > 2 and rel_freq[r] > 2 and time_freq[T] > 2) and test_flag:
            append_flag = False
            # r not in train
            if ent_map_list[h] != -1 and ent_map_list[t] != -1 and time_map_list[T] != -1 and rel_map_list[r] == -1:
                if len(query_urel) <= int(len(triples) * 0.1):
                    query_urel.append(tri)
                    append_flag = True
            # time not in train
            elif ent_map_list[h] != -1 and ent_map_list[t] != -1 and rel_map_list[r] != -1 and time_map_list[T] == -1:
                if len(query_utime) <= int(len(triples) * 0.1):
                    query_utime.append(tri)
                    append_flag = True
            # h or t not in train
            elif (ent_map_list[h] == -1 or ent_map_list[t] == -1) and rel_map_list[r] != -1 and time_map_list[T] != -1:
                if len(query_uent) <= int(len(triples) * 0.1):
                    query_uent.append(tri)
                    append_flag = True
            elif (ent_map_list[h] == -1 or ent_map_list[t] == -1) and rel_map_list[r] == -1 and time_map_list[T] != -1:
                if len(query_uentrel) <= int(len(triples) * 0.1):
                    query_uentrel.append(tri)
                    append_flag = True
            elif (ent_map_list[h] == -1 or ent_map_list[t] == -1) and rel_map_list[r] != -1 and time_map_list[T] == -1:
                if len(query_uenttime) <= int(len(triples) * 0.1):
                    query_uenttime.append(tri)
                    append_flag = True
            elif ent_map_list[h] != -1 and ent_map_list[t] != -1 and time_map_list[T] == -1 and rel_map_list[r] == -1:
                if len(query_ureltime) <= int(len(triples) * 0.1):
                    query_ureltime.append(tri)
                    append_flag = True

            # h,t or r,t not in train simultaneously
            elif (ent_map_list[h] == -1 or ent_map_list[t] == -1) and rel_map_list[r] == -1 and time_map_list[T] == -1:
                if len(query_uall) <= int(len(triples) * 0.1):
                    query_uall.append(tri)
                    append_flag = True

            if append_flag:
                ent_freq[h] -= 1
                ent_freq[t] -= 1
                rel_freq[r] -= 1
            # the number of triples in query is enough
            else:
                support_triples.append(tri)
        else:
            support_triples.append(tri)

    return support_triples, query_uent, query_urel, query_utime, query_uentrel, query_uenttime, query_ureltime, query_uall

valid_sup_tris, valid_que_uent, valid_que_urel, valid_que_utime, valid_que_uentrel, valid_que_uenttime, valid_que_ureltime, valid_que_uall = split_triples(valid_triples,
                                                                                valid_ent_freq, valid_rel_freq, valid_time_freq,
                                                                                valid_ent_map_list, valid_rel_map_list, valid_time_map_list)

test_sup_tris, test_que_uent, test_que_urel, test_que_utime, test_que_uentrel, test_que_uenttime, test_que_ureltime, test_que_uall = split_triples(test_triples,
                                                                            test_ent_freq, test_rel_freq, test_time_freq,
                                                                            test_ent_map_list, test_rel_map_list, test_time_map_list)
train_time_reidx_inv = {v: k for k, v in train_time_reidx.items()}
valid_time_reidx_inv = {v: k for k, v in valid_time_reidx.items()}
test_time_reidx_inv = {v: k for k, v in test_time_reidx.items()}

def remain_ori_time(time_idx_inv, ori_triples):
    new_triples = []
    for idx, tri in enumerate(ori_triples):
        h, r, t, T = tri
        new_triples.append((h, r, t, time_idx_inv[T]))
    return new_triples

new_train_triples = remain_ori_time(train_time_reidx_inv, train_triples)

new_valid_sup_tris = remain_ori_time(valid_time_reidx_inv, valid_sup_tris)
new_valid_que_uent = remain_ori_time(valid_time_reidx_inv, valid_que_uent)
new_valid_que_urel = remain_ori_time(valid_time_reidx_inv, valid_que_urel)
new_valid_que_utime = remain_ori_time(valid_time_reidx_inv, valid_que_utime)
new_valid_que_uentrel = remain_ori_time(valid_time_reidx_inv, valid_que_uentrel)
new_valid_que_uenttime = remain_ori_time(valid_time_reidx_inv, valid_que_uenttime)
new_valid_que_ureltime = remain_ori_time(valid_time_reidx_inv, valid_que_ureltime)
new_valid_que_uall = remain_ori_time(valid_time_reidx_inv, valid_que_uall)

new_test_sup_tris = remain_ori_time(test_time_reidx_inv, test_sup_tris)
new_test_que_uent = remain_ori_time(test_time_reidx_inv, test_que_uent)
new_test_que_urel = remain_ori_time(test_time_reidx_inv, test_que_urel)
new_test_que_utime = remain_ori_time(test_time_reidx_inv, test_que_utime)
new_test_que_uentrel = remain_ori_time(test_time_reidx_inv, test_que_uentrel)
new_test_que_uenttime = remain_ori_time(test_time_reidx_inv, test_que_uenttime)
new_test_que_ureltime = remain_ori_time(test_time_reidx_inv, test_que_ureltime)
new_test_que_uall = remain_ori_time(test_time_reidx_inv, test_que_uall)


#ent_map_list new ent id in train, corresponding to test/valid
data_dict = {'train': {'triples': new_train_triples, 'ent2id': train_ent2id, 'rel2id': train_rel2id, 'time2id': time_dict, 'new_time2id': train_time2id},
             'valid': {'support': new_valid_sup_tris, 'query': new_valid_que_uent + new_valid_que_urel + new_valid_que_utime + \
                        new_valid_que_uentrel+new_valid_que_uenttime+ new_valid_que_ureltime+new_valid_que_uall,
                       'ent_map_list': valid_ent_map_list, 'rel_map_list': valid_rel_map_list, 'time_map_list': valid_time_map_list,
                       'ent2id': valid_ent2id, 'rel2id': valid_rel2id, 'time2id': time_dict, 'new_time2id': valid_time2id},
             'test': {'support': new_test_sup_tris, 'query_uent': new_test_que_uent,
                      'query_urel': new_test_que_urel, 'query_utime': new_test_que_utime, 'query_uentrel': new_test_que_uentrel,
                      'query_uenttime': new_test_que_uenttime, 'query_ureltime': new_test_que_ureltime, 'query_uall': new_test_que_uall,
                      'ent_map_list': test_ent_map_list, 'rel_map_list': test_rel_map_list, 'time_map_list': test_time_map_list,
                      'ent2id': test_ent2id, 'rel2id': test_rel2id, 'time2id': time_dict, 'new_time2id': test_time2id}}

pickle.dump(data_dict, open(os.path.join(data_path, '{}_ext.pkl'.format(ds)), 'wb'))




