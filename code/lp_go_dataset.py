import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(
    description="Knowledge Graph Completion"
)

parser.add_argument(
    '--directory', type=str, default='GO_lp/'
)

args = parser.parse_args()


trp = pd.read_csv('/home/ukjung18/GIE/GIE-master/src_data/GO0523/edges.tsv', delimiter='\t')
go_level = pd.read_csv('/home/ukjung18/GIE/GIE-master/src_data/GO0523/entities.tsv', delimiter='\t')


# In[4]:


trp = trp[['head', 'rel', 'tail']]
go_level = go_level[['term', 'level']]


# In[5]:


trp = trp.sample(frac=1).reset_index(drop=True) # shuffle


# In[6]:


num_rel = trp.shape[0]
train_idx = int(num_rel*0.8)
valid_idx = train_idx + (num_rel-train_idx)//2


# In[8]:


ent = set()
rel = set()

for i in range(num_rel):
    # if (trp.loc[i]['rel'] == 'biolink:subPropertyOf') or (trp.loc[i]['predicate'] == 'biolink:inverseOf'):
    #     continue
    ent.add(trp.loc[i]['head'])
    rel.add(trp.loc[i]['rel'])
    ent.add(trp.loc[i]['tail'])
ent_to_id = {x: i for (i, x) in enumerate(sorted(ent))}
go_level['ent']
rel_to_id = {x: i for (i, x) in enumerate(sorted(rel))}


# In[12]:

# 'biolink:has_part': 0,
# 'biolink:inverseOf': 1,
# 'biolink:occurs_in': 2,
# 'biolink:part_of': 3,
# 'biolink:regulates': 4,
# 'biolink:related_to': 5,
# 'biolink:subPropertyOf': 6,
# 'biolink:subclass_of': 7

# avoid disconnection
u = []
v = []
r = []
for i in range(0, num_rel):
    # if (trp.loc[i]['predicate'] == 'biolink:subPropertyOf') or (trp.loc[i]['predicate'] == 'biolink:inverseOf'):
    #     continue
    u.append(ent_to_id[trp.loc[i]['head']])
    v.append(ent_to_id[trp.loc[i]['rel']])
    r.append(rel_to_id[trp.loc[i]['tail']])


# In[13]:


n = len(ent)
dup_u = []
dup_r = []
dup_v = []
s = []
t = []
sp_mat = np.zeros((n, n))
rel_list = []
for i, j, k in zip(u, v, r):
    if sp_mat[i][j] == 1 or sp_mat[j][i] == 1:
        dup_u.append(i)
        dup_v.append(j)
        dup_r.append(k)
        continue
    s.append(i)
    t.append(j)
    sp_mat[i][j] = 1
    rel_list.append(k)


# In[14]:


from scipy.sparse import csr_array
row = np.array(s)
col = np.array(t)
data = np.array([1]*len(s))
csr_mat2 = csr_array((data, (row, col)), shape=(n, n))


# In[17]:


adj = csr_mat2


# In[18]:


import networkx as nx
g = nx.Graph(adj)
np.random.seed(0)
adj_sparse = nx.to_scipy_sparse_array(g)


# In[20]:


# Perform train-test split
from lp_preprocessing import mask_test_edges
adj_train, train_edges, val_edges, test_edges, \
test_edges_false, test_inv_false, \
train_rel, val_rel, test_rel = mask_test_edges(adj_sparse, s, t, rel_list, test_frac=.1, val_frac=.1, prevent_disconnect=True, verbose=True)
g_train = nx.from_scipy_sparse_array(adj_train)


# In[26]:

path = '/home/ukjung18/GIE1/GIE/GIE-master/src_data/' + args.directory

try:
    if not os.path.exists(path):
        os.makedirs(path)
except OSError:
    print ('Error: Directory exists.')

with open(path+'train', "w") as f:
    for edge, tr in zip(train_edges, train_rel):
        f.write(str(edge[0])+'\t'+str(tr)+'\t'+str(edge[1])+'\n')

with open(path+'valid', "w") as f:
    for edge, vr in zip(val_edges, val_rel):
        f.write(str(edge[0])+'\t'+str(vr)+'\t'+str(edge[1])+'\n')
        
with open(path+'test', "w") as f:
    for edge, t_r in zip(test_edges, test_rel):
        f.write(str(edge[0])+'\t'+str(t_r)+'\t'+str(edge[1])+'\n')
        
with open(path+'test_neg', "w") as f:
    for edge, t_r in zip(test_edges_false, test_rel):
        f.write(str(edge[0])+'\t'+str(t_r)+'\t'+str(edge[1])+'\n')
        
with open(path+'test_neg_inv', "w") as f:
    for edge, t_r in zip(test_inv_false, test_rel):
        f.write(str(edge[0])+'\t'+str(t_r)+'\t'+str(edge[1])+'\n')

# with open(path+'test_neg_rel', "w") as f:
#     for edge, t_r in zip(test_edges, test_rel_false):
#         f.write(str(edge[0])+'\t'+str(t_r)+'\t'+str(edge[1])+'\n')
        
# with open(path+'val_neg_rel', "w") as f:
#     for edge, t_r in zip(test_edges, val_rel_false):
#         f.write(str(edge[0])+'\t'+str(t_r)+'\t'+str(edge[1])+'\n')

