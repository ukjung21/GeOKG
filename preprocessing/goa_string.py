import pandas as pd
import numpy as np
import pickle
from scipy.sparse import coo_matrix
import numpy as np
import os

def data_split(trp):
    train_df = pd.DataFrame()
    valid_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for r in trp['rel'].unique():
        tmp = trp[trp['rel'] == r]
        tmp = tmp.sample(frac=1).reset_index(drop=True)
        n = tmp.shape[0]
        train_idx = int(n*0.8)
        valid_idx = train_idx + (n-train_idx)//2
        
        train_df = pd.concat([train_df, tmp.iloc[:train_idx]])
        valid_df = pd.concat([valid_df, tmp.iloc[train_idx:valid_idx]])
        test_df = pd.concat([test_df, tmp.iloc[valid_idx:]])
        
    return train_df, valid_df, test_df

def neg_sampling(u, v, goa_num, test_df):

	n = goa_num
	adj = coo_matrix((np.ones(len(u)), (np.array(u), np.array(v))), shape=(n, n))
	k_hop = 5
	k_mat = adj.tocsr()**5

	neg_u, neg_v = np.where((k_mat.todense() > 0) & (adj.todense() == 0))

	neg_eids = np.random.choice(len(neg_u), len(test_df), replace=False)
	test_neg_u, test_neg_v = neg_u[neg_eids], neg_v[neg_eids]

	neg_df = pd.DataFrame(list(zip(test_neg_u, test_df['rel'], test_neg_v)), columns=['head', 'rel', 'tail'])

	return neg_df

def save_indexed_goa(goa_dir, train_df, valid_df, test_df, neg_df):
	# Check if we have the ./data directory already
	goa_dir = '../data/GOA0404/'
	if(not os.path.isfile(goa_dir)):
		# Emulate mkdir -p (no error if folder exists)
		try:
			os.mkdir(goa_dir)
		except OSError as e:
			if(e.errno != 17):
				raise e
	else:
		raise Exception('Data path (' + goa_dir + ') exists as a file. '
					'Please rename, remove or change the desired location of the data path.')
		
	with open(goa_dir+"train.pickle", 'wb') as f:
		pickle.dump(train_df.to_numpy().astype('uint64'), f)
	with open(goa_dir+"valid.pickle", 'wb') as f:
		pickle.dump(valid_df.to_numpy().astype('uint64'), f)
	with open(goa_dir+"test.pickle", 'wb') as f:
		pickle.dump(test_df.to_numpy().astype('uint64'), f)
	with open(goa_dir+"test_neg.pickle", 'wb') as f:
		pickle.dump(neg_df.to_numpy().astype('uint64'), f)

def save_indexed_ppi_bin(src_dir, prot_to_id, eval_bin_pth):
	ppi_df = pd.read_csv(src_dir+'goa_bin_ppi.tsv', sep='\t')

	ppi = []
	for s,t in zip(ppi_df['source'], ppi_df['target']):
		ppi.append([prot_to_id[s],prot_to_id[t],1])
	ppi = pd.DataFrame(ppi, columns=['source', 'target', 'class'])

	edge_tuples = ppi.apply(lambda row: (min(row[0], row[1]), max(row[0], row[1])), axis=1)
	all_edge_tuples = set(edge_tuples)
	ppi_set = pd.DataFrame(all_edge_tuples)

	false_edge_set = set()
	n_min = min(prot_to_id.values())
	n_max = max(prot_to_id.values())

	while len(false_edge_set) < ppi_set.shape[0]:
		head = np.random.randint(n_min, n_max+1)
		tail = np.random.randint(n_min, n_max+1)
		if head == tail:
			continue
		false_edge = (min(head,tail), max(head,tail))
		if false_edge in all_edge_tuples:
			continue
		if false_edge in false_edge_set:
			continue
		else:
			false_edge_set.add(false_edge)

	ppi_neg=pd.DataFrame(false_edge_set, columns=[0, 1])

	ppi_set['class']=[1]*ppi_set.shape[0]
	ppi_neg['class']=[0]*ppi_neg.shape[0]
	ppi = pd.concat([ppi_set, ppi_neg], ignore_index=True)

	ppi.to_csv(eval_bin_pth, index=False, header=None)

def save_indexed_ppi_score(src_dir, prot_to_id, eval_score_pth):
	ppi_score_df = pd.read_csv(src_dir+'goa_score_ppi.tsv', sep='\t')
	ppi_score_df['source'] = ppi_score_df.apply(lambda row: prot_to_id[row['source']], axis=1)
	ppi_score_df['target'] = ppi_score_df.apply(lambda row: prot_to_id[row['target']], axis=1)
	ppi_score_df.to_csv(eval_score_pth, index=False, header=None)

def save_indexed_ppi_type(src_dir, prot_to_id, eval_type_pth):
	ppi_type_df = pd.read_csv(src_dir+'goa_type_ppi.tsv', sep='\t')
	ppi_type_df['item_id_a'] = ppi_type_df.apply(lambda row: prot_to_id[row['item_id_a']], axis=1)
	ppi_type_df['item_id_b'] = ppi_type_df.apply(lambda row: prot_to_id[row['item_id_b']], axis=1)
	ppi_type_df.to_csv(eval_type_pth, index=False, header=None)

def main():
	src_dir = "../src_data/GO0719/"
	eval_bin_pth = "../evalGene/GOA0719_ppi.csv"
	eval_score_pth = "../evalGene/GOA0719_score_ppi.csv"
	eval_type_pth = "../evalGene/GOA0719_type_ppi.csv"
	goa_dir = "../data/GOA0719/"

	go_trp = pd.read_csv(src_dir+'edges.tsv', sep = "\t")
	goa_trp = pd.read_csv(src_dir+'goa_edges.tsv', sep = "\t")
	ent_df = pd.read_csv(src_dir+'entities.tsv', sep='\t')

	trp = pd.concat([go_trp, goa_trp], ignore_index=True)

	with open(src_dir+"prt_list.pkl","rb") as f:
		prot_list = pickle.load(f)
		
	ent_to_id={}
	ent_to_id = {x: i for (i, x) in enumerate(ent_df['term'])}
	cnt = len(ent_to_id)

	prot_to_id = {x: i+cnt for (i, x) in enumerate(prot_list)}
	ent_to_id.update(prot_to_id)

	rel = set()
	for r in trp['rel']:
		rel.add(r)
	rel_to_id = {x: i for (i, x) in enumerate(sorted(rel))}

	u = trp.apply(lambda row: ent_to_id[row['head']], axis=1)
	r = trp.apply(lambda row: rel_to_id[row['rel']], axis=1)
	v = trp.apply(lambda row: ent_to_id[row['tail']], axis=1)
	trp = pd.DataFrame(list(zip(u, r, v)), columns=['head', 'rel', 'tail'])

	train_df, valid_df, test_df = data_split(trp)
	neg_df = neg_sampling(u, v, goa_num=len(ent_to_id), test_df=test_df)

	save_indexed_goa(goa_dir, train_df, valid_df, test_df, neg_df)

	save_indexed_ppi_bin(src_dir, prot_to_id, eval_bin_pth)
	save_indexed_ppi_score(src_dir, prot_to_id, eval_score_pth)
	save_indexed_ppi_type(src_dir, prot_to_id, eval_type_pth)
    
if __name__ == "__main__":
    main()