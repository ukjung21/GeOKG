from goatools import obo_parser
import wget
import os
import pandas as pd
from collections import defaultdict
from scipy.sparse import coo_matrix, dok_matrix
import numpy as np
import pickle
import Bio.UniProt.GOA as GOA
from ftplib import FTP
import gzip
import shutil

def load_go(data_folder, go_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'):
	# Check if we have the ./data directory already
	if(not os.path.isfile(data_folder)):
		# Emulate mkdir -p (no error if folder exists)
		try:
			os.mkdir(data_folder)
		except OSError as e:
			if(e.errno != 17):
				raise e
	else:
		raise Exception('Data path (' + data_folder + ') exists as a file.\n \
						Please rename, remove or change the desired location of the data path.')

	# Check if the file exists already
	if(not os.path.isfile(data_folder+'go-basic.obo')):
		go_obo = wget.download(go_obo_url, data_folder+'go-basic.obo')
	else:
		go_obo = data_folder+'go-basic.obo'
		
	go = obo_parser.GODag(go_obo, optional_attrs=['relationship'])
	
	return go

def neg_sampling(u, v, go_num, test_df):
	adj = dok_matrix((go_num, go_num))
	for (h, t) in zip(u,v):
		adj[t, h] += 1
		adj[h, t] += 1
	adj = adj.tocsr()
	k_mat = adj.tocsr()**3

	neg_u, neg_v = np.where((k_mat.todense() > 0) & (adj.todense() == 0))

	neg_eids = np.random.choice(len(neg_u), test_df.shape[0], replace=False)
	test_neg_u, test_neg_v = neg_u[neg_eids], neg_v[neg_eids]

	neg_df = pd.DataFrame(list(zip(test_neg_u, test_df['rel'], test_neg_v)), columns=['head', 'rel', 'tail'])

	return neg_df

def data_split(go_id_df):
	train_df = pd.DataFrame()
	valid_df = pd.DataFrame()
	test_df = pd.DataFrame()
	for r in go_id_df['rel'].unique():
		tmp = go_id_df[go_id_df['rel'] == r]
		tmp = tmp.sample(frac=1).reset_index(drop=True)
		
		n = tmp.shape[0]
		train_idx = int(n*0.8)
		valid_idx = train_idx + (n-train_idx)//2

		train_df = pd.concat([train_df, tmp.iloc[:train_idx]])
		valid_df = pd.concat([valid_df, tmp.iloc[train_idx:valid_idx]])
		test_df = pd.concat([test_df, tmp.iloc[valid_idx:]])

	return train_df, valid_df, test_df

def save_go(train_df, valid_df, test_df, neg_df, go_dir):
	if(not os.path.isfile(go_dir)):
    # Emulate mkdir -p (no error if folder exists)
		try:
			os.mkdir(go_dir)
		except OSError as e:
			if(e.errno != 17):
				raise e
	else:
		raise Exception('Data path (' + go_dir + ') exists as a file. '
					'Please rename, remove or change the desired location of the data path.')
	
	with open(go_dir+"train.pickle", 'wb') as f:
		pickle.dump(train_df.to_numpy().astype('uint64'), f)
	with open(go_dir+"valid.pickle", 'wb') as f:
		pickle.dump(valid_df.to_numpy().astype('uint64'), f)
	with open(go_dir+"test.pickle", 'wb') as f:
		pickle.dump(test_df.to_numpy().astype('uint64'), f)
	with open(go_dir+"test_neg.pickle", 'wb') as f:
		pickle.dump(neg_df.to_numpy().astype('uint64'), f)

def save_src_data(go_df, ent_df, src_dir):
	if(not os.path.isfile(src_dir)):
    # Emulate mkdir -p (no error if folder exists)
		try:
			os.mkdir(src_dir)
		except OSError as e:
			if(e.errno != 17):
				raise e
			
	else:
		raise Exception('Data path (' + src_dir + ') exists as a file. '
					'Please rename, remove or change the desired location of the data path.')
		
	go_df.to_csv(src_dir+'edges.tsv', sep='\t', index=False)
	ent_df.to_csv(src_dir+'entities.tsv', sep='\t', index=False)

def load_goa(human_uri, raw_data_dir):
	human_fn = human_uri.split('/')[-1]

	# Check if the file exists already
	human_gaf = os.path.join(raw_data_dir, human_fn)
	if(not os.path.isfile(human_gaf)):
		# Login to FTP server
		ebi_ftp = FTP('ftp.ebi.ac.uk')
		ebi_ftp.login() # Logs in anonymously
		
		# Download
		with open(human_gaf,'wb') as human_fp:
			ebi_ftp.retrbinary('RETR {}'.format(human_uri), human_fp.write)
			
		# Logout from FTP server
		ebi_ftp.quit()

	gaf_list =[]
	with gzip.open(human_gaf, 'rt') as human_gaf_fp:
		for entry in GOA.gafiterator(human_gaf_fp):
			gaf_list.append(entry)

	goa=[] # protein - qualifier - GO
	acts_list = ['acts_upstream_of', 'acts_upstream_of_positive_effect', 'acts_upstream_of_negative_effect', 'acts_upstream_of_or_within_negative_effect', 'acts_upstream_of_or_within_positive_effect']
	for gene in gaf_list:
		if gene['Evidence'] == 'ND':
			continue
		if len(gene['Qualifier']) > 1:
			continue
		else:
			q = gene['Qualifier'][0]
			if q in acts_list:
				q='acts_upstream_of_or_within'
			goa.append([gene['DB_Object_ID'],'goa_'+q,gene['GO_ID']])
	goa_df=pd.DataFrame(goa, columns=['head', 'rel', 'tail'])

	return goa_df

def save_goa(goa_df, uniprt_ls, src_dir):
	# Remove proteins in goa, but not in str2prot
	condition = goa_df['head'].isin(uniprt_ls)
	goa_df_filtered = goa_df[condition]
	goa_df_filtered.reset_index(drop=True, inplace=True)
	goa_df_filtered.to_csv(src_dir+'goa_edges.tsv', index=False, sep='\t')

	prot_set = set()
	for prot in goa_df_filtered['head']:
		prot_set.add(prot)
	prt_ls = list(prot_set)
	
	with open(src_dir+"prt_list.pkl","wb") as f:
		pickle.dump(prt_ls, f)

	return prt_ls

def load_ppi(raw_data_dir):

	ppi_url = "https://stringdb-downloads.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz"
	ppi_physical_url = "https://stringdb-downloads.org/download/protein.physical.links.v11.5/9606.protein.physical.links.v11.5.txt.gz"
	ppi_action_url = "https://stringdb-static.org/download/protein.actions.v11.0/9606.protein.actions.v11.0.txt.gz"

	wget.download(ppi_url, out=raw_data_dir)
	wget.download(ppi_physical_url, out=raw_data_dir)
	wget.download(ppi_action_url, out=raw_data_dir)

	for url in [ppi_url, ppi_physical_url, ppi_action_url]:
		with gzip.open(f"{raw_data_dir}{url.split('/')[-1]}", 'rb') as f_in:
			with open(f"{raw_data_dir}{url.split('/')[-1].replace('.gz','')}", 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)

def save_ppi_bin(src_dir, strprt_ls, ppi_dict, prt_ls, raw_data_dir):
	link = pd.read_csv(raw_data_dir+"9606.protein.links.v11.5.txt", sep='\s', engine='python', encoding='cp949')

	# Remove proteins in ppi, but not in str2prot
	condition1 = link['protein1'].isin(strprt_ls)
	condition2 = link['protein2'].isin(strprt_ls)
	condition = condition1 & condition2
	link_filtered = link[condition]
	link_filtered.reset_index(drop=True, inplace=True)

	ppi_df = pd.DataFrame(columns=['source', 'target', 'score'])
	ppi_df['source'] = link_filtered.apply(lambda row: ppi_dict[row['protein1']], axis=1)
	ppi_df['target'] = link_filtered.apply(lambda row: ppi_dict[row['protein2']], axis=1)
	ppi_df['score'] = link_filtered['combined_score']

	condition1 = ppi_df['source'].isin(prt_ls)
	condition2 = ppi_df['target'].isin(prt_ls)
	condition = condition1 & condition2
	ppi_df_filtered = ppi_df[condition]
	ppi_df_filtered.reset_index(drop=True, inplace=True)

	# generate positive links for binary classification
	ppi_df_filtered_bin = ppi_df_filtered[ppi_df_filtered['score']>=900]
	ppi_df_filtered_bin.reset_index(drop=True, inplace=True)
	ppi_df_filtered_bin.to_csv(src_dir+'goa_bin_ppi.tsv', sep='\t', index=False)

def save_ppi_score(src_dir, strprt_ls, ppi_dict, prt_ls, raw_data_dir):
	physical_link_df = pd.read_csv(raw_data_dir+"9606.protein.physical.links.v11.5.txt", sep='\s', engine='python', encoding='cp949')

	condition1 = physical_link_df['protein1'].isin(strprt_ls)
	condition2 = physical_link_df['protein2'].isin(strprt_ls)
	condition = condition1 & condition2
	physical_link_df_filtered = physical_link_df[condition]
	physical_link_df_filtered.reset_index(drop=True, inplace=True)
	
	ppi_score_df = pd.DataFrame(columns=['source', 'target', 'score'])
	ppi_score_df['source'] = physical_link_df_filtered.apply(lambda row: ppi_dict[row['protein1']], axis=1)
	ppi_score_df['target'] = physical_link_df_filtered.apply(lambda row: ppi_dict[row['protein2']], axis=1)
	ppi_score_df['score'] = physical_link_df_filtered['combined_score']

	condition1 = ppi_score_df['source'].isin(prt_ls)
	condition2 = ppi_score_df['target'].isin(prt_ls)
	condition = condition1 & condition2

	ppi_score_df_filtered = ppi_score_df[condition]
	ppi_score_df_filtered.reset_index(drop=True, inplace=True)
	ppi_score_df_filtered.drop_duplicates(inplace=True, ignore_index=True)
	ppi_score_df_filtered.to_csv(src_dir+'goa_score_ppi.tsv', sep='\t', index=False)

def save_ppi_type(src_dir, str2prot, ppi_dict, prt_ls, raw_data_dir):
	action_link_df = pd.read_csv(raw_data_dir+"9606.protein.actions.v11.0.txt", sep='\s', engine='python', encoding='cp949')

	one_hot_encoding = pd.get_dummies(action_link_df['mode'])
	type_df = pd.concat([action_link_df, one_hot_encoding], axis=1)
	type_df = type_df.drop(['mode', 'action', 'is_directional', 'a_is_acting', 'score'], axis=1)
	type_df.drop_duplicates(inplace=True, ignore_index=True)
	ppi_type_df = type_df.groupby(['item_id_a', 'item_id_b']).sum().reset_index()

	prt_cond = str2prot['Entry'].isin(prt_ls)
	str2prot_filtered = str2prot[prt_cond]
	ppi_dict = {i:j for i,j in zip(str2prot_filtered['From'], str2prot_filtered['Entry'])}
	strprt_ls_type = list(str2prot_filtered['From'])

	condition1 = ppi_type_df['item_id_a'].isin(strprt_ls_type)
	condition2 = ppi_type_df['item_id_b'].isin(strprt_ls_type)
	condition = condition1 & condition2
	ppi_type_df_filtered = ppi_type_df[condition]
	ppi_type_df_filtered.reset_index(drop=True, inplace=True)

	ppi_type_df_filtered['item_id_a'] = ppi_type_df_filtered.apply(lambda row: ppi_dict[row['item_id_a']], axis=1)
	ppi_type_df_filtered['item_id_b'] = ppi_type_df_filtered.apply(lambda row: ppi_dict[row['item_id_b']], axis=1)

	ppi_type_df_filtered.to_csv(src_dir+'goa_type_ppi.tsv', sep='\t', index=False)

def main():

	raw_data_dir = "../raw_data/"
	go_dir = "../data/GO0719/"
	src_dir = "../src_data/GO0719/"

	go = load_go(raw_data_dir)

	go_edges = []
	go_entities = []
	alts = []
	obs_cnt = 0
	for go_id in go:
		go_term = go[go_id]
		go_entities.append([go_term.id, go_term.level, go_term.namespace])
		
		if go_term.is_obsolete:
			obs_cnt+=1
			continue
		if go_term._parents:
			for parents in go_term.parents:
				go_edges.append([go_term.id, 'is_a', go[parents.id].id])
		if go_term.alt_ids:
			alts+=go_term.alt_ids
		if go_term.relationship:
			for rel in go_term.relationship:
				for e in go_term.relationship[rel]:
					go_edges.append([go_term.id, rel, go[e.id].id])
	
	go_df = pd.DataFrame(go_edges, columns=['head', 'rel', 'tail'])
	go_df.drop_duplicates(inplace=True, ignore_index=True)

	ent_df = pd.DataFrame(go_entities, columns=['term', 'level', 'class'])
	ent_df.drop_duplicates(inplace=True, ignore_index=True)

	node_dict = defaultdict(int) # {GO:0000001 : 0}
	rel_dict = defaultdict(int)
	go_list = list(ent_df['term']) # {'is_a' : 0}
	rel_list = list(go_df['rel'].unique())

	for i in range(len(go_list)):
		node_dict[go_list[i]] = i
	for i in range(len(rel_list)):
		rel_dict[rel_list[i]] = i

	u = go_df.apply(lambda row: node_dict[row['head']], axis=1)
	r = go_df.apply(lambda row: rel_dict[row['rel']], axis=1)
	v = go_df.apply(lambda row: node_dict[row['tail']], axis=1)
	go_id_df = pd.DataFrame(list(zip(u, r, v)), columns=['head', 'rel', 'tail'])

	train_df, valid_df, test_df = data_split(go_id_df)
	neg_df = neg_sampling(u, v, go_num=ent_df.shape[0], test_df=test_df)

	save_go(train_df, valid_df, test_df, neg_df, go_dir)
	save_src_data(go_df, ent_df, src_dir)

	goa_df = load_goa("/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz", raw_data_dir)
	ent_ls = list(ent_df['term'])
	condition = goa_df['tail'].isin(ent_ls)
	goa_df_filtered = goa_df[condition]
	goa_df_filtered.reset_index(drop=True, inplace=True)
	goa_df_filtered.drop_duplicates(inplace=True, ignore_index=True)

	str2prot = pd.read_csv(raw_data_dir+"string2uniprot.tsv", sep='\t')
	uniprt_ls = list(str2prot['Entry'])
	strprt_ls = list(str2prot['From'])
	ppi_dict = {i:j for i,j in zip(str2prot['From'], str2prot['Entry'])}

	prt_ls = save_goa(goa_df_filtered, uniprt_ls, src_dir)

	load_ppi(raw_data_dir)

	save_ppi_bin(src_dir, strprt_ls, ppi_dict, prt_ls, raw_data_dir)
	save_ppi_score(src_dir, strprt_ls, ppi_dict, prt_ls, raw_data_dir)
	save_ppi_type(src_dir, str2prot, ppi_dict, prt_ls, raw_data_dir)

if __name__ == "__main__":
    main()

