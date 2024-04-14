# GeOKG: Geometry-aware representation of Gene Ontology and Genes

### Data Preprocessing
To download and preprocess datasets (GO, GOA, STRING), run `preprocessing/go_string.ipynb` and `preprocessing/goa_string.ipynb`.
Then run `filter_gen.py`

### Embedding GO
To train GeOKG for GO embeddings 
```
$ python3 run.py --dataset GO0404 --model GeOKG_go --rank 200 --batch_size 256 --multi_c
```
Then the evaluation results (GO-level link prediction, link reconstruction, relation-type prediction) are recorded in `evalGO/eval_metrics.tsv`

### Embedding GOA
To train GeOKG for GOA embeddings 
```
$ python3 run.py --dataset GOA0404 --model GeOKG --rank 200 --batch_size 256 --multi_c
```

### PPI Prediction
To evaluate gene embeddings for 3 PPI predictions
* PPI Binary Prediction
```
$ python evalGene/binary_prediction_NN.py -path evalGO/{date}/GOA0404/entity_embedding.npy -dset evalGene/GOA0404_ppi.csv -model result/GeOKG_binary.pth -fout evalGene/binary_output.txt
```
* Binding Affinity Prediction
```
$ python evalGene/score_prediction_NN.py -path evalGO/{date}/GOA0404/entity_embedding.npy -dset evalGene/GOA0404_score_ppi.csv -model result/GeOKG_score.pth -fout evalGene/score_output.txt
```
* Interaction Type Prediction
```
$ python evalGene/type_prediction_NN.py -path evalGO/{date}/GOA0404/entity_embedding.npy -dset evalGene/GOA0404_type_ppi.csv -model result/GeOKG_type.pth -fout evalGene/type_output.txt
```
