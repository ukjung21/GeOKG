# GeOKG: Geometry-aware representation of Gene Ontology and Genes

### Data Preprocessing
To download and preprocess datasets (GO, GOA, STRING), run `preprocessing/go_string.py` and `preprocessing/goa_string.py`.
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
* Interaction Binary Prediction
```
$ python evalGene/binary_prediction_NN.py -path evalGO/{date}/GOA0404/entity_embedding.npy -dset evalGene/GOA0404_ppi.csv -model result/GeOKG_binary.pth -fout evalGene/binary_output.txt
```
* Interaction Score Prediction
```
$ python evalGene/score_prediction_NN.py -path evalGO/{date}/GOA0404/entity_embedding.npy -dset evalGene/GOA0404_score_ppi.csv -model result/GeOKG_score.pth -fout evalGene/score_output.txt
```
* Interaction Type Prediction
```
$ python evalGene/type_prediction_NN.py -path evalGO/{date}/GOA0404/entity_embedding.npy -dset evalGene/GOA0404_type_ppi.csv -model result/GeOKG_type.pth -fout evalGene/type_output.txt
```   
   
To replicate the gene-level experiment results, follow the download link below and utilize the provided gene embeddings.   
[GeOKG GOA embeddings 50, 100, 200, 500 and 1000 dim](https://drive.google.com/drive/folders/1sQXpW-jtdMdo4KFr5vOJlWf13uj5JTL1?usp=drive_link)
