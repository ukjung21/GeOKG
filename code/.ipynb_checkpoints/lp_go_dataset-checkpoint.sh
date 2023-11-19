python3 lp_go_dataset.py --directory GO_rel_21/
python3 process_datasets.py --directory GO_rel_21
python3 lp_go_dataset.py --directory GO_rel_22/
python3 process_datasets.py --directory GO_rel_22
python3 lp_go_dataset.py --directory GO_rel_23/
python3 process_datasets.py --directory GO_rel_23
python3 lp_go_dataset.py --directory GO_rel_24/
python3 process_datasets.py --directory GO_rel_24
python3 lp_go_dataset.py --directory GO_rel_25/
python3 process_datasets.py --directory GO_rel_25
python3 lp_go_dataset.py --directory GO_rel_26/
python3 process_datasets.py --directory GO_rel_26
python3 lp_go_dataset.py --directory GO_rel_27/
python3 process_datasets.py --directory GO_rel_27
python3 lp_go_dataset.py --directory GO_rel_28/
python3 process_datasets.py --directory GO_rel_28
python3 lp_go_dataset.py --directory GO_rel_29/
python3 process_datasets.py --directory GO_rel_29
python3 lp_go_dataset.py --directory GO_rel_30/
python3 process_datasets.py --directory GO_rel_30


# CUDA_VISIBLE_DEVICES=2 python3 ../run.py --dataset GO_1 --model GIE --rank 100 --regularizer N3 --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 5 --batch_size 1000 --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn --dtype double --space att3 --double_neg --multi_c