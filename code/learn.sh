# TransE MurE TransH MurP RotH RefH AttH ATT2_rot ATT2_ref ATT2_trans GIE_rot GIE_ref GIE_trans

MODELS="GIE_rot"
for i in $MODELS
do
    CUDA_VISIBLE_DEVICES=0 python3 ../run.py --dataset GO1117 --model $i --rank 300 --regularizer N3 --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 5 --batch_size 1000 --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn --dtype double --multi_c
done

# Finetuning GOA with pretrained GO embeddings
# MODELS=(GIE_rot)
# PTH=LOG_DIR/11_17/GO1117/
# PRTD=(GIE_rot_15_27_23) # pretrained_GO_embeddings
# for i in "${!MODELS[@]}"
# do
#     CUDA_VISIBLE_DEVICES=0 python3 ../run.py --dataset GOA1117 --model ${MODELS[i]} --rank 300 --regularizer N3 \
#         --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 1 --batch_size 1000 \
#         --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn \
#         --dtype double --multi_c --prtd ${PTH}${PRTD[i]}
# done