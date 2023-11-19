# TransE MurE TransH MurP RotH RefH AttH ATT2_rot ATT2_ref ATT2_trans GIE_rot GIE_ref GIE_trans
# MODELS="TransH MurP RotH RefH AttH ScaleH"
# for i in $MODELS
# do
#     CUDA_VISIBLE_DEVICES=1 python3 ../run.py --dataset GO1117 --model $i --rank 300 --regularizer N3 --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 5 --batch_size 1000 --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn --dtype double --multi_c
# done

MODELS=(TransH MurP RotH RefH AttH ScaleH)
PTH=/home/ukjung18/GIE1/GIE/GIE-master/LOG_DIR/11_17/GO1117/
PRTD=(TransH_15_27_31 MurP_15_57_44 RotH_17_28_21 RefH_18_50_48 AttH_20_08_11 ScaleH_22_15_31)
for i in "${!MODELS[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python3 ../run.py --dataset GOA1117 --model ${MODELS[i]} --rank 300 --regularizer N3 \
        --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 5 --batch_size 1000 \
        --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn \
        --dtype double --multi_c --prtd ${PTH}${PRTD[i]}
done

# GOREL=(is_a part_of has_part regulates positively_regulates negatively_regulates occurs_in)
# for i in "${!MODELS[@]}"
# do
#     for j in "${!GOREL[@]}"
#     do
#         CUDA_VISIBLE_DEVICES=1 python3 ../run.py --dataset ${GOREL[j]} --model ${MODELS[i]} --rank 20 --regularizer N3 \
#             --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 5 --batch_size 1000 \
#             --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn \
#             --dtype double --multi_c
#     done
# done
