# TransE MurE TransH MurP RotH RefH AttH ATT2_rot ATT2_ref ATT2_trans GIE_rot GIE_ref GIE_trans
# MODELS="TransE MurE RotE RefE AttE" # TransE MurE RotE RefE AttE
# for i in $MODELS
# do
#     CUDA_VISIBLE_DEVICES=0 python3 ../run.py --dataset GO1117 --model $i --rank 300 --regularizer N3 --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 5 --batch_size 1000 --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn --dtype double --multi_c
# done

MODELS=(TransE MurE RotE RefE AttE) # RotE RefE AttE TransE MurE GIE_rot GIE_ref
PTH=/home/ukjung18/GIE1/GIE/GIE-master/LOG_DIR/11_17/GO1117/
PRTD=(TransE_15_27_35 MurE_15_42_27 RotE_16_01_53 RefE_16_39_05 AttE_17_16_10)
for i in "${!MODELS[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python3 ../run.py --dataset GOA1117 --model ${MODELS[i]} --rank 300 --regularizer N3 \
        --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 5 --batch_size 1000 \
        --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn \
        --dtype double --multi_c --prtd ${PTH}${PRTD[i]}
done

# GOREL=(GO1015-is_a GO1015-part_of GO1015-has_part GO1015-regulates GO1015-positively_regulates GO1015-negatively_regulates GO1015-occurs_in)
# for i in "${!MODELS[@]}"
# do
#     for j in "${!GOREL[@]}"
#     do
#         CUDA_VISIBLE_DEVICES=0 python3 ../run.py --dataset ${GOREL[j]} --model ${MODELS[i]} --rank 300 --regularizer N3 \
#             --reg 0.0 --optimizer Adam --max_epochs 500 --patience 15 --valid 5 --batch_size 1000 \
#             --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn \
#             --dtype double --multi_c
#     done
# done

# CUDA_VISIBLE_DEVICES=0 python3 learn.py --dataset GO1008 --model GIE --rank 800 --optimizer Adagrad \
#     --learning_rate 1e-1 --batch_size 1000 --regularizer N3 --reg 5e-2 --max_epochs 300 --valid 5 -train -id GO1008_learn -save