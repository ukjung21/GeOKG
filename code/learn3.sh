# TransE MurE RotE RefE AttE TransH MurP RotH RefH AttH ATT2_rot ATT2_ref ATT2_trans GIE_rot GIE_ref GIE_trans
# MODELS="customGIEAtt_3 customGIEAtt_4 customATT2Att_3 customATT2Att_4 customGIEAtt3 customATT2Att3"
# GOREL=(is_a part_of has_part regulates positively_regulates negatively_regulates occurs_in)
# MODELS="ATT2_rot ATT2_ref ATT2_trans"
# for i in $MODELS
# do
#     CUDA_VISIBLE_DEVICES=2 python3 ../run.py --dataset GO1117 --model $i --rank 300 --regularizer N3 --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 5 --batch_size 1000 --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn --dtype double --multi_c
# done

MODELS=(ATT2_rot ATT2_ref ATT2_trans)
PTH=/home/ukjung18/GIE1/GIE/GIE-master/LOG_DIR/11_17/GO1117/
PRTD=(ATT2_rot_15_27_28 ATT2_ref_17_31_31 ATT2_trans_19_43_31)
for i in "${!MODELS[@]}"
do
    CUDA_VISIBLE_DEVICES=2 python3 ../run.py --dataset GOA1117 --model ${MODELS[i]} --rank 300 --regularizer N3 \
        --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 5 --batch_size 1000 \
        --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn \
        --dtype double --multi_c --prtd ${PTH}${PRTD[i]}
done

# GOREL=(GO1015-is_a GO1015-part_of GO1015-has_part GO1015-regulates GO1015-positively_regulates GO1015-negatively_regulates GO1015-occurs_in)
# for i in "${!MODELS[@]}"
# do
#     for j in "${!GOREL[@]}"
#     do
#         CUDA_VISIBLE_DEVICES=2 python3 ../run.py --dataset ${GOREL[j]} --model ${MODELS[i]} --rank 20 --regularizer N3 \
#             --reg 0.0 --optimizer Adam --max_epochs 300 --patience 15 --valid 5 --batch_size 1000 \
#             --neg_sample_size 50 --init_size 0.001 --learning_rate 0.001 --gamma 0.0 --bias learn \
#             --dtype double --multi_c
#     done
# done