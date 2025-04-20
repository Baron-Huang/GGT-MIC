
#
for seed in 3407
do
    echo "Running with random_seed=$seed"
    python HCFMIL_main.py --abla_type sota --run_mode test --random_seed ${seed} --feat_extract\
            --bags_len 1025 --test_weights_feature /data/MIL/TicMIL/Weights_Result_Text/WSI/Cervix/96_96/sota_test/Simple_SwinT_sota_Feature_Epoch13_Seed3407.pth \
            --test_weights_head /data/MIL/TicMIL/Weights_Result_Text/WSI/Cervix/96_96/sota_test/Simple_SwinT_sota_Head_Epoch13_Seed3407.pth
done
