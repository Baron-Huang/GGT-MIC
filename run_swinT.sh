
#
for seed in 3407
do
    echo "Running with random_seed=$seed"
    python HCFMIL_main.py --abla_type sota --run_mode test --random_seed ${seed} --feat_extract\
            --bags_len 1025
done
