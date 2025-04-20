
#
for seed in 3407
do
    echo "Running with random_seed=$seed"
    python HCFMIL_main_lscc.py --abla_type sota --run_mode test --random_seed ${seed} --bag_weight \
            --bags_len 1042
done
