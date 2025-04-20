#paras = argparse.ArgumentParser(description='TicMIL Hyparameters')
#    paras.add_argument('--random_seed', type=int, default=1)
#    paras.add_argument('--gpu_device', type=int, default=0)
#    paras.add_argument('--class_num', type=int, default=3)
#    paras.add_argument('--batch_size', type=int, default=2)
#    paras.add_argument('--epochs', type=int, default=100)
#    paras.add_argument('--img_size', type=list, default=[96, 96])
#    paras.add_argument('--bags_len', type=int, default=1025)
#    paras.add_argument('--num_workers', type=int, default=16)
#    paras.add_argument('--worker_time_out', type=int, default=0)
#    paras.add_argument('--data_parallel', type=bool, default=True)
#    paras.add_argument('--run_mode', type=str, default='train')
#    paras.add_argument('--abla_type', type=str, default='baseline')
#    paras.add_argument('--parallel_gpu_ids', type=list, default=[0,1,2,3])
#    paras.add_argument('--train_read_path', type=str,
#                default=r'/data/MIL/TicMIL/Datasets/Cervix/New_Bags/Train')
#    paras.add_argument('--test_read_path', type=str,
#                default=r'/data/MIL/TicMIL/Datasets/Cervix/New_Bags/Test')
#    paras.add_argument('--val_read_path', type=str,
#                default=r'/data/MIL/TicMIL/Datasets/Cervix/New_Bags/Test')
#
#    paras.add_argument('--weights_save_path', type=str,
#                default=r'/data/MIL/TicMIL/Weights_Result_Text/WSI/Cervix_WSI_Baseline_20240817.pth')
#    paras.add_argument('--test_weights_path', type=str,
#                default=r'/data/MIL/TicMIL/Weights_Result_Text/WSI/Cervix_WSI_SOTA_20240813.pth')
#
#
#    ### Parallel save
#    paras.add_argument('--weights_save_feature', type=str,
#                        default=r'/data/MIL/TicMIL/Weights_Result_Text/WSI/xxxx_feature.pth')
#    paras.add_argument('--weights_save_head', type=str,
#                        default=r'/data/MIL/TicMIL/Weights_Result_Text/WSI/xxxx_head.pth')
#
#    ### Parallel test
#    paras.add_argument('--test_weights_feature', type=str,
#                        default=r'/data/MIL/TicMIL/Weights_Result_Text/WSI/TicMIL_Feature.pth')
#    paras.add_argument('--test_weights_head', type=str,
#                        default=r'/data/MIL/TicMIL/Weights_Result_Text/WSI/TicMIL_Head.pth')
#
#    ### Pretrained
#    paras.add_argument('--pretrained_weights_path', type=str,
#                default=r'/data/MIL/TicMIL/Weights/SwinT/swin_tiny_patch4_window7_224_22k.pth')





#
for seed in 3407
do
    echo "Running with random_seed=$seed"
    python TicMIL_main_lscc.py --abla_type sota --run_mode test --random_seed ${seed} --bag_weight \
            --bags_len 1042 --test_weights_feature /data/MIL/TicMIL/Weights_Result_Text/WSI/Larynx/96_96/sota/Simple_SwinT_sota_Feature_final_seed3407.pth \
            --test_weights_head /data/MIL/TicMIL/Weights_Result_Text/WSI/Larynx/96_96/sota/Simple_SwinT_sota_Head_final_seed3407.pth
done