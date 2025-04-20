############################# fit_functions ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: fitting functions for DHM_MIL models
import os.path
import pdb
from itertools import accumulate

from torch.cuda.amp import autocast as autocast, GradScaler
########################## API Section #########################
import torch
from torch import nn
import time
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             roc_curve, matthews_corrcoef, cohen_kappa_score, confusion_matrix,auc)
import cv2
import numpy as np
from torch import optim
import pandas as pd
from Models.ViT_models.ViT import VisionTransformer
from Models.ViT_models.ViT_model_modules import ViT_Net
import random
from Models.ViT_models.ViT_model_modules import creating_ViT
# from Models.Mixer_models.models.Mixer_model_modules import creating_Mixer
import warnings
from tqdm import tqdm
from PIL import Image
warnings.filterwarnings('ignore')


########################## seed_function #########################
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


########################## learning functions #########################
def cnn_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-4
    elif epoch < 75:
        lr = 2e-5
    else:
        lr = 1e-6
    return lr


def vit_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-5
    elif epoch < 75:
        lr = 5e-6
    else:
        lr = 1e-6
    return lr

def mamba_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-6
    elif epoch < 75:
        lr = 5e-7
    else:
        lr = 1e-7
    return lr

def vit_lr_for_breast_schedule(epoch):
    if epoch < 50:
        lr = 2e-5
    elif epoch < 75:
        lr = 1e-5
    else:
        lr = 1e-6
    return lr


def fusion_lr_schedule(epoch):
    if epoch < 50:
        lr = 5e-5
    elif epoch < 75:
        lr = 2e-6
    else:
        lr = 1e-6
    return lr


def one_hot(org_x=None, pre_dim=3):
    one_x = np.zeros((org_x.shape[0], pre_dim))
    for i in range(org_x.shape[0]):
        one_x[i, int(org_x[i])] = 1
    return one_x



def view_results_for_picmil_vig_parallel(mil_feature=None, mil_head=None, train_loader=None, data_parallel=False,
                          loss_fn=None, proba_mode=False, gpu_device=None, proba_value=0.85,
                          batch_size=4, bags_len=100, abla_type='tic'):
    mil_feature.eval()
    mil_head.eval()
    train_acc = []
    train_loss = []
    for train_img_list, train_label in tqdm(train_loader):
        train_label = train_label.cuda()
        with torch.no_grad():
            train_pre_y = torch.zeros((1, mil_head.num_features)).cuda()
            for train_img in train_img_list:
                train_pre_y = torch.cat((train_pre_y, mil_feature(train_img.cuda())))
            train_pre_y = train_pre_y[1:]
            if abla_type == 'tic':
                train_pre_y, _, _ = mil_head(train_pre_y)
            else:
                train_pre_y = mil_head(train_pre_y)
            train_loss.append(loss_fn(train_pre_y, train_label).detach().cpu().numpy())
            if proba_mode == True:
                train_pre_y = torch.softmax(train_pre_y, dim=1)
                train_pre_label_proba = torch.argmax(train_pre_y, dim=1)
                for proba_in in range(train_pre_label_proba.shape[0]):
                    if train_pre_y[proba_in, train_pre_label_proba[proba_in]] < torch.tensor(proba_value).cuda():
                        train_pre_label_proba[proba_in] = torch.tensor(3).cuda()
                train_pre_label = train_pre_label_proba
            elif proba_mode == False:
                train_pre_label = torch.argmax(train_pre_y, dim=1)
            else:
                print('error! Please select probability mode!!!')
                break

        train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                        train_pre_label.detach().cpu().numpy()))
    return train_acc, train_loss, train_label, train_pre_label




def view_results_for_picmil_mamba_parallel(mil_feature=None, mil_head=None, train_loader=None, data_parallel=False,
                          loss_fn=None, proba_mode=False, gpu_device=None, proba_value=0.85,
                          batch_size=4, bags_len=100, abla_type='tic'):
    mil_feature.eval()
    mil_head.eval()
    train_acc = []
    train_loss = []
    for train_img_list, train_label in tqdm(train_loader):
        train_label = train_label.cuda()
        with torch.no_grad():
            train_pre_y = torch.zeros((1, mil_head.num_features)).cuda()
            for train_img in train_img_list:
                train_pre_y = torch.cat((train_pre_y, mil_feature(train_img.cuda())))
            train_pre_y = train_pre_y[1:]
            if abla_type == 'tic':
                train_pre_y, _, _ = mil_head(train_pre_y)
            elif abla_type == 'baseline':
                train_pre_y = mil_head(train_pre_y)
            train_loss.append(loss_fn(train_pre_y, train_label).detach().cpu().numpy())
            if proba_mode == True:
                train_pre_y = torch.softmax(train_pre_y, dim=1)
                train_pre_label_proba = torch.argmax(train_pre_y, dim=1)
                for proba_in in range(train_pre_label_proba.shape[0]):
                    if train_pre_y[proba_in, train_pre_label_proba[proba_in]] < torch.tensor(proba_value).cuda():
                        train_pre_label_proba[proba_in] = torch.tensor(3).cuda()
                train_pre_label = train_pre_label_proba
            elif proba_mode == False:
                train_pre_label = torch.argmax(train_pre_y, dim=1)
            else:
                print('error! Please select probability mode!!!')
                break

        train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                        train_pre_label.detach().cpu().numpy()))
    return train_acc, train_loss, train_label, train_pre_label




def view_results_for_pacmil_parallel(mil_feature=None, mil_head=None, train_loader=None,
                          loss_fn=None, proba_mode=False, gpu_device=None, proba_value=0.85,
                          batch_size=4, bags_len=100, abla_type='tic'):
    mil_feature.eval()
    mil_head.eval()
    train_acc = []
    train_loss = []
    for train_img_list, train_label in tqdm(train_loader):
        train_label = train_label.cuda()
        with torch.no_grad():
            train_pre_y = torch.zeros((1, 768)).cuda()
            for train_img in train_img_list:
                train_pre_y = torch.cat((train_pre_y, mil_feature(train_img.cuda())))
            train_pre_y = train_pre_y[1:]
            if abla_type == 'tic' or abla_type == 'sota':
                train_pre_y, _, _ = mil_head(train_pre_y)
            elif abla_type == 'baseline':
                train_pre_y = mil_head(train_pre_y)
            train_loss.append(loss_fn(train_pre_y, train_label).detach().cpu().numpy())
            if proba_mode == True:
                train_pre_y = torch.softmax(train_pre_y, dim=1)
                train_pre_label_proba = torch.argmax(train_pre_y, dim=1)
                for proba_in in range(train_pre_label_proba.shape[0]):
                    if train_pre_y[proba_in, train_pre_label_proba[proba_in]] < torch.tensor(proba_value).cuda():
                        train_pre_label_proba[proba_in] = torch.tensor(3).cuda()
                train_pre_label = train_pre_label_proba
            elif proba_mode == False:
                train_pre_label = torch.argmax(train_pre_y, dim=1)
            else:
                print('error! Please select probability mode!!!')
                break

        train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                        train_pre_label.detach().cpu().numpy()))


    return train_acc, train_loss, train_label, train_pre_label





def testing_for_pacmil_parallel(mil_feature=None, mil_head=None, train_loader=None,class_num=None,
                           loss_fn=None, proba_mode=False, gpu_device=None, proba_value=0.85, roc_save_path=None,
                           batch_size=4, bags_len=100, val_loader=None, test_loader=None, abla_type=None):
    loss_fn = nn.CrossEntropyLoss()

    # train_acc, train_loss, _, _ = view_results_for_pacmil_parallel(mil_feature=mil_feature, mil_head=mil_head,
    #                                                     train_loader=train_loader, data_parallel=data_parallel,
    #                                                     loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=None,
    #                                                     proba_value=proba_value, batch_size=batch_size,
    #                                                     bags_len=bags_len)
    # val_acc, val_loss, _, _ = view_results_for_pacmil_parallel(mil_feature=mil_feature, mil_head=mil_head,
    #                                                 train_loader=val_loader,
    #                                                 loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=None,
    #                                                 proba_value=proba_value, batch_size=batch_size,
    #                                                 bags_len=bags_len, abla_type=abla_type)
    # test_acc, test_loss, _, _ = view_results_for_pacmil_parallel(mil_feature=mil_feature, mil_head=mil_head,
    #                                                   train_loader=test_loader,
    #                                                   loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=None,
    #                                                   proba_value=proba_value, batch_size=batch_size,
    #                                                   bags_len=bags_len, abla_type=abla_type)
    # print(' val_acc:{:.4}'.format(np.mean(val_acc)))
    # print(' test_acc:{:.4}'.format(np.mean(test_acc)))

    ## ablation details

    mil_feature.eval()
    mil_head.eval()
    test_acc = []
    sum_label = torch.zeros(2).cuda(gpu_device)
    pre_y_sum = torch.zeros(2,class_num).cuda(gpu_device)
    pre_label = torch.zeros(2).cuda(gpu_device)
    for img_list, label in tqdm(test_loader):
        label = label.cuda()
        with torch.no_grad():
            pre_y = torch.zeros((1, 768)).cuda()
            for img in img_list:
                pre_y = torch.cat((pre_y, mil_feature(img.cuda())))
            pre_y = pre_y[1:]
            if abla_type == 'tic' or abla_type == 'sota':
                pre_y, _, _ = mil_head(pre_y)
            elif abla_type == 'baseline':
                pre_y = mil_head(pre_y)
            if proba_mode == True:
                pre_y = torch.softmax(pre_y, dim=1)
                train_pre_label_proba = torch.argmax(pre_y, dim=1)
                for proba_in in range(train_pre_label_proba.shape[0]):
                    if pre_y[proba_in, train_pre_label_proba[proba_in]] < torch.tensor(proba_value).cuda():
                        train_pre_label_proba[proba_in] = torch.tensor(3).cuda()
                train_pre_label = train_pre_label_proba
            elif proba_mode == False:
                test_pre_label = torch.argmax(pre_y, dim=1)
            else:
                print('error! Please select probability mode!!!')
                break
            test_acc.append(accuracy_score(label.detach().cpu().numpy(),
                                        test_pre_label.detach().cpu().numpy()))
            pre_label = torch.cat((pre_label, test_pre_label))
            sum_label = torch.cat((sum_label, label))
            pre_y_sum = torch.cat((pre_y_sum, pre_y))
    pre_label = pre_label[2:]
    pre_y_sum = torch.softmax(pre_y_sum[2:],dim=1)
    sum_label = sum_label[2:]
    print('-----------------------------------------------------------------------')
    print(' test_acc:{:.4}'.format(np.mean(test_acc)))
    print('-----------------------------------------------------------------------')
    print('classification_report:', '\n',
          classification_report(sum_label.cpu().numpy(), pre_label.cpu().numpy(), digits=4))
    print('-----------------------------------------------------------------------')
    print('AUC:',
          roc_auc_score(to_category(sum_label, class_num=class_num).ravel(),
                        pre_y_sum.cpu().numpy().ravel()))
    print('-----------------------------------------------------------------------')
    print('MCC:', matthews_corrcoef(sum_label.cpu().numpy(), pre_label.cpu().numpy()))
    print('kappa:', cohen_kappa_score(sum_label.cpu().numpy(), pre_label.cpu().numpy()))
    print('confusion matrix:',confusion_matrix(sum_label.cpu().numpy(), pre_label.cpu().numpy()))
    fpr, tpr, _ = roc_curve(to_category(sum_label, class_num=class_num).ravel(),
                        pre_y_sum.cpu().numpy().ravel())
    print(auc(fpr, tpr))
    #
    # write_dict = {'fpr': fpr, 'tpr': tpr}
    # roc_pd = pd.DataFrame(write_dict)
    # roc_pd.to_csv(roc_save_path, index=False)

def to_category(label_tensor=None, class_num=3):
    label_tensor = label_tensor.cpu().numpy()
    label_inter = np.zeros((label_tensor.size, class_num))
    for i in range(label_tensor.size):
        label_inter[i, int(label_tensor[i])] = 1
    return label_inter



########################## single-out-parallel fitting function #########################
#### ddai_net:
#### train_loader:
#### val_loader:
#### test_loader:
#### epoch:
#### gpu_device:
#### train_mode:
def training_for_pacmil_parallel(mil_feature=None, mil_head=None, train_loader=None, val_loader=None, test_loader=None,
                            proba_mode=False, lr_fn=None, epoch=100, gpu_device=0, onecycle_mr=1e-2, proba_value=0.85,
                            weight_path=r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx\SwinT_1.pth',
                            batch_size=4, bags_len=100, max_input_len=None, current_lr=None,abla_type='tic'):
    loss_fn = nn.CrossEntropyLoss()
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    mil_paras = [{'params': mil_feature.parameters()},
                 {'params': mil_head.parameters()}]
    print('########################## training results #########################')

    print('')
    print(f'###################### training in {abla_type}##########################')

    if lr_fn == 'onecycle':
        rmp_optim = torch.optim.AdamW(mil_paras, lr=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(rmp_optim, max_lr=onecycle_mr,
                                                  epochs=epoch, steps_per_epoch=len(train_loader))

    scaler = GradScaler()
    best_val_acc = 0
    best_epoch = 0
    for i in range(epoch):
        start_time = time.time()
        if lr_fn == 'vit':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=vit_lr_schedule(i))
        elif lr_fn == 'cnn':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=cnn_lr_schedule(i))
        elif lr_fn == 'vit_for_breast':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=vit_lr_for_breast_schedule(i))
        elif lr_fn == 'onecycle':
            pass
        elif lr_fn == 'searching_best_lr':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=current_lr)
        else:
            print('erorr!!!!')
            return 0

        mil_feature.train()
        mil_head.train()
        for img_data_list, img_label in tqdm(train_loader):
            # torch.autograd.set_detect_anomaly(True)
            img_label = img_label.cuda()
            pre_y = torch.zeros((1, 768)).cuda()

            # 从这里开始，每一个bag中的实例长度不一致，因此batchSize只能为1，img_data_list只有一个bag
            for img_data in img_data_list:

                # img_data = img_data.cuda()
                # cur_tensor = mil_feature(img_data)
                # pre_y = torch.cat((pre_y, cur_tensor))
                # torch.cuda.empty_cache()

                # max_input_len截断，分步载入
                if img_data.shape[0] <= max_input_len:
                    pre_y = torch.cat((pre_y, mil_feature(img_data.cuda())))
                else:
                    group_count = int(img_data.shape[0] / max_input_len)
                    for img_data_i in range(group_count):
                        # print(pre_y.shape)
                        groupIn_y = mil_feature(
                            img_data[max_input_len * img_data_i:(max_input_len * (img_data_i + 1)), :, :, :].cuda())
                        # print(kkk.shape)

                        pre_y = torch.cat((pre_y, groupIn_y))
                        torch.cuda.empty_cache()
                    if group_count * max_input_len < img_data.shape[0]:
                        remain_y = mil_feature(img_data[group_count * max_input_len:, :, :, :].cuda())
                        pre_y = torch.cat((pre_y, remain_y))
                    else:
                        pass

            pre_y = pre_y[1:]
            if abla_type == 'tic' or abla_type == 'sota':
                pre_y, min_dis, non_min_dis = mil_head(pre_y)
                # pdb.set_trace()
                loss_value = loss_fn(pre_y, img_label)
            elif abla_type == 'baseline':
                pre_y = mil_head(pre_y)
                loss_value = loss_fn(pre_y, img_label)
            #
            loss_value.backward()
            rmp_optim.step()
            rmp_optim.zero_grad()

            # with autocast():
            #     # 从这里开始，每一个bag中的实例长度不一致，因此batchSize只能为1，img_data_list只有一个bag
            #     for img_data in img_data_list:
            #
            #         # img_data = img_data.cuda()
            #         # cur_tensor = mil_feature(img_data)
            #         # pre_y = torch.cat((pre_y, cur_tensor))
            #         # torch.cuda.empty_cache()
            #
            #         # max_input_len截断，分步载入
            #         if img_data.shape[0] <= max_input_len:
            #             pre_y = torch.cat((pre_y, mil_feature(img_data.cuda())))
            #         else:
            #             group_count = int(img_data.shape[0] / max_input_len)
            #             for img_data_i in range(group_count):
            #                 # print(pre_y.shape)
            #                 groupIn_y = mil_feature(
            #                     img_data[max_input_len * img_data_i:(max_input_len * (img_data_i + 1)), :, :, :].cuda())
            #                 # print(kkk.shape)
            #
            #                 pre_y = torch.cat((pre_y, groupIn_y))
            #                 torch.cuda.empty_cache()
            #             if group_count * max_input_len < img_data.shape[0]:
            #                 remain_y = mil_feature(img_data[group_count * max_input_len:, :, :, :].cuda())
            #                 pre_y = torch.cat((pre_y, remain_y))
            #             else:
            #                 pass
            #
            #     pre_y = pre_y[1:]
            #     if abla_type == 'tic':
            #         pre_y, min_dis, non_min_dis = mil_head(pre_y)
            #         # pdb.set_trace()
            #         loss_value = loss_fn(pre_y, img_label) + min_dis - non_min_dis
            #     elif abla_type == 'baseline':
            #         pre_y = mil_head(pre_y)
            #         loss_value = loss_fn(pre_y, img_label)
            #
            # scaler.scale(loss_value).backward()
            # scaler.step(rmp_optim)
            # scaler.update()
            #
            # rmp_optim.zero_grad()

        # print(ddai_net.w)
        print('')
        print('viewing Results..............')
        # train_acc, train_loss, _, _ = view_results_for_pacmil_parallel(mil_feature=mil_feature, mil_head=mil_head,
        #                                                     train_loader=train_loader, loss_fn=loss_fn,
        #                                                     proba_mode=proba_mode, gpu_device=gpu_device,
        #                                                     proba_value=proba_value, batch_size=batch_size,
        #                                                     bags_len=bags_len, abla_type=abla_type)

        val_acc, val_loss, _, _ = view_results_for_pacmil_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                        train_loader=val_loader, loss_fn=loss_fn,
                                                        proba_mode=proba_mode, gpu_device=gpu_device,
                                                        proba_value=proba_value, batch_size=batch_size,
                                                        bags_len=bags_len, abla_type=abla_type)

        end_time = time.time()

        # print('epoch ' + str(i + 1),
        #       ' Time:{:.3}'.format(end_time - start_time),
        #       ' train_loss:{:.4}'.format(np.mean(train_loss)),
        #       ' train_acc:{:.4}'.format(np.mean(train_acc)),
        #       ' val_loss:{:.4}'.format(np.mean(val_loss)),
        #       ' val_acc:{:.4}'.format(np.mean(val_acc)))
        # print('')

        print('epoch ' + str(i + 1),
              ' Time:{:.3}'.format(end_time - start_time),
              ' val_loss:{:.4}'.format(np.mean(val_loss)),
              ' val_acc:{:.4}'.format(np.mean(val_acc)))
        print('')
        # save best
        cur_val_acc = np.mean(val_acc)
        if cur_val_acc > 0.9:
            best_val_acc = cur_val_acc
            best_epoch = i + 1
            g = mil_feature.state_dict()
            torch.save(g, os.path.join(weight_path,
                                       f'Simple_SwinT_{mil_head.abla_type}_Feature_ValAcc_{best_val_acc}_Epoch{best_epoch}_Seed{mil_head.seed}.pth'))
            g_1 = mil_head.state_dict()
            torch.save(g_1, os.path.join(weight_path,
                                         f'Simple_SwinT_{mil_head.abla_type}_Head_ValAcc_{best_val_acc}_Epoch{best_epoch}_Seed{mil_head.seed}.pth'))
        print(f'Best model is saved at Epoch:{best_epoch} with Val_Acc:{best_val_acc}')
        print('')


    test_acc, test_loss, _, _ = view_results_for_pacmil_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                      train_loader=test_loader, loss_fn=loss_fn,
                                                      proba_mode=proba_mode, gpu_device=gpu_device,
                                                      proba_value=proba_value, batch_size=batch_size,
                                                      bags_len=bags_len, abla_type=abla_type)


    print('########################## testing results #########################')
    # print('train_acc:{:.4}'.format(np.mean(train_acc)),
    #       ' val_acc:{:.4}'.format(np.mean(val_acc)),
    #       ' test_acc:{:.4}'.format(np.mean(test_acc)))
    print(
          ' test_acc:{:.4}'.format(np.mean(test_acc)))
    g = mil_feature.state_dict()
    torch.save(g, os.path.join(weight_path,
                                f'Simple_SwinT_{mil_head.abla_type}_Feature_TestAcc_{np.mean(test_acc)}_final_Seed{mil_head.seed}.pth'))
    g_1 = mil_head.state_dict()
    torch.save(g_1, os.path.join(weight_path,
                                    f'Simple_SwinT_{mil_head.abla_type}_Head_TestAcc_{np.mean(test_acc)}_finalSeed{mil_head.seed}.pth'))

    return test_acc

def training_for_picmil_mambda_parallel(mil_feature=None, mil_head=None, train_loader=None, val_loader=None, test_loader=None,
                            proba_mode=False, lr_fn=None, epoch=100, gpu_device=0, onecycle_mr=1e-2, proba_value=0.85,
                            weight_path=r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx\SwinT_1.pth',
                            batch_size=4, bags_len=100, weight_head_path=None, current_lr=None,abla_type='tic', use_amp=True):
    loss_fn = nn.CrossEntropyLoss()
    mil_paras = [{'params': mil_feature.parameters()},
                 {'params': mil_head.parameters()}]
    # rmp_optim = optim.SGD(mil_paras, lr=0.01, momentum=0.9, weight_decay=5e-3)
    # # 学习率调整策略 MultiStep：
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=rmp_optim,
    #                                            milestones=[int(100 * 0.56), int(100 * 0.78)],
    #                                            gamma=0.1, last_epoch=-1)
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()

    print('########################## training results #########################')
    if lr_fn == 'onecycle':
        rmp_optim = torch.optim.AdamW(mil_paras, lr=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(rmp_optim, max_lr=onecycle_mr,
                                                  epochs=epoch, steps_per_epoch=len(train_loader))
    print('')
    print(f'###################### training in {abla_type}##########################')
    # rmp_optim = torch.optim.AdamW(mil_paras, lr=1e-4, weight_decay=0.05)
    scaler = GradScaler()
    best_val_acc = 0
    best_epoch = 0
    for i in range(epoch):
        accumulattion_step = 2
        start_time = time.time()
        if lr_fn == 'mamba':
            rmp_optim = torch.optim.AdamW(mil_paras, lr=mamba_lr_schedule(i))
        elif lr_fn == 'vit':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=vit_lr_schedule(i))
        elif lr_fn == 'cnn':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=cnn_lr_schedule(i))
        elif lr_fn == 'vit_for_breast':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=vit_lr_for_breast_schedule(i))
        elif lr_fn == 'onecycle':
            pass
        elif lr_fn == 'searching_best_lr':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=current_lr)
        else:
            print('erorr!!!!')

        mil_feature.train()
        mil_head.train()
        for img_data_list, img_label in tqdm(train_loader):
            img_label = img_label.cuda()
            pre_y = torch.zeros((1, mil_head.num_features)).cuda()
            rmp_optim.zero_grad()

            if use_amp:
                # 混合精度训练
                with autocast():
                    for img_data in img_data_list:
                        img_data = img_data.cuda()
                        cur_tensor = mil_feature(img_data)
                        pre_y = torch.cat((pre_y, cur_tensor))
                        torch.cuda.empty_cache()
                    pre_y = pre_y[1:]
                    if abla_type == 'tic':
                        pre_y, min_dis, non_min_dis = mil_head(pre_y)
                        # pdb.set_trace()
                        loss_value = loss_fn(pre_y, img_label) + min_dis - non_min_dis
                    elif abla_type == 'baseline':
                        # pdb.set_trace()
                        pre_y = mil_head(pre_y)
                        loss_value = loss_fn(pre_y, img_label)


                scaler.scale(loss_value).backward()
                scaler.step(rmp_optim)
                scaler.update()

            else:
                for img_data in img_data_list:
                    img_data = img_data.cuda()
                    cur_tensor = mil_feature(img_data)
                    pre_y = torch.cat((pre_y, cur_tensor))
                    torch.cuda.empty_cache()
                    pass
                pre_y = pre_y[1:]

                if abla_type == 'tic':
                    pre_y, min_dis, non_min_dis = mil_head(pre_y)
                    # pdb.set_trace()
                    loss_value = loss_fn(pre_y, img_label) + min_dis - non_min_dis
                elif abla_type == 'baseline':
                    pre_y = mil_head(pre_y)
                    loss_value = loss_fn(pre_y, img_label)

                loss_value.backward()
                rmp_optim.step()
                rmp_optim.zero_grad()
        # 更新学习率并查看当前学习率
        # scheduler.step()
        # print('\t last_lr:', scheduler.get_last_lr())





        # print(ddai_net.w)
        print('')
        print('viewing Results..............')
        train_acc, train_loss, _, _ = view_results_for_picmil_mamba_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                            train_loader=train_loader, loss_fn=loss_fn,
                                                            proba_mode=proba_mode, gpu_device=gpu_device,
                                                            proba_value=proba_value, batch_size=batch_size,
                                                            bags_len=bags_len,abla_type=abla_type)

        val_acc, val_loss, _, _ = view_results_for_picmil_mamba_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                        train_loader=val_loader, loss_fn=loss_fn,
                                                        proba_mode=proba_mode, gpu_device=gpu_device,
                                                        proba_value=proba_value, batch_size=batch_size,
                                                        bags_len=bags_len,abla_type=abla_type)

        end_time = time.time()
        torch.cuda.empty_cache()
        print('epoch ' + str(i + 1),
              ' Time:{:.3}'.format(end_time - start_time),
              ' train_loss:{:.4}'.format(np.mean(train_loss)),
              ' train_acc:{:.4}'.format(np.mean(train_acc)),
              ' val_loss:{:.4}'.format(np.mean(val_loss)),
              ' val_acc:{:.4}'.format(np.mean(val_acc)))
        print('')
        cur_val_acc = np.mean(val_acc)
        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc
            best_epoch = i + 1
            g = mil_feature.state_dict()
            torch.save(g, os.path.join(weight_path,f'Cervix_WSI_Mamba3BaselineFeature_ValAcc_{best_val_acc}_Epoch{best_epoch}.pth'))
            g_1 = mil_head.state_dict()
            torch.save(g_1, os.path.join(weight_path,f'Cervix_WSI_Mamba3BaselineHead_ValAcc_{best_val_acc}_Epoch{best_epoch}.pth'))

        print(f'Best model is saved at Epoch:{best_epoch} with Val_Acc:{best_val_acc}')
        print('')

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    test_acc, test_loss, _, _ = view_results_for_picmil_mamba_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                      train_loader=test_loader, loss_fn=loss_fn,
                                                      proba_mode=proba_mode, gpu_device=gpu_device,
                                                      proba_value=proba_value, batch_size=batch_size,
                                                      bags_len=bags_len,abla_type=abla_type)

    print('########################## testing results #########################')
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))

    # g = mil_feature.state_dict()
    # torch.save(g, weight_path)
    # g_1 = mil_head.state_dict()
    # torch.save(g_1, weight_head_path)

    return test_acc


def training_for_picmil_vig_parallel(mil_feature=None, mil_head=None, train_loader=None, val_loader=None, test_loader=None,
                            proba_mode=False, lr_fn=None, epoch=100, gpu_device=0, onecycle_mr=1e-2, proba_value=0.85,
                            weight_path=r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx\SwinT_1.pth',
                            batch_size=4, bags_len=100, weight_head_path=None, current_lr=None,abla_type='tic', use_amp=True):
    loss_fn = nn.CrossEntropyLoss()
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    mil_paras = [{'params': mil_feature.parameters()},
                 {'params': mil_head.parameters()}]
    print('########################## training results #########################')
    if lr_fn == 'onecycle':
        rmp_optim = torch.optim.AdamW(mil_paras, lr=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(rmp_optim, max_lr=onecycle_mr,
                                                  epochs=epoch, steps_per_epoch=len(train_loader))
    scaler = GradScaler()
    best_val_acc = 0
    best_epoch = 0
    for i in range(epoch):
        start_time = time.time()
        if lr_fn == 'mamba':
            rmp_optim = torch.optim.AdamW(mil_paras, lr=vit_lr_schedule(i),weight_decay=0.05)
        elif lr_fn == 'vit':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=vit_lr_schedule(i))
        elif lr_fn == 'cnn':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=cnn_lr_schedule(i))
        elif lr_fn == 'vit_for_breast':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=vit_lr_for_breast_schedule(i))
        elif lr_fn == 'onecycle':
            pass
        elif lr_fn == 'searching_best_lr':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=current_lr)
        else:
            print('erorr!!!!')
            return 0

        mil_feature.train()
        mil_head.train()
        for img_data_list, img_label in tqdm(train_loader):
            img_label = img_label.cuda()
            pre_y = torch.zeros((1, mil_head.num_features)).cuda()

            if use_amp:
                # 混合精度训练
                with autocast():
                    for img_data in img_data_list:
                        img_data = img_data.cuda()
                        cur_tensor = mil_feature(img_data)
                        pre_y = torch.cat((pre_y, cur_tensor))
                        torch.cuda.empty_cache()
                    pre_y = pre_y[1:]
                    if abla_type == 'tic':
                        pre_y, min_dis, non_min_dis = mil_head(pre_y)
                        # pdb.set_trace()
                        loss_value = loss_fn(pre_y, img_label) + min_dis - non_min_dis
                    elif abla_type == 'baseline':
                        pre_y = mil_head(pre_y)
                        loss_value = loss_fn(pre_y, img_label)

                # # 检查梯度（应该是None，因为还没有进行反向传播）
                # print("Gradients before backward:")
                # for name, param in mil_head.named_parameters():
                #     print(name, param.grad)  # 应该输出None或者之前的梯度值（如果有的话）

                scaler.scale(loss_value).backward()
                scaler.step(rmp_optim)
                scaler.update()

                rmp_optim.zero_grad()
            else:
                for img_data in img_data_list:
                    img_data = img_data.cuda()
                    cur_tensor = mil_feature(img_data)
                    pre_y = torch.cat((pre_y, cur_tensor))
                    torch.cuda.empty_cache()

                pre_y = pre_y[1:]

                if abla_type == 'tic':
                    pre_y, min_dis, non_min_dis = mil_head(pre_y)
                    # pdb.set_trace()
                    loss_value = loss_fn(pre_y, img_label) + min_dis - non_min_dis
                elif abla_type == 'baseline':
                    pre_y = mil_head(pre_y)
                    loss_value = loss_fn(pre_y, img_label)

                loss_value.backward()
                rmp_optim.step()
                rmp_optim.zero_grad()
        # lr_sch.step()



        # print(ddai_net.w)
        print('viewing Results..............')
        train_acc, train_loss, _, _ = view_results_for_picmil_vig_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                            train_loader=train_loader, loss_fn=loss_fn,
                                                            proba_mode=proba_mode, gpu_device=gpu_device,
                                                            proba_value=proba_value, batch_size=batch_size,
                                                            bags_len=bags_len, abla_type=abla_type)

        val_acc, val_loss, _, _ = view_results_for_picmil_vig_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                        train_loader=val_loader, loss_fn=loss_fn,
                                                        proba_mode=proba_mode, gpu_device=gpu_device,
                                                        proba_value=proba_value, batch_size=batch_size,
                                                        bags_len=bags_len, abla_type=abla_type)

        end_time = time.time()
        torch.cuda.empty_cache()
        print('epoch ' + str(i + 1),
              ' Time:{:.3}'.format(end_time - start_time),
              ' train_loss:{:.4}'.format(np.mean(train_loss)),
              ' train_acc:{:.4}'.format(np.mean(train_acc)),
              ' val_loss:{:.4}'.format(np.mean(val_loss)),
              ' val_acc:{:.4}'.format(np.mean(val_acc)))
        print('')

        cur_val_acc = np.mean(val_acc)
        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc
            best_epoch = i + 1
            g = mil_feature.state_dict()
            torch.save(g, os.path.join(weight_path,
                                       f'Cervix_WSI_ViGBaselineFeature_ValAcc_{best_val_acc}_Epoch{best_epoch}.pth'))
            g_1 = mil_head.state_dict()
            torch.save(g_1, os.path.join(weight_path,
                                         f'Cervix_WSI_ViGBaselineHead_ValAcc_{best_val_acc}_Epoch{best_epoch}.pth'))

        print(f'Best model is saved at Epoch:{best_epoch} with Val_Acc:{best_val_acc}')
        print('')
        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    test_acc, test_loss, _, _ = view_results_for_picmil_mamba_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                      train_loader=test_loader, loss_fn=loss_fn,
                                                      proba_mode=proba_mode, gpu_device=gpu_device,
                                                      proba_value=proba_value, batch_size=batch_size,
                                                      bags_len=bags_len, abla_type=abla_type)

    print('########################## testing results #########################')
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))

    return test_acc








def view_results_for_baseline_parallel(mil_feature=None, mil_head=None, train_loader=None, data_parallel=False,
                          loss_fn=None, proba_mode=False, gpu_device=None, proba_value=0.85,
                          batch_size=4, bags_len=100):
    mil_feature.eval()
    mil_head.eval()
    train_acc = []
    train_loss = []
    for train_img_list, train_label in train_loader:
        train_label = train_label.cuda()
        with torch.no_grad():
            train_pre_y = torch.zeros((1, 768)).cuda()
            for train_img in train_img_list:
                train_pre_y = torch.cat((train_pre_y, mil_feature(train_img.cuda())))
            train_pre_y = train_pre_y[1:]
            train_pre_y = mil_head(train_pre_y)
            train_loss.append(loss_fn(train_pre_y, train_label).detach().cpu().numpy())
            if proba_mode == True:
                train_pre_y = torch.softmax(train_pre_y, dim=1)
                train_pre_label_proba = torch.argmax(train_pre_y, dim=1)
                for proba_in in range(train_pre_label_proba.shape[0]):
                    if train_pre_y[proba_in, train_pre_label_proba[proba_in]] < torch.tensor(proba_value).cuda():
                        train_pre_label_proba[proba_in] = torch.tensor(3).cuda()
                train_pre_label = train_pre_label_proba
            elif proba_mode == False:
                train_pre_label = torch.argmax(train_pre_y, dim=1)
            else:
                print('error! Please select probability mode!!!')
                break

        train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                        train_pre_label.detach().cpu().numpy()))
    return train_acc, train_loss, train_label, train_pre_label

def training_for_baseline_parallel(mil_feature=None, mil_head=None, train_loader=None, val_loader=None, test_loader=None,
                            proba_mode=False, lr_fn=None, epoch=100, gpu_device=0, onecycle_mr=1e-2, proba_value=0.85,
                            weight_path=r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx\SwinT_1.pth',
                            batch_size=4, bags_len=100, weight_head_path=None, current_lr=None):
    loss_fn = nn.CrossEntropyLoss()
    # ce_loss = nn.TripletMarginLoss()
    # rmp_optim = torch.optim.RMSprop(ddai_net.parameters(), lr=1e-6, weight_decay=0.0001)
    # scheduler = lr_scheduler.OneCycleLR(rmp_optim, max_lr=2e-5, epochs=500, steps_per_epoch=len(train_loader))
    # scheduler = lr_scheduler.StepLR(optimizer=rmp_optim, step_size=50, gamma=0.1)
    # torch.cuda.empty_cache()
    # torch.cuda.empty_cache()
    mil_paras = [{'params': mil_feature.parameters()},
                 {'params': mil_head.parameters()}]
    print('########################## training results #########################')
    if lr_fn == 'onecycle':
        rmp_optim = torch.optim.AdamW(mil_paras, lr=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(rmp_optim, max_lr=onecycle_mr,
                                                  epochs=epoch, steps_per_epoch=len(train_loader))
    for i in range(epoch):
        start_time = time.time()
        if lr_fn == 'vit':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=vit_lr_schedule(i))
        elif lr_fn == 'cnn':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=cnn_lr_schedule(i))
        elif lr_fn == 'vit_for_breast':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=vit_lr_for_breast_schedule(i))
        elif lr_fn == 'onecycle':
            pass
        elif lr_fn == 'searching_best_lr':
            rmp_optim = torch.optim.RMSprop(mil_paras, lr=current_lr)
        else:
            print('erorr!!!!')
            return 0

        mil_feature.train()
        mil_head.train()
        for img_data_list, img_label in train_loader:
            img_label = img_label.cuda()
            pre_y = torch.zeros((1, 768)).cuda()
            for img_data in img_data_list:
                pre_y = torch.cat((pre_y, mil_feature(img_data.cuda())))
            pre_y = pre_y[1:]

            pre_y = mil_head(pre_y)
            loss_value = loss_fn(pre_y, img_label)

            loss_value.backward()
            rmp_optim.step()
            rmp_optim.zero_grad()

        # print(ddai_net.w)
        train_acc, train_loss, _, _ = view_results_for_baseline_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                            train_loader=train_loader, loss_fn=loss_fn,
                                                            proba_mode=proba_mode, gpu_device=gpu_device,
                                                            proba_value=proba_value, batch_size=batch_size,
                                                            bags_len=bags_len)

        val_acc, val_loss, _, _ = view_results_for_baseline_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                        train_loader=val_loader, loss_fn=loss_fn,
                                                        proba_mode=proba_mode, gpu_device=gpu_device,
                                                        proba_value=proba_value, batch_size=batch_size,
                                                        bags_len=bags_len)

        end_time = time.time()
        print('epoch ' + str(i + 1),
              ' Time:{:.3}'.format(end_time - start_time),
              ' train_loss:{:.4}'.format(np.mean(train_loss)),
              ' train_acc:{:.4}'.format(np.mean(train_acc)),
              ' val_loss:{:.4}'.format(np.mean(val_loss)),
              ' val_acc:{:.4}'.format(np.mean(val_acc)))

        # write_1.add_scalar('train_acc',np.mean(train_acc), global_step = i)
        # write_1.add_scalar('train_loss', loss_value.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_loss', val_loss.detach().cpu().numpy(), global_step=i)
        # write_1.add_scalar('val_acc', np.mean(val_acc), global_step=i)

    test_acc, test_loss, _, _ = view_results_for_baseline_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                      train_loader=test_loader, loss_fn=loss_fn,
                                                      proba_mode=proba_mode, gpu_device=gpu_device,
                                                      proba_value=proba_value, batch_size=batch_size,
                                                      bags_len=bags_len)

    print('########################## testing results #########################')
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))

    g = mil_feature.state_dict()
    torch.save(g, weight_path)
    g_1 = mil_head.state_dict()
    torch.save(g_1, weight_head_path)

    return test_acc


def testing_for_parallel_baseline(mil_feature=None, mil_head=None, train_loader=None, data_parallel=False,
                           loss_fn=None, proba_mode=False, gpu_device=None, proba_value=0.85, roc_save_path=None,
                           batch_size=4, bags_len=100, val_loader=None, test_loader=None):
    loss_fn = nn.CrossEntropyLoss()
    train_acc, train_loss, _, _ = view_results_for_baseline_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                        train_loader=train_loader, data_parallel=data_parallel,
                                                        loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=None,
                                                        proba_value=proba_value, batch_size=batch_size,
                                                        bags_len=bags_len)
    val_acc, val_loss, _, _ = view_results_for_baseline_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                    train_loader=val_loader, data_parallel=data_parallel,
                                                    loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=None,
                                                    proba_value=proba_value, batch_size=batch_size,
                                                    bags_len=bags_len)
    test_acc, test_loss, _, _ = view_results_for_baseline_parallel(mil_feature=mil_feature, mil_head=mil_head,
                                                      train_loader=test_loader, data_parallel=data_parallel,
                                                      loss_fn=loss_fn, proba_mode=proba_mode, gpu_device=None,
                                                      proba_value=proba_value, batch_size=batch_size,
                                                      bags_len=bags_len)
    print('train_acc:{:.4}'.format(np.mean(train_acc)),
          ' val_acc:{:.4}'.format(np.mean(val_acc)),
          ' test_acc:{:.4}'.format(np.mean(test_acc)))
    print('train_loss:{:.4}'.format(np.mean(train_loss)),
          ' val_loss:{:.4}'.format(np.mean(val_loss)),
          ' test_loss:{:.4}'.format(np.mean(test_loss)))


def extracting_feat_for_ticmil(mil_feature=None, mil_head=None, train_loader=None,class_num=None,
                           loss_fn=None, proba_mode=False, gpu_device=None, proba_value=0.85,
                           batch_size=4, bags_len=100, val_loader=None, test_loader=None, abla_type=None):


    mil_feature.eval()
    mil_head.eval()

    sum_label = torch.zeros(2).cuda(gpu_device)
    sum_feat = []
    for img_list, label in tqdm(test_loader):
        label = label.cuda()
        with torch.no_grad():
            pre_y = torch.zeros((1, 768)).cuda()
            for img in img_list:
                pre_y = torch.cat((pre_y, mil_feature(img.cuda())))
            pre_y = pre_y[1:]
            pre_y = mil_head(pre_y)
            sum_feat.append(pre_y)
            sum_label = torch.cat((sum_label, label))
    sum_label = sum_label[2:]
    sum_feat = torch.cat(sum_feat,dim=0)

    # print(sum_label.shape)
    # print(sum_feat.shape)
    import umap
    import matplotlib.pyplot as plt
    import pandas as pd
    plt.rcParams['font.family'] = 'DejaVu Sans'

    sum_label = sum_label.cpu().numpy()
    sum_feat = sum_feat.cpu().numpy()


    # for cluster2D
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(sum_feat)

    dfs = []

    for label, name in zip([0, 1, 2], ['Grade I', 'Grade II', 'Grade III']):
        idx = sum_label == label
        data = embedding[idx]
        count = data.shape[0]

        df = pd.DataFrame({
            f'{name}_x': data[:, 0],
            f'{name}_y': data[:, 1],
        })
        dfs.append(df)

    # 为对齐行数，统一 DataFrame 长度（按最大行数补空）
    max_len = max(df.shape[0] for df in dfs)
    for i in range(len(dfs)):
        dfs[i] = dfs[i].reindex(range(max_len))  # 自动填 NaN

    # 拼接
    final_df = pd.concat(dfs, axis=1)

    # 保存
    final_df.to_csv('/data/MIL/TicMIL/Results/2d_cluster/sota.csv', index=False, encoding='utf-8')

    # for anova
    # reducer = umap.UMAP(n_components=1)  # 降维到1维
    # embedding = reducer.fit_transform(sum_feat)
    #
    # dfs = []
    #
    # for label, name in zip([0, 1, 2], ['Grade I', 'Grade II', 'Grade III']):
    #     idx = sum_label == label
    #     data = embedding[idx]
    #     count = data.shape[0]
    #
    #     df = pd.DataFrame({
    #         f'{name}_x': data[:, 0],  # 只保留一列
    #     })
    #     dfs.append(df)
    #
    # # 为对齐行数，统一 DataFrame 长度（按最大行数补空）
    # max_len = max(df.shape[0] for df in dfs)
    # for i in range(len(dfs)):
    #     dfs[i] = dfs[i].reindex(range(max_len))  # 自动填 NaN
    #
    # # 拼接
    # final_df = pd.concat(dfs, axis=1)
    # final_df.to_csv('/data/MIL/TicMIL/Results/anova/sota.csv', index=False, encoding='utf-8')


def interpret_bag_for_ticmil(mil_feature=None, mil_head=None, train_loader=None,class_num=None,
                           loss_fn=None, proba_mode=False, gpu_device=None, proba_value=0.85,
                           batch_size=4, bags_len=100, val_loader=None, test_loader=None, abla_type=None):


    mil_feature.eval()
    mil_head.eval()
    bag_weight_list = []
    for img_list, label in tqdm(test_loader):
        label = label.cuda()
        with torch.no_grad():
            pre_y = torch.zeros((1,768)).cuda()
            for img in img_list:
                pre_y = torch.cat((pre_y, mil_feature(img.cuda())))
            pre_y = pre_y[1:]
            bag_w = mil_head(pre_y)  # batch_size * 1042 * 1
            bag_weight_list.append(bag_w[:, :961, :])
    bag_weight_sum = torch.cat(bag_weight_list,dim=0).cpu().numpy() # 138 * 961 * 1

    count = 0
    for cate in ['I','II','III']:
        merged_img_dir = f'/data/MIL/TicMIL/Datasets/Cervix/WSI_org/{cate}'  # 拼接好的 bag 图像目录
        save_dir = f'/data/MIL/TicMIL/Results/interpret/{cate}'  # 热图叠加图保存路径

        bag_names = sorted(os.listdir(merged_img_dir))  # 默认和图片名字一一对应
        grid_size = 31
        tile_size = 96
        alpha = 0.3  # 热图透明度


        for idx, bag_img_name in tqdm(enumerate(bag_names), total=len(bag_names)):

            img_path = os.path.join(merged_img_dir, bag_img_name)
            weights = bag_weight_sum[count+idx].flatten()

            # 读取原图
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)

            # 处理权重为热图
            weights -= weights.min()
            weights /= weights.max() + 1e-8
            heatmap_small = weights.reshape(grid_size, grid_size)
            heatmap_large = cv2.resize(heatmap_small, (grid_size * tile_size, grid_size * tile_size),
                                       interpolation=cv2.INTER_CUBIC)
            heatmap_color = cv2.applyColorMap((heatmap_large * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            # 合并原图与热图
            overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)

            # 保存结果
            save_name = os.path.splitext(bag_img_name)[0] + '_overlay.jpg'
            save_path = os.path.join(save_dir, save_name)
            Image.fromarray(overlay).save(save_path)
        count += len(bag_names)




if __name__ == '__main__':
    max_lr = 1e-3
    min_lr = 1e-7
    max_boundary = -np.log10(max_lr)
    min_boundary = -np.log10(min_lr)

    change_log = 0
    for k in range(int(min_boundary - max_boundary) * 10):
        if k % 10 == 0:
            change_log += 1
        lr = (max_lr / (10 ** (change_log - 1))) - (k - (change_log - 1) * 10) * (max_lr / (10 ** change_log))
        print(lr)





