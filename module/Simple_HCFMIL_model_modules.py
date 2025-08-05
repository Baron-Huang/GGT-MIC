import time

from sympy.polys.polyconfig import query

########################## API Section #########################
from Models.SwinT_models.models.swin_transformer import SwinTransformer
from torch import nn
import torch
import random
import torch.nn.functional as F
import math
from Loss_functions.mmd import MMDLoss


class CrossAttnAggregate(nn.Module):
    def __init__(self, dropout):
        super(CrossAttnAggregate, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, y_guide, y_c):
        query = y_guide.clone()
        if isinstance(y_c,list):
            # Assign x1 and x2 to query and key

            key_c1 = y_c[0]
            key_c2 = y_c[1]
            d1 = key_c1.shape[-1]
            d2 = key_c2.shape[-1]

            scores = torch.mm(query, key_c1.transpose(0, 1)) / math.sqrt(d1)
            output_c1 = torch.mm(self.dropout(F.softmax(scores, dim=-1)), y_c[0])

            scores = torch.mm(query, key_c2.transpose(0, 1)) / math.sqrt(d2)
            output_c2 = torch.mm(self.dropout(F.softmax(scores, dim=-1)), y_c[1])
            return output_c1 + output_c2
        else:
            key = y_c.clone()
            scores = torch.mm(query, key.transpose(0, 1)) / math.sqrt(d)
            output = torch.mm(self.dropout(F.softmax(scores, dim=-1)), y_c)
            return output



class TgiClustering():
    def __init__(self, k_nums = 3, sel_dis = 'l1', train_iters = 50, p = 1):
        super(TgiClustering, self).__init__()
        self.k_nums = k_nums
        self.sel_dis = sel_dis
        self.train_iters = train_iters
        self.p = p

    def mmd_dis(self,x,y):

        mmd = MMDLoss()
        return mmd(x,y)

    def l2_distance(self, x, y):
        return torch.sqrt((x - y).permute(1, 0) @ (x - y))

    def l1_distance(self, x, y):
        return torch.sum(torch.abs(x - y))

    def lmax_distance(self, x, y):
        return torch.max(torch.abs(x - y))

    def lp_distance(self, x, y, p):
        lp_sum = 0
        for i in range(int(x.shape[0])):
            lp_sum += (x[i] - y[i]) ** p
        lp_sum = torch.abs(lp_sum) ** (1 / p)
        return lp_sum

    def init_cluster_centre(self, x, k_num, mode='random'):
        if mode == 'random':
            y_shape = x.shape[1]
            clus_center = torch.zeros((1, y_shape)).cuda()
            for i in range(k_num):
                clus_center_k = torch.zeros((1, 1)).cuda()
                for j in range(y_shape):
                    clus_center_inter = random.uniform(torch.max(x[:, j]), torch.min(x[:, j]))
                    clus_center_inter = torch.reshape(clus_center_inter, (1, 1)).cuda()
                    clus_center_k = torch.cat((clus_center_k, clus_center_inter), dim=1)
                clus_center_k = clus_center_k[:, 1:]
                clus_center = torch.cat((clus_center, clus_center_k))
            clus_center = clus_center[1:, :]
            return clus_center

        elif mode == 'kpp':
            dims = x.shape[1]
            init = torch.zeros((k_num, dims)).cuda()

            r = torch.distributions.uniform.Uniform(0, 1)
            for i in range(k_num):
                if i == 0:
                    init[i, :] = x[torch.randint(x.shape[0], [1])]

                else:
                    D2 = torch.cdist(init[:i, :][None, :], x[None, :], p=2)[0].amin(dim=0)
                    probs = D2 / torch.sum(D2)
                    cumprobs = torch.cumsum(probs, dim=0)
                    init[i, :] = x[torch.searchsorted(
                        cumprobs, r.sample([1]).cuda())]

            return init
        return None

    def init_cluster_centre_simple(self, x, k_num):
        y_shape = x.shape[1]
        clus_center = torch.randn((k_num, y_shape)).cuda()
        return clus_center

    def assign_data_point(self, x, init_cluster_cen):
        assigned_set = {}
        for i in range(init_cluster_cen.shape[0]):
            assigned_set[str(i)] = []

        for i in range(x.shape[0]):
            cont_dis = torch.zeros((1, 1)).cuda()
            for j in range(init_cluster_cen.shape[0]):
                if self.sel_dis == 'l2':
                    dis_value = \
                    self.l2_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1))
                elif self.sel_dis == 'l1':
                    dis_value = \
                    self.l1_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1))
                elif self.sel_dis == 'lp':
                    dis_value = \
                self.lp_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1), p=self.p)
                elif self.sel_dis == 'lmax':
                    dis_value = \
                    self.lmax_distance(x[i, :].reshape(x.shape[1], 1), init_cluster_cen[j, :].reshape(x.shape[1], 1))
                else:
                    pass
                cont_dis = torch.cat((cont_dis, dis_value.reshape(1, 1)))
            cont_dis = cont_dis[1:, :]
            max_id = torch.argmin(cont_dis).cpu().numpy()
            assigned_set[str(max_id)].append(i)
        return assigned_set

    def assign_data_point_mat_ver(self, x, init_cluster_cen):
        assigned_set = {}
        init_cluster_order_matrix = torch.zeros((x.shape[0], 1)).cuda()
        for i in range(init_cluster_cen.shape[0]):
            assigned_set[str(i)] = []
            x_y = x - init_cluster_cen[i].expand(x.shape[0], -1)
            x_y_2 = x_y ** 2
            xxx = torch.sum(x_y_2, dim=1)
            xxx_sqrt = torch.sqrt(xxx)
            xxx_sqrt = xxx_sqrt.reshape(xxx.shape[0], 1)
            init_cluster_order_matrix = torch.cat((init_cluster_order_matrix, xxx_sqrt), dim=1)
        init_cluster_order_matrix = init_cluster_order_matrix[:, 1:]
        init_cluster_order = torch.argmin(init_cluster_order_matrix, dim=1)
        for i in range(init_cluster_cen.shape[0]):
            k = torch.nonzero(init_cluster_order == torch.tensor(i)).detach().cpu().numpy()
            k = list(k.reshape((k.shape[0])))
            assigned_set[str(i)] = k
        return assigned_set

    def upgrade_cluster_centre(self, x, assigned_set):
        new_centre = torch.zeros((1, x.shape[1])).cuda()
        for i in range(len(assigned_set)):
            new_inter = torch.mean(x[assigned_set[str(i)], :], dim=0)
            new_centre = torch.cat((new_centre, new_inter.reshape(1, x.shape[1])))
        new_centre = new_centre[1:, :]
        return new_centre

    def forward(self, x):
        # k = self.k_nums
        clus_center = self.init_cluster_centre(x, self.k_nums,mode='kpp')
        for train_i in range(self.train_iters):
            assiged_set = self.assign_data_point_mat_ver(x, clus_center)
            # assiged_set = self.assign_data_point(x, clus_center)
            new_centre = self.upgrade_cluster_centre(x, assiged_set)
            if torch.mean(new_centre) == torch.mean(clus_center):
                break
            else:
                clus_center = new_centre
        #print('train_i:', train_i)
        return assiged_set



class TicMIL_Parallel_Feature(nn.Module):
    def __init__(self, base_model=None):
        super(TicMIL_Parallel_Feature, self).__init__()
        self.layers_0 = base_model.layers[0]
        self.layers_1 = base_model.layers[1]
        self.layers_2 = base_model.layers[2]
        self.layers_3 = base_model.layers[3]
        self.patch_embed = base_model.patch_embed
        self.pos_drop = base_model.pos_drop
        self.norm = base_model.norm
        self.avgp = nn.AvgPool1d(kernel_size=9, stride=9)  # only for 96*96 input
        # self.avgp = nn.AvgPool1d(kernel_size=49, stride=49)   # for 224* 224 input


    def forward(self, x):
        # with torch.no_grad():
        y = self.patch_embed(x)
        y = self.pos_drop(y)
        y = self.layers_0(y)
        y = self.layers_1(y)
        y = self.layers_2(y)
        y = self.layers_3(y)
        y = self.norm(y)
        y = self.avgp(y.permute(0, 2, 1))
        y = torch.reshape(y, (y.shape[0], y.shape[1]))
        return y

class TicMIL_Parallel_Head(nn.Module):
    def __init__(self, base_model = None, class_num = 3, seed=None, batch_size = 2, bags_len = 1042, model_stats = 'train',abla_type='tic',feat_extract=False, bag_weight=False):
        super(TicMIL_Parallel_Head, self).__init__()
        self.head = base_model.head
        self.head_2 = base_model.head
        self.batch_size = batch_size
        self.bags_len = bags_len
        self.feat_extract = feat_extract
        self.bag_weight = bag_weight
        self.seed = seed
        self.model_stats = model_stats
        self.abla_type = abla_type
        self.class_num = class_num

        self.tgi_clustering_block = TgiClustering(k_nums=3)
        self.cross_attn_aggregation = CrossAttnAggregate(dropout=0.1)



    def forward(self, x):

        if x.shape[0] / self.bags_len > 1 and self.abla_type == 'sota':

            # clustering guiding
            y = torch.reshape(x, (int(x.shape[0] / self.bags_len), self.bags_len, x.shape[1]))
            if self.bag_weight:
                bag_w = torch.mean(y,dim=2,keepdim=True)
                return bag_w

            #t_1 = time.time()
            final_y = torch.zeros(self.batch_size, 768).cuda()
            agg_y = torch.zeros(self.batch_size, 768).cuda()

            # divided into batch again
            for i in range(y.shape[0]):

                assigned_sets = self.tgi_clustering_block.forward(y[i][0:961, :])
                target_guiding_y = y[i][961:, :]
                assign_y_0 = y[i][0:961, :][assigned_sets['0'], :]
                assign_y_1 = y[i][0:961, :][assigned_sets['1'], :]
                assign_y_2 = y[i][0:961, :][assigned_sets['2'], :]

                # clustering guiding aggregate
                dis_0_tar = self.tgi_clustering_block.mmd_dis(target_guiding_y, assign_y_0)
                dis_1_tar = self.tgi_clustering_block.mmd_dis(target_guiding_y, assign_y_1)
                dis_2_tar = self.tgi_clustering_block.mmd_dis(target_guiding_y, assign_y_2)

                ##adaptive dis
                y[i][0:961, :][assigned_sets['0'], :] = (1/dis_0_tar) * y[i][0:961, :][assigned_sets['0'], :]
                y[i][0:961, :][assigned_sets['1'], :] = (1/dis_1_tar) * y[i][0:961, :][assigned_sets['1'], :]
                y[i][0:961, :][assigned_sets['2'], :] = (1/dis_2_tar) * y[i][0:961, :][assigned_sets['2'], :]

                ## del max and aggregate 
                max_ord = torch.argmax(torch.tensor([dis_0_tar, dis_1_tar, dis_2_tar]).cuda())
                min_list = []
                for j in range(3):
                    if max_ord == torch.tensor(j).cuda():
                        y[i][0:961, :][assigned_sets[str(j)], :] = 0
                    else:
                        min_list.append(j)

                attn_output = self.cross_attn_aggregation(target_guiding_y,[[assign_y_0, assign_y_1, assign_y_2][k] for k in min_list])
                agg_y[i] = torch.mean(attn_output, dim=0, keepdim=True)


                mask = (y[i] != 0).any(dim=1)
                var = y[i][mask]
                final_y[i] = torch.mean(var, dim=0, keepdim=True)

            y1 = self.head(final_y)
            y2 = self.head_2(agg_y)
            
            if self.feat_extract:
                return final_y
            else:
                return 0.8 * y1 + 0.2 * y2


