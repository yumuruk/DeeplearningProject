import os
import torch
import torch.nn as nn

from .net import *
from .model_util import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.

class Dual_Block(nn.Module):
    def __init__(self):
        super(Dual_Block, self).__init__()
        
        print("Loading Enhance Net...")
        Proximal_t = [RDN(3,1)]
        self.Proximal_t = nn.Sequential(*Proximal_t)
        
        self.t_1D_Net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=False),
            # nn.ReLU()
        )

        self.rho_1 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.rho_2 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.rho_3 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.rho_4 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.rho_5 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)
        self.rho_6 = nn.Parameter(torch.tensor([3.001]),requires_grad=True)


    def forward(self, I, t_p, B_p, B, t, J, Aux_J, Aux_t, Lag_J, Lag_t, Map_u, Map_v, DCP, BCP, patch_size = 35, eps = 1e-6):
        eta_0 = 1.0
        eta_1 = 0.7
        eta_2 = 0.3
        eta_3 = 1.0
        eta_4 = 1.0
        rho_1 = self.rho_1
        rho_2 = self.rho_2
        rho_3 = self.rho_3
        rho_4 = self.rho_4
        rho_5 = self.rho_5
        rho_6 = self.rho_6
        
        
        #### f_c_J regularization 시작 ####        
        sorted_channel_means, sorted_channel_indices, J_l, J_m, J_s, large_channel_idx, medium_channel_idx, small_channel_idx = extract_channel_statistics(J)
        J_l_bar = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(sorted_channel_means[:,2],1),1),1).to(DEVICE)
        J_m_bar = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(sorted_channel_means[:,1],1),1),1).to(DEVICE)
        J_s_bar = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(sorted_channel_means[:,0],1),1),1).to(DEVICE)
        
        # 동일하게 동작하는데 더 짧은 표현
        # J_l_bar = sorted_channel_means[:,2].view(-1,1,1,1).to(DEVICE)
        # J_m_bar = sorted_channel_means[:,1].view(-1,1,1,1).to(DEVICE)
        # J_s_bar = sorted_channel_means[:,0].view(-1,1,1,1).to(DEVICE)
        
        J_l = J_l.to(DEVICE)
        J_m = J_m.to(DEVICE)
        J_s = J_s.to(DEVICE)
        
        J_m = J_m + torch.mul(J_l_bar - J_m_bar, J_l)
        J_s = J_s + torch.mul(J_l_bar - J_s_bar, J_l)
        
        J = replace_channel_by_index(J.clone(), J_m.clone(), medium_channel_idx)
        J = replace_channel_by_index(J.clone(), J_s.clone(), small_channel_idx)
        ####### f_c_J regularization 끝 ####
        
        ###### 최적화 시작 ######
        ####### B부터 업데이트 시작 #######
        D = torch.ones(I.shape).to(DEVICE)  # D는 모든 요소가 1인 텐서로 초기화
        B = (eta_2*B_p - eta_0*(J*t - I)*(1-t))/(eta_0*(1.0 - t)*(1.0 - t) + eta_2 )
        B = torch.mean(B, (2,3), True)  # B는 전체 영상에 대해서 평균값으로 사용함
        # B = self.B_net(B) # B_net 도 시도해봤으나 의미 없어서 삭제한듯
        B = B*D
        ####### B 업데이트 끝 ########
        ####### t 업데이트 시작 ########
        t = (eta_2*t_p + rho_4*Aux_t - Lag_t - eta_0*(B - I)*(J - B))/(eta_0*(J - B)*(J - B) + eta_1 + rho_4)
        t = self.t_1D_Net(t)
        t = torch.cat((t, t, t), 1)  # t는 3채널로 확장
        ####### t 업데이트 끝 ########
        ####### J 업데이트 시작 ########
        J = (eta_0*(t*(I-B*(1.0 - t)) + rho_3*Aux_J - Lag_J + rho_5*DCP - rho_6*BCP + rho_6))/(eta_0*t*t + rho_3 + rho_5 + rho_6)
        ####### J 업데이트 끝 ########
        M_T_P = DCP
        M_T_Q = BCP
        DCP = (rho_1*M_T_P + rho_4*J)/(rho_1 + rho_4)
        BCP = (rho_2*M_T_Q - rho_5*J + rho_5)/(rho_2 + rho_5)
        
        Aux_t = self.Proximal_t(t + (1.0/rho_4)*Lag_t)
        
        Lag_J = Lag_J + rho_3*(J - Aux_J)
        Lag_t = Lag_t + rho_4*(t - Aux_t)
        
        M_u, index_map_dark = get_dark_channel(DCP, patch_size)
        M_v, index_map_dark = get_dark_channel(BCP, patch_size)
        
        Map_u = softThresh(M_u, eta_3/rho_1)
        Map_v = softThresh(M_v, eta_4/rho_2)
        # Q = P
        return B, t, J, Aux_J, Aux_t, Lag_J, Lag_t, Map_u, Map_v , DCP, BCP, rho_3
        
        
class IPMM(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=80, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(IPMM, self).__init__()
        act = nn.PReLU()
        self.shallow_feat2 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        self.r1 = nn.Parameter(torch.Tensor([0.5]))
        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.merge12=mergeblock(n_feat,3,True)
    
    def forward(self,x2_img, stage1_img,feat1,res1,x2_samfeats):
        ## PMM
        x2 = self.shallow_feat2(x2_img)
        x2_cat = self.merge12(x2, x2_samfeats)
        feat2,feat_fin2 = self.stage2_encoder(x2_cat, feat1, res1)
        res2 = self.stage2_decoder(feat_fin2,feat2)
        x3_samfeats, stage2_img = self.sam23(res2[-1], x2_img)
        return x3_samfeats, stage2_img, feat2, res2
        
class Dual_Net(nn.Module):
    def __init__(self, Block_number=5):
        super(Dual_Net, self).__init__()
        self.Block_number = Block_number
        block_list = []
        for i in range(self.Block_number):
            block_list.append(Dual_Block())
        self.enhance_net = nn.ModuleList(block_list)
        
        #### Proxiaml J module 초기화
        ### 하이퍼 파라미터 등 필요한 변수 설정
        n_feat=40; scale_unetfeats=20; kernel_size=3; reduction=4; bias=False
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4, csff=True)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats,depth=4)
        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.r1 = nn.Parameter(torch.Tensor([0.5]))

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.merge12=mergeblock(n_feat,3,True)
        
        self.Proximal_J = IPMM(in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False)

    def forward(self, I, t_p, B_p):
        
        bs, _, _, _ = I.shape
        B = torch.zeros((bs, 3, 1, 1)).to(DEVICE)  # B는 초기값으로 0으로 설정
        t = torch.zeros(I.shape).to(DEVICE)  # t는 초기값으로 0으로 설정
        J = I.to(DEVICE)  # J는 입력 이미지 I로 초기화
        
        
        Aux_J = torch.zeros(I.shape).to(DEVICE)  # G는 초기값으로 0으로 설정
        Aux_t = torch.zeros(I.shape).to(DEVICE)
        
        Lag_J = torch.zeros(I.shape).to(DEVICE)         
        Lag_t = torch.zeros(I.shape).to(DEVICE)      
        
        Map_u = torch.zeros(I.shape).to(DEVICE)
        Map_v = torch.zeros(I.shape).to(DEVICE)
        
        DCP = torch.zeros(I.shape).to(DEVICE)  # u는 초기값으로 0으로 설정
        BCP = torch.zeros(I.shape).to(DEVICE)
        
        list_J = []
        list_t = []
        list_B = []
        
        list_Aux_J = []
        list_Aux_t = []
        
        list_Lag_J = []
        list_Lag_t = []
        
        list_Map_u = []
        list_Map_v = []
        
        list_DCP = []
        list_BCP = []
        
        rho_3 = torch.tensor([3.001]).to(DEVICE)
        x1_img = J +(1.0/rho_3) * Lag_t
        x1 = self.shallow_feat1(x1_img)
        feat1, feat_fin1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat_fin1, feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], J)
        
        Aux_J = stage1_img        
        
        for j in range(self.Block_number):

            [B, t, J, Aux_J, Aux_t, Lag_J, Lag_t, Map_u, Map_v, DCP, BCP, rho_3] = self.enhance_net[j](I, t_p, B_p, B, t, J, Aux_J, Aux_t, Lag_J, Lag_t, Map_u, Map_v, DCP, BCP)
            
            ## IPMM module 
            ## Proximal J module
            
            img = J + (1.0/rho_3)*Lag_t
            x2_samfeats, stage1_img, feat1, res1 = self.Proximal_J(img, stage1_img,feat1,res1,x2_samfeats)
            Aux_J = stage1_img
            
            list_J.append(J)
            list_t.append(t)
            list_B.append(B)
            
            # list_Aux_J.append(Aux_J)
            # list_Aux_t.append(torch.cat((Aux_t, Aux_t, Aux_t), 1))
            
            # list_Lag_J.append(Lag_J)
            # list_Lag_t.append(Lag_t)
            
            # list_Map_u.append(torch.cat((Map_u, Map_u, Map_u), 1))
            # list_Map_v.append(torch.cat((Map_v, Map_v, Map_v), 1))
            
            # list_DCP.append(DCP)  
            # list_BCP.append(BCP)
            
        # return list_J, list_t, list_B, list_Aux_J, list_Aux_t, list_Lag_J, list_Lag_t, list_Map_u, list_Map_v, list_DCP, list_BCP
        return list_J, list_t, list_B