import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from utils import *
from multiprocessing.dummy import Pool as ThreadPool

import math
eps = 1e-12

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Spatial(nn.Module):
    def __init__(self, dim, k,A):
        super().__init__()

        self.k=k
        self.kk=k*k
        self.A=A
        self.kkA=self.kk*self.A

        self.ln = nn.LayerNorm(self.A)

        self.w=nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(self.kk, self.kk, 1)))
        self.b=nn.Parameter(torch.ones(self.kk))
        self.activate=nn.GELU()

    def forward(self, x):  # (50,49,2048)
        x = torch.squeeze(x, -1)
        input = x
        b, kkA= x.shape
        gate=x.view(b,self.kk,self.A)

        gate = self.ln(gate)
        gate = nn.functional.conv1d(gate, self.w, bias=self.b)
        gate = self.activate(gate)

        gate = gate.view(b,self.kkA)
        return gate+input


class Frequency(nn.Module):
    def __init__(self, dim, h=4):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        x = torch.squeeze(x)
        B, N, P = x.shape
        input = x
        x = x.to(torch.float32)

        x = torch.fft.rfft(x, dim=-2, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft(x, n=N, dim=-2, norm='ortho')

        return x+input

class GlobalRouting(nn.Module):
    def __init__(self, A, B, C, D, kernel_size=3, stride=1, padding=1, pose_out=False):
        super(GlobalRouting, self).__init__()
        self.A = A      #num_caps
        self.B = B      #num_caps
        self.C = C      #caps_size
        self.D = D      #caps_size

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A
        self.kkA2 = int(self.kkA / 2) + 1

        self.stride = stride
        self.pad = padding

        self.pose_out = pose_out
        self.mpose=nn.Linear(C,B*D)
        self.gelu=nn.GELU()
        self.softmax=nn.Softmax(dim=1)
        self.cpose2=nn.Linear(C,B)
        self.aLayerNorm=nn.LayerNorm(C)
        self.filter = Frequency(C, self.kkA2)
        self.spatinal = Spatial(B,self.k,self.A)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, a, pose):
        b, _, h, w = a.shape
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        pose = pose.view(b, self.A, self.C, self.kk, l)
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        pose = pose.view(b, l, self.kkA, self.C, 1)
        if self.pose_out:
            temp_pose=pose.view(b * l, self.kkA, self.C)
            pose_out = self.mpose(temp_pose)
            pose_out = self.gelu(pose_out)
            pose_outs=torch.split(pose_out.view(b*l,self.kkA,self.B,self.D),1,dim=-2)
            pose_out = torch.stack([self.filter(torch.squeeze(pose))
                                     for pose in pose_outs], dim=2).view(b*l,self.kkA,self.B*self.D)
            pose_out = pose_out.view(b, l, self.kkA, self.B, self.D)
        logit = self.cpose2(pose.view(b * l, self.kkA, self.C))
        logits = torch.split(logit,1,dim=-1)
        logit = logit + torch.stack([self.spatinal(torch.squeeze(gate,-1)) for gate in logits],dim=-1)
        r = torch.softmax(logit, dim=2)
        r = r.view(b,l,self.kkA,self.B)
        a = F.unfold(a, self.k, stride=self.stride, padding=self.pad)
        a = a.view(b, self.A, self.kk, l)
        a = a.permute(0, 3, 2, 1).contiguous()
        a = a.view(b, l, self.kkA, 1)
        ar = a * r
        ar_sum = ar.sum(dim=2, keepdim=True)
        coeff = (ar / (ar_sum)).unsqueeze(-1)
        a_out = ar_sum / a.sum(dim=2, keepdim=True)
        a_out = a_out.squeeze(2)
        a_out = a_out.transpose(1,2)
        if self.pose_out:
            pose_out = (coeff * pose_out).sum(dim=2)
            pose_out = pose_out.view(b, l, -1)
            pose_out = pose_out.transpose(1,2)
        oh = ow = math.floor(l**(1/2))
        a_out = a_out.view(b, -1, oh, ow)
        if self.pose_out:
            pose_out = pose_out.view(b, -1, oh, ow)
        else:
            pose_out = None
        return a_out, pose_out

class EmRouting2d(nn.Module):
    def __init__(self, A, B, caps_size, kernel_size=3, stride=1, padding=1, iters=3, final_lambda=1e-2):
        super(EmRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.psize = caps_size
        self.mat_dim = int(caps_size ** 0.5)

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A
        #edit externel
        self.kkA2 = int(self.kkA / 2) + 1

        self.stride = stride
        self.pad = padding

        self.iters = iters

        #self.W = nn.Parameter(torch.FloatTensor(self.kkA, B, self.mat_dim, self.mat_dim))
        #nn.init.kaiming_uniform_(self.W.data)

        #edit dylinear pose
        self.mpose = nn.Linear(self.psize, B * self.psize)
        #self.gelu = nn.GELU()
        #self.cpose2 = nn.Linear(self.psize, B)

        #eidt spatial
        self.spatinal = Spatial(B, self.k, self.A)

        #edit externel

        self.filter = Frequency(self.psize, self.kkA2)

        self.beta_u = nn.Parameter(torch.FloatTensor(1, 1, B, 1))
        self.beta_a = nn.Parameter(torch.FloatTensor(1, 1, B))
        nn.init.constant_(self.beta_u, 0)
        nn.init.constant_(self.beta_a, 0)

        self.final_lambda = final_lambda
        self.ln_2pi = math.log(2*math.pi)

    def m_step(self, v, a_in, r):
        b, l, _, _, _ = v.shape
        r = r * a_in.view(b, l, -1, 1, 1)
        r_sum = r.sum(dim=2, keepdim=True)
        coeff = r / (r_sum + eps)
        mu = torch.sum(coeff * v, dim=2, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=2, keepdim=True) + eps
        r_sum = r_sum.squeeze(2)
        sigma_sq = sigma_sq.squeeze(2)
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        a_out = torch.sigmoid(self.lambda_*(self.beta_a - cost_h.sum(dim=3)))

        return a_out, mu, sigma_sq

    def e_step(self, v, a_out, mu, sigma_sq):
        b, l, _ = a_out.shape
        sigma_sq = sigma_sq.unsqueeze(2)

        ln_p_j = -0.5 * torch.sum(torch.log(sigma_sq*self.ln_2pi), dim=-1) \
                    - torch.sum((v - mu)**2 / (2 * sigma_sq), dim=-1)

        ln_ap = ln_p_j + torch.log(a_out.view(b, l, 1, self.B))
        r = torch.softmax(ln_ap, dim=-1)
        return r.unsqueeze(-1)

    def forward(self, a_in, pose):
        batch_size = a_in.shape[0]

        b, _, h, w = a_in.shape
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        pose = pose.view(b, self.A, self.psize, self.kk, l)
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        pose = pose.view(b, l, self.kkA, self.psize)
        pose = pose.view(batch_size, l, self.kkA, self.mat_dim, self.mat_dim).unsqueeze(3)

        #pose_out = torch.matmul(pose, self.W)  #(8,16,288,32,4,4)
        #edit dylinear_pose
        temp_pose = pose.view(b*l,self.kkA,self.psize)
        pose_out = self.mpose(temp_pose)

        #edit externel
        pose_out = pose_out.view(b * l, self.kkA, self.B, self.psize)
        pose_outs = torch.split(pose_out, 1, dim=-2)
        pose_out = torch.stack([self.filter(torch.squeeze(pose))
                                for pose in pose_outs], dim=2).view(b * l, self.kkA, self.B * self.psize)  # 加全局信息
        pose_out = pose_out.view(b, l, self.kkA, self.B, 4,4)

        v = pose_out.view(b,l,self.kkA,self.B,self.psize)

        # [b, l, kkA, B, psize]
        #v = pose_out.view(batch_size, l, self.kkA, self.B, -1)

        # [b, kkA, l]
        a_in = F.unfold(a_in, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a_in = a_in.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a_in = a_in.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA]
        a_in = a_in.view(b, l, self.kkA)

        r = a_in.new_ones(batch_size, l, self.kkA, self.B, 1)
        #edit dylinear_r
        #logit = self.cpose2(temp_pose)
        #r = logit

        #edit spatial
        r_list=torch.split(r.view(batch_size*l,self.kkA,self.B),1,dim=-1)
        logit = torch.stack([self.spatinal(torch.squeeze(gate.squeeze(-1),-1)) for gate in r_list],dim=-1)
        r = torch.softmax(logit,dim=2)
        r = r.view(b, l, self.kkA,self.B).unsqueeze(-1)

        for i in range(self.iters):
            self.lambda_ = self.final_lambda * (1 - 0.95 ** (i+1))
            a_out, pose_out, sigma_sq = self.m_step(v, a_in, r)
            if i < self.iters - 1:
                r = self.e_step(v, a_out, pose_out, sigma_sq)

        pose_out = pose_out.squeeze(2).view(b, l, -1)
        pose_out = pose_out.transpose(1, 2)
        a_out = a_out.transpose(1, 2).contiguous()

        oh = ow = math.floor(l**(1/2))

        a_out = a_out.view(b, -1, oh, ow)
        pose_out = pose_out.view(b, -1, oh, ow)

        return a_out, pose_out