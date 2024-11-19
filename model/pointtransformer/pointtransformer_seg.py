import torch
import torch.nn as nn
import time
import numpy as np
from lib.pointops.functions import pointops


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes // share_planes),
                                    nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                    nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, pxo) -> torch.Tensor:
        if len(pxo) == 6:
            p, x, o, p2, x2, o2 = pxo  # (n, 3), (n, c), (b)
            x_q, x_k, x_v = self.linear_q(x), self.linear_k(x2), self.linear_v(x2)  # (n, c)
            x_k = pointops.queryandgroup(self.nsample, p2, p, x_k, None, o2, o, use_xyz=True)  # (n, nsample, 3+c)
            x_v = pointops.queryandgroup(self.nsample, p2, p, x_v, None, o2, o, use_xyz=False)  # (n, nsample, c)
            p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
            for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
            w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
            for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
            w = self.softmax(w)  # (n, nsample, c)
            n, nsample, c = x_v.shape; s = self.share_planes
            x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        else:
            p, x, o = pxo  # (n, 3), (n, c), (b)
            x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)
            x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)  # (n, nsample, 3+c)
            x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)  # (n, nsample, c)
            p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
            for i, layer in enumerate(self.linear_p): p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)    # (n, nsample, c)
            w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)
            for i, layer in enumerate(self.linear_w): w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
            w = self.softmax(w)  # (n, nsample, c)
            n, nsample, c = x_v.shape; s = self.share_planes
            x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        p, x, o = pxo # (n, 3), (n, c), (b)
        x_orig = torch.tensor(x)
        o_orig = torch.tensor(o)
        p_orig = torch.tensor(p)
        if self.stride != 1:
            new_stride = self.stride / 2
            n_o, count = [o[0].item() // new_stride], o[0].item() // new_stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // new_stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            

            idx1 = torch.empty(0, dtype=idx.dtype, device=idx.device)
            idx2 = torch.empty(0, dtype=idx.dtype, device=idx.device)
            n_o_aug = torch.cat((torch.tensor([0], device=n_o.device, dtype = n_o.dtype), n_o))
            n_o_1 = []
            n_o_2 =[]
            n1_count = 0
            n2_count = 0
            for i in range(n_o.shape[0]):
                npoints = n_o_aug[i + 1] - n_o_aug[i]
                if npoints % 2 == 0:
                    npoint1 = int(npoints / 2)
                    npoint2 = int(npoints / 2)
                else:
                    npoint1 = int(npoints // 2 + 1)
                    npoint2 = int(npoints // 2)
                n1_count += npoint1
                n2_count += npoint2
                added_part1 = idx[n_o_aug[i]:n_o_aug[i + 1]][0:npoint1]
                added_part2 = idx[n_o_aug[i]:n_o_aug[i + 1]][npoint1:npoint1+npoint2]
                idx1 = torch.cat((idx1, added_part1))
                idx2 = torch.cat((idx2, added_part2))

                n_o_1.append(n1_count)
                n_o_2.append(n2_count)


            # only for test:
            n_o = torch.tensor(n_o_1, dtype = n_o.dtype, device = n_o.device)
            n_o2 = torch.tensor(n_o_2, dtype = n_o.dtype, device = n_o.device)

            
            
                
            n_p = p[idx1.long(), :]  # (m, 3)
            n_p2 = p[idx2.long(), :]  # (m, 3)
            

            
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)

            x2 = pointops.queryandgroup(self.nsample, p_orig, n_p2, x_orig, None, o_orig, n_o2, use_xyz=True)  # (m, 3+c, nsample)
            x2 = self.relu(self.bn(self.linear(x2).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x2 = self.pool(x2).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
            p2, o2 = n_p2, n_o2

        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
            p2 = torch.tensor(p)
            o2 = torch.tensor(o)
            x2 = torch.tensor(x)
        return [p, x, o, p2, x2, o2]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, pxo):
        if len(pxo) == 6:
            p, x, o, p2, x2, o2 = pxo  # (n, 3), (n, c), (b)
            identity = x
            x = self.relu(self.bn1(self.linear1(x)))
            x = self.relu(self.bn2(self.transformer2([p, x, o, p2,x2,  o2])))
            x = self.bn3(self.linear3(x))
            x += identity
            x = self.relu(x)
            return [p2, x2, o2, p,x, o]
        else:
            p, x, o = pxo  # (n, 3), (n, c), (b)
            identity = x
            x = self.relu(self.bn1(self.linear1(x)))
            x = self.relu(self.bn2(self.transformer2([p, x, o])))
            x = self.bn3(self.linear3(x))
            x += identity
            x = self.relu(x)
            return [p, x, o]



class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1, _, _, _ = self.enc1([p0, x0, o0])
        p2, x2, o2, _, _, _ = self.enc2([p1, x1, o1])
        p3, x3, o3, _, _, _ = self.enc3([p2, x2, o2])
        p4, x4, o4, _, _, _ = self.enc4([p3, x3, o3])
        p5, x5, o5, _, _, _ = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x


def pointtransformer_seg_repro(**kwargs):
    model = PointTransformerSeg(PointTransformerBlock, [2, 3, 4, 6, 3], **kwargs)
    return model
