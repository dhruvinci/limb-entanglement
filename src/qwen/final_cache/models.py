"""Small cloud-trained readouts; no backbone/model-weight loading."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def training_scaler(arrays,indices):
    if not indices:raise ValueError('No training rows')
    x=np.concatenate([np.asarray(arrays[i],np.float32) for i in indices],axis=0)
    return x.mean(0),x.std(0).clip(1e-4)


def ownership_loss(logits,labels,contact=None,contact_weight=1.):
    if contact_weight<0:raise ValueError('Negative contact weight')
    if labels.shape[-2:]!=logits.shape[-2:]:
        logits=F.interpolate(logits,size=labels.shape[-2:],mode='bilinear',align_corners=False)
    valid=labels!=255
    if not valid.any():raise ValueError('No supervised pixels')
    raw=F.cross_entropy(logits.float(),labels.long(),ignore_index=255,reduction='none')
    weights=torch.ones_like(raw)
    if contact is not None:weights=weights+(contact>0).float()*(contact_weight-1)
    losses=[]
    for k in range(3):
        keep=(labels==k)&(weights>0)
        if keep.any():losses.append((raw[keep]*weights[keep]).sum()/weights[keep].sum())
    if not losses:raise ValueError('No supervised pixels after contact exclusion')
    return torch.stack(losses).mean()


class OwnershipHead(nn.Module):
    def __init__(self,channels=64,semantic_dim=64,width=24,mode='appearance',gated=False):
        super().__init__();self.mode=mode;self.width=width
        self.fusion_gate=nn.Conv2d(channels,2,1) if gated else None
        self.visual=nn.Sequential(nn.Conv2d(channels,width,1),nn.GroupNorm(4,width),nn.GELU(),nn.Conv2d(width,width,3,padding=1),nn.GELU())
        self.query=nn.Linear(channels,width)
        self.semantic=nn.Linear(semantic_dim,width,bias=False)
        self.context=nn.Linear(width,width,bias=False)
        self.background=nn.Conv2d(width,1,1)
        self.score=nn.Sequential(nn.Conv2d(width*4,width,1),nn.GELU(),nn.Conv2d(width,1,1))

    def forward(self,x,appearance,semantic=None):
        if self.fusion_gate is not None:
            gates=self.fusion_gate(x).softmax(1)*2;half=x.shape[1]//2
            x=torch.cat([x[:,:half]*gates[:,0:1],x[:,half:]*gates[:,1:2]],1)
        v=self.visual(x);qa=self.query(appearance)
        if semantic is None:semantic=torch.zeros((x.shape[0],2,self.semantic.in_features),device=x.device,dtype=x.dtype)
        qs=self.semantic(F.normalize(semantic,dim=-1))
        scores=[self.background(v)]
        for a in range(2):
            q=qa[:,a]
            if self.mode=='global':q=q+qs[:,a]
            if self.mode=='local':
                attention=torch.softmax(torch.einsum('bc,bchw->bhw',qs[:,a],v).flatten(1)/self.width**.5,dim=-1)
                context=torch.einsum('bn,bcn->bc',attention,v.flatten(2))
                q=q+qs[:,a]+self.context(context)
            q=q[:,:,None,None].expand_as(v)
            scores.append(self.score(torch.cat([v,q,v*q,(v-q).abs()],dim=1)))
        return torch.cat(scores,dim=1)


def connectivity(logits,features,steps=3):
    f=F.normalize(F.interpolate(features[None],size=logits.shape[-2:],mode='bilinear',align_corners=False)[0],dim=0)
    out=logits.clone();h,w=out.shape[-2:]
    for _ in range(steps):
        total=out[1:].clone();weight=torch.ones((1,h,w),device=out.device)
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            q=f.roll((dy,dx),(1,2));g=((f*q).sum(0,keepdim=True).clamp(-1,1)-1).mul(10).exp()
            if dy<0:g[:,h+dy:,:]=0
            if dy>0:g[:,:dy,:]=0
            if dx<0:g[:,:,w+dx:]=0
            if dx>0:g[:,:,:dx]=0
            total+=out[1:].roll((dy,dx),(1,2))*g;weight+=g
        out=torch.cat([logits[:1],.75*out[1:]+.25*total/weight],0)
    return out


class TemporalHead(nn.Module):
    def __init__(self,width=24):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(12,width,3,padding=1),nn.GELU(),nn.Conv2d(width,width,3,padding=1),nn.GELU(),nn.Conv2d(width,3,1))
    def forward(self,x):return self.net(x)+x[:,:3]
