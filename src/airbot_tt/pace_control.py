"""Tensor lifecycle and bounded action command for the M7 simulation interface."""
import torch
from .single_shot import EVENT_RULES


class PositionCommand:
    """Cubic interpolation of position requests, with bounded command derivatives.

    End velocity uses the last three requested positions (backward difference).
    The reference has continuous position/velocity; actual commands additionally
    obey per-tick velocity/acceleration bounds. No robot state is written.
    """
    def __init__(self, nominal, scale, lower, upper, n, vmax, amax, dt, period=1/48):
        self.nominal, self.scale, self.lower, self.upper = nominal, scale, lower, upper
        self.vmax, self.amax, self.dt, self.period = vmax, amax, dt, period
        self.q=nominal.repeat(n,1);self.v=torch.zeros_like(self.q)
        self.request=self.q.clone();self.previous_request=self.q.clone()
        self.start_v=self.v.clone();self.c2=self.v.clone();self.c3=self.v.clone()
        self.t=torch.zeros(n,1,device=nominal.device)

    def reset(self, ids):
        self.q[ids]=self.nominal;self.v[ids]=0
        self.request[ids]=self.nominal;self.previous_request[ids]=self.nominal
        self.start_v[ids]=0;self.c2[ids]=0;self.c3[ids]=0;self.t[ids]=0

    def set_action(self, action):
        target=torch.clamp(self.nominal+self.scale*action.clamp(-1,1),self.lower,self.upper)
        h=self.period
        end_v=((3*target-4*self.request+self.previous_request)/(2*h)).clamp(-self.vmax,self.vmax)
        self.previous_request=self.request.clone();self.request=target
        delta=target-self.q
        self.start_v=self.v.clone()
        self.c2=3*delta/h**2-(2*self.v+end_v)/h
        self.c3=-2*delta/h**3+(self.v+end_v)/h**2
        self.t.zero_()

    def step(self):
        self.t=(self.t+self.dt).clamp_max(self.period)
        desired=(self.start_v+2*self.c2*self.t+3*self.c3*self.t**2).clamp(-self.vmax,self.vmax)
        velocity=self.v+(desired-self.v).clamp(-self.amax*self.dt,self.amax*self.dt)
        self.q=torch.clamp(self.q+(self.v+velocity)*(.5*self.dt),self.lower,self.upper)
        self.v=velocity
        return self.q,self.v


class TensorShotEvents:
    """Same priority and event rules as ShotEvents, batched on the physics device.

    phase: incoming=0, own_bounce=1, hit=2, crossed=3.
    outcome: none=0, unexpected=1, arm_fault=2, net=3, illegal_hit=4,
             wrong_bounce=5, legal_return=6, miss=7, out=8, timeout=9.
    Contacts columns: table, net, paddle, illegal (arm/fixer/handle).
    """
    rules=EVENT_RULES
    # Code 3 remains reserved to read historical reports; v2 never emits net.
    names=('none','unexpected_contact','arm_fault','net','illegal_hit','wrong_bounce','legal_return','miss','out','timeout')

    def __init__(self,n,device,c,timeout):
        self.c,self.timeout=c,timeout
        self.phase=torch.zeros(n,dtype=torch.long,device=device)
        self.outcome=torch.zeros_like(self.phase)
        self.contacts=torch.zeros(n,4,dtype=torch.bool,device=device)
        self.previous=torch.zeros(n,6,device=device)
        self.has_previous=torch.zeros(n,dtype=torch.bool,device=device)
        self.hit=torch.zeros_like(self.has_previous)
        self.net_touched=torch.zeros_like(self.has_previous)
        self.crossing=torch.zeros(n,3,device=device)
        self.landing=torch.zeros(n,3,device=device)

    def reset(self,ids):
        self.phase[ids]=0; self.outcome[ids]=0; self.contacts[ids]=False
        self.has_previous[ids]=False; self.hit[ids]=False; self.net_touched[ids]=False; self.crossing[ids]=0; self.landing[ids]=0

    def update(self,time,state,contact,fault):
        active=self.outcome==0
        new=contact & ~self.contacts
        self.contacts=contact.clone()
        def finish(mask,code):
            self.outcome[active & (self.outcome==0) & mask]=code
        self.net_touched |= active & contact[:,1]
        finish(fault,2);finish(contact[:,3],1)
        paddle=active & (self.outcome==0) & new[:,2]
        finish(paddle & (self.phase!=1),4)
        legal=paddle & (self.phase==1)
        self.phase[legal]=2;self.hit|=legal
        cross=active & (self.outcome==0) & self.has_previous & (self.phase==2) & (self.previous[:,0]<0) & (state[:,0]>=0) & (state[:,3]>0)
        fraction=-self.previous[:,0]/(state[:,0]-self.previous[:,0]).clamp_min(1e-12)
        point=self.previous[:,:3]+fraction[:,None]*(state[:,:3]-self.previous[:,:3])
        self.crossing[cross]=point[cross]
        finish(cross & (point[:,1].abs()>self.c['table_size_m'][1]/2),8)
        self.phase[cross & (self.outcome==0)]=3
        table=active & (self.outcome==0) & new[:,0]
        own=table & (self.phase==0) & (state[:,0]<0)
        returned=table & (self.phase==3) & (state[:,0]>0)
        self.landing[returned]=state[returned,:3]
        finish(returned,6);finish(table & ~own & ~returned,5)
        self.phase[own]=1
        low=state[:,2]<self.c['table_top_m']-.05
        finish(low & (self.phase<2),7);finish(low,8)
        finish((state[:,0].abs()>2)|(state[:,1].abs()>1.2),8)
        finish(time>=self.timeout,9)
        self.previous=state.clone();self.has_previous|=active
        return legal


def bootstrap_timeout(reward,terminal_value,truncated,gamma):
    """Bootstrap from reset-before state exactly once; real termination adds zero."""
    return reward+gamma*terminal_value.flatten()*truncated
