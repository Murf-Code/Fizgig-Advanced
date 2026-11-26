import copy, gc
import torch
from fizgig.training.activation_offload import HostBudget, checkpoint_bounded, self_test

def test_original_budget_and_rng():
    self_test()

def test_full_backward_hook_views():
    torch.manual_seed(7)
    model=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(16,16),torch.nn.Dropout(.2)) for _ in range(4)])
    def run(enabled):
        m=copy.deepcopy(model); order=[]
        for i,layer in enumerate(m):
            layer.register_full_backward_hook(lambda *args,i=i:order.append(i))
        torch.manual_seed(19); x=torch.randn(3,16,requires_grad=True);y=x
        budget=HostBudget(4*3*16*4)
        for layer in m:y=checkpoint_bounded(layer,y,budget=budget,enabled=enabled,cpu_test=True)
        y.sum().backward();values=(y.detach(),x.grad,*[p.grad for p in m.parameters()],torch.rand(2));del y;gc.collect()
        return values,order,budget
    ref,ro,_=run(False); actual,ao,b=run(True)
    assert ro==ao==[3,2,1,0]
    for x,y in zip(ref,actual):torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert b.accepted==4 and b.live==0 and b.layout_skipped==0

def test_partial_storage_and_host_floor():
    x=torch.randn(4,16,requires_grad=True);layer=torch.nn.Linear(16,16)
    b=HostBudget(1024,host_floor_bytes=100,available=lambda:110)
    y=checkpoint_bounded(layer,x[:2],budget=b,cpu_test=True);y.sum().backward()
    assert b.layout_skipped==1 and b.accepted==0
    y=checkpoint_bounded(layer,x,budget=b,cpu_test=True);y.sum().backward()
    assert b.skipped==1 and b.live==0
