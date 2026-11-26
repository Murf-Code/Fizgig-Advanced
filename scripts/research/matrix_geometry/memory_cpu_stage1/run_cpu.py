import copy,json,torch
from memory_core import *
from bounded_checkpoint_offload import self_test

def rel(a,b): return float((a-b).norm()/b.norm().clamp_min(1e-20))

def run():
    torch.set_num_threads(1);torch.manual_seed(5)
    result={'torch':torch.__version__,'device':'CPU','drift':[]}
    for scenario in ['iid','correlated','outlier','subbin']:
        for block in [64,128,256]:
            for stochastic in [False,True]:
                g=torch.Generator().manual_seed(45); rnd=torch.Generator().manual_seed(551); m=torch.zeros(64,128);q,s=pack8(m,block);h=torch.randn(m.shape,generator=g)
                errors=[]
                for i in range(1000):
                    grad=torch.randn(m.shape,generator=g)
                    if scenario=='correlated': grad=.95*h+.05*grad
                    if scenario=='outlier': grad.flatten()[::256]=100
                    if scenario=='subbin': grad.fill_(.1);grad.flatten()[::256]=100
                    m.lerp_(grad,.05);v=unpack8(q,s,m.shape,block);v.lerp_(grad,.05);q,s=pack8(v,block,stochastic=stochastic,generator=rnd);v=unpack8(q,s,m.shape,block)
                    if i>=900: errors.append(rel(v,m))
                idx=torch.arange(m.numel())%256!=0
                result['drift'].append(dict(scenario=scenario,block=block,stochastic=stochastic,final_relative=rel(v,m),tail_mean=sum(errors)/len(errors),weak_relative=rel(v.flatten()[idx],m.flatten()[idx]),weak_zero_fraction=float((v.flatten()[idx]==0).float().mean())))
    p=torch.nn.Parameter(torch.randn(17,33));o=PackedMomentum([('p',p)],block=64)
    for _ in range(5): p.grad=torch.randn_like(p);o.step()
    sd=copy.deepcopy(o.state_dict());p2=torch.nn.Parameter(p.detach().clone());o2=PackedMomentum([('p',p2)],block=64);o2.load_state_dict(sd)
    assert o2.state[p2]['q'].dtype==torch.int8
    for _ in range(10):
        p.grad=torch.randn_like(p);p2.grad=p.grad.clone();o.step();o2.step()
    assert torch.equal(p,p2) and torch.equal(o.state[p]['q'],o2.state[p2]['q']) and torch.equal(o.state[p]['scale'],o2.state[p2]['scale'])
    bad=copy.deepcopy(sd);bad['state'][0]['scale'][0]=float('nan');before=o2.state[p2]['q'].clone()
    try: o2.load_state_dict(bad)
    except ValueError: pass
    else: raise AssertionError('malformed state accepted')
    assert torch.equal(before,o2.state[p2]['q'])
    result['restore']='bitwise continuation; malformed state rejected without publishing; CPU harness, not Muon'
    a=torch.empty(100);assert unique_bytes([a,a[:50],a[50:]])==400
    plan=phase_plan(12*1024,6000,[Phase('forward',4000,200),Phase('backward',4500,200),Phase('optimizer',2000)],768)
    assert plan['peak']==10700 and plan['fits'];assert scratch_plan(500,100,50,100,512,70,20)==3
    try: scratch_plan(100,0,0,0,100,101,1)
    except MemoryError:pass
    else:raise AssertionError('oversized unit accepted')
    result['budget']='unique-storage views; phase maxima; strict scratch gate passed';self_test();result['activation']='bounded synchronous CPU self-test passed'
    with open('cpu_results.json','w') as f:json.dump(result,f,indent=2)
    print('PASS: 24 momentum arms; loader continuation; malformed checkpoint; storage accounting; phase budget; activation test')
if __name__=='__main__':run()
