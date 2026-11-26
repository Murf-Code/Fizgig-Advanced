"""CPU correctness tests using the user's separately licensed pilot."""
import argparse
from pathlib import Path
import sys
import torch
import adaptive

p = argparse.ArgumentParser()
p.add_argument("--pilot", required=True, type=Path)
args = p.parse_args()
sys.path.insert(0, str(args.pilot.resolve()))
import compact
from transfer_layer import import_file
geometry = import_file("test_geometry", Path(__file__).resolve().parents[3] /
                       "src/fizgig/training/matrix_geometry.py")
torch.set_num_threads(2)
g = torch.Generator().manual_seed(123)

def rows():
    return (torch.randn(32,64,generator=g).bfloat16(),
            torch.randn(32,64,generator=g).bfloat16())

cal, held = rows(), rows()
a = torch.where(torch.randn(1,64,generator=g)>0,1.,-1.)
b = torch.where(torch.randn(64,1,generator=g)>0,1.,-1.)
w = (b@a).bfloat16()
stock = lambda x: torch.nn.functional.linear(x,w)
r = adaptive.adaptive(w,stock,cal,held,compact,geometry,[2,4,8],0.05)
assert r["selected"] != "int8" and r["selected_bytes"] < r["int8_bytes"]
assert len(r["candidates"]) == 20
assert all(e<=0.05 for e in r["heldout_candidate"]["heldout_error"])
print("structured:", r["selected"], "spectrum", r["spectrum"])

w = torch.randn(64,64,generator=g).bfloat16()
r = adaptive.adaptive(w,stock,cal,held,compact,geometry,[2,4,8],0.000001)
assert r["selected"] == "int8"
print("strict random:", r["selected"])

w = torch.eye(64,dtype=torch.bfloat16)
x = torch.zeros(4,64,dtype=torch.bfloat16)
x[:,0] = 1
cal = (x,x.clone())
x2 = torch.zeros_like(x)
x2[:,1] = 1
held = (x2,x2.clone())

class Fake:
    @staticmethod
    def encode(w,c,**kw):
        return torch.ones(1,dtype=torch.uint8),{"codec":c}
    @staticmethod
    def validate(p,s):
        pass
    @staticmethod
    def linear(x,p,s):
        return x*torch.nn.functional.one_hot(torch.tensor(0),64).to(x.dtype)

r = adaptive.adaptive(w,stock,cal,held,Fake,geometry,[2],0.05)
assert r["heldout_candidate"]["calibration_pass"] and r["selected"] == "int8"
print("heldout rejection:", r["heldout_candidate"]["heldout_error"])
print("PASS adaptive selection, actual codecs, strict fallback, heldout veto")
