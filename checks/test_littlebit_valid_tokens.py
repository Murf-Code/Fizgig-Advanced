import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
from valid_tokens import valid_rows


def test_no_padding_even_with_interior_holes():
    mask=torch.tensor([[True,False,True,False,True]+[False]*235])
    idx,meta=valid_rows(16,mask)
    assert len(idx)==16 and len(set(idx.tolist()))==16
    assert set(idx[idx>=16].tolist())=={16,18,20}
    assert meta["sampled_image"]==13 and meta["sampled_text"]==3
    assert meta["padding_tokens"]==237


def test_balanced_and_no_text():
    idx,m=valid_rows(100,torch.ones(1,10,dtype=torch.bool))
    assert m["sampled_image"]==m["sampled_text"]==8
    assert idx[:8].max()<100 and idx[8:].min()>=100
    idx,m=valid_rows(3,torch.zeros(1,253,dtype=torch.bool))
    assert idx.tolist()==[0,1,2] and m["sampled_text"]==0


def test_invalid():
    with pytest.raises(ValueError):
        valid_rows(3,torch.ones(2,3,dtype=torch.bool))
