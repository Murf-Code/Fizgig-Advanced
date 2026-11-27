import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research/krea2_littlebit"))
from native_codec import encode,linear
from screen import evaluate
from fast_screen import q6_probes
from preparation_cache import references,link_or_copy,write_checksum,verify_checksum


def test_q6_probe_reuse_exact():
    torch.manual_seed(42)
    p,s=encode(torch.randn(259,128).bfloat16(),"q6-had")
    probes=[(torch.randn(k,128).bfloat16(),torch.randn(k,259).bfloat16()) for k in (1,3,8,16)]
    ref=[evaluate(lambda t:linear(t,p,s),x,dy) for x,dy in probes]
    for pair,want in zip(q6_probes(p,s,probes),ref):
        for a,b in zip(pair,want):
            torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_reference_cache(tmp_path):
    probes=[(torch.ones(2,3,dtype=torch.bfloat16),torch.ones(2,4,dtype=torch.bfloat16))]
    path=tmp_path/"ref.safetensors"
    r,hit=references(path,probes,lambda x,dy:(dy,x))
    assert not hit
    def fail(*args): raise AssertionError("unexpected reference recomputation")
    s,hit=references(path,probes,fail)
    assert hit and all(torch.equal(a,b) for a,b in zip(r[0],s[0]))


def test_archive_link(tmp_path):
    source=tmp_path/"source"; source.write_bytes(b"archive")
    dest=tmp_path/"cache"/"archive"
    assert link_or_copy(source,dest) in ("hardlink","copy")
    assert dest.read_bytes()==source.read_bytes()
    with pytest.raises(FileExistsError): link_or_copy(source,dest)


def test_cache_checksum_rejects_damage(tmp_path):
    path=tmp_path/"archive.safetensors"
    path.write_bytes(b"archive")
    expected=write_checksum(path)
    assert verify_checksum(path)==expected
    path.write_bytes(b"damage!")
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        verify_checksum(path)
