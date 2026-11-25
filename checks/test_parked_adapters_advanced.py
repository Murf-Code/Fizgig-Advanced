"""Upstream #405 regression adapted to Advanced's block swap paths."""
from unittest.mock import patch
import torch
from torch import nn
from fizgig.modules import offloading as shared
from fizgig.krea2 import offloading as krea


def test_parked_adapter_excluded_from_weight_moves():
    block = nn.ModuleDict({"base": nn.Linear(2, 2), "preview": nn.Sequential(nn.Linear(2, 2))})
    block["preview"]._fizgig_parked = True
    parked = block["preview"][0].weight
    assert id(block["preview"][0]) in shared.parked_ids(block)
    assert set(shared._streamed_linears(block)) == {"base"}
    original = torch.Tensor.to
    moved = []
    def track(t, *args, **kwargs):
        moved.append(t.data_ptr())
        return original(t, *args, **kwargs)
    with patch.object(torch.Tensor, "to", track):
        shared.block_to(block, "cpu")
        krea.weighs_to_device(block, torch.device("cpu"))
    assert block["base"].weight.data_ptr() in moved
    assert parked.data_ptr() not in moved


def test_parked_linear_not_selected_by_send_only():
    blocks = [nn.Sequential(nn.Linear(2, 2), nn.Sequential(nn.Linear(2, 2))) for _ in range(4)]
    for b in blocks:
        b.requires_grad_(False)
        b[1]._fizgig_parked = True
        b[1][0].weight.requires_grad_(True)
    assert krea.send_only_unfit(blocks, 4, 2) is None
