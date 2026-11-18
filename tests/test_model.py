import pytest
import torch
from safetensors.torch import load_file

from src.model import HybridSASRec
from src.train import sampled_softmax_loss


def make(toy_processed, **kw):
    torch.manual_seed(0)
    m = HybridSASRec(n_items=10, content_dim=16, d_model=32, n_layers=2, n_heads=4, max_len=8,
                     domain_ranges=[[1, 7], [7, 11]], **kw)
    if m.use_content:
        m.set_content(load_file(str(toy_processed / "item_content.safetensors"))["content"])
    return m.eval()


def test_left_padding_does_not_change_user_state(toy_processed):
    m = make(toy_processed)
    a = m.user_state(torch.tensor([[1, 2, 3]]), torch.tensor([0]))
    b = m.user_state(torch.tensor([[0, 0, 0, 1, 2, 3]]), torch.tensor([0]))
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.isfinite(m(torch.tensor([[0, 0, 0, 1]]), torch.tensor([0]))).all()


def test_causal(toy_processed):
    m = make(toy_processed)
    h1 = m(torch.tensor([[1, 2, 3, 4]]), torch.tensor([0]))
    h2 = m(torch.tensor([[1, 2, 3, 6]]), torch.tensor([0]))
    assert torch.allclose(h1[:, :3], h2[:, :3], atol=1e-6)
    assert not torch.allclose(h1[:, 3], h2[:, 3])


def test_save_load_roundtrip(toy_processed, tmp_path):
    m = make(toy_processed)
    m.save_pretrained(tmp_path / "m")
    m2 = HybridSASRec.from_pretrained(str(tmp_path / "m"))
    m2.set_content(load_file(str(toy_processed / "item_content.safetensors"))["content"])
    m2.eval()
    x, d = torch.tensor([[0, 3, 1, 5]]), torch.tensor([1])
    assert torch.allclose(m.user_state(x, d), m2.user_state(x, d))
    assert m2.domain_ranges == [[1, 7], [7, 11]] and m2.encoder == m.encoder
    assert "content" not in m2.state_dict()  # shipped separately


def test_content_only_and_id_only(toy_processed):
    m = make(toy_processed)
    v = m.domain_items(1, id_scale=0.0)
    assert v.shape == (4, 32)
    ido = make(toy_processed, use_content=False)
    with pytest.raises(ValueError):
        ido.domain_items(0, id_scale=0.0)


def test_loss_backward(toy_processed):
    m = make(toy_processed).train()
    inp = torch.tensor([[0, 1, 2, 3], [0, 0, 7, 8]])
    tgt = torch.tensor([[0, 2, 3, 4], [0, 0, 8, 9]])
    loss = sampled_softmax_loss(m, inp, tgt, 0, 1, 7, n_neg=5)
    loss.backward()
    assert torch.isfinite(loss) and m.content_proj.weight.grad is not None
