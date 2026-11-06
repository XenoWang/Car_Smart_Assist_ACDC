"""Independent FRISA equations, receptive fields, gradient and identity oracles."""
import math

import pytest
import torch

from car_smart_assist.config.detection import frisa_settings
from car_smart_assist.perception.frisa import FRISA, FrisaP4
from ultralytics.nn.modules.block import C3k2


def test_equation_oracle_preserves_residual_and_both_branch_weights():
    layer = FRISA(1).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.zero_()
        for module in layer.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.weight.fill_(1)
                module.running_var.fill_(1-module.eps)
        layer.decompose.weight.fill_(1)
        layer.output_gate[0].weight.fill_(1)
    semantic = 1 / (1+math.exp(-1))
    fused = semantic/8 + (1-semantic)/16
    expected = 1 + 1/(1+math.exp(-fused))
    result = layer(torch.ones(2,1,5,7))
    torch.testing.assert_close(result, torch.full((2,1,5,7),expected),rtol=0,atol=1e-6)


def test_zero_gate_logits_give_one_and_a_half_input_including_negative_features():
    layer=FRISA(2).eval()
    with torch.no_grad():
        layer.output_gate[0].weight.zero_()
        layer.output_gate[1].bias.zero_()
    x=torch.linspace(-2,2,48).reshape(2,2,3,4)
    torch.testing.assert_close(layer(x),x*1.5,atol=1e-6,rtol=0)


@pytest.mark.parametrize("index,dilation",[(0,1),(1,2),(2,3)])
def test_detail_branches_sample_distinct_reviewed_offsets(index,dilation):
    conv=FRISA(1).details[index]
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0,0,0,1]=1
    x=torch.arange(1,64,dtype=torch.float32).reshape(1,1,9,7)
    expected=torch.zeros_like(x)
    expected[:,:,dilation:]=x[:,:,:-dilation]
    torch.testing.assert_close(conv(x),expected,rtol=0,atol=0)


def test_every_gate_and_detail_branch_participates_in_backward():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        layer=FRISA(4).train()
        x=torch.randn(2,4,9,7,requires_grad=True)
        layer(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name,param in layer.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(),name
    for group in (layer.decompose,layer.prefilter,*layer.details,layer.channel,
                  layer.residual_to_semantic,layer.semantic_to_residual,layer.spatial,layer.output_gate):
        assert sum(float(p.grad.abs().sum()) for p in group.parameters())>0


def test_p4_keeps_native_keys_metadata_and_cpu_rng():
    block=C3k2(8,8,n=1).eval()
    block.i,block.f,block.type,block.np=19,-1,"C3k2",sum(p.numel() for p in block.parameters())
    before={k:v.clone() for k,v in block.state_dict().items()}
    random_before=torch.get_rng_state().clone()
    result=FrisaP4(block)
    assert torch.equal(random_before,torch.get_rng_state())
    assert (result.i,result.f,result.c)==(19,-1,4)
    assert {k for k in result.state_dict() if not k.startswith("frisa.")}==set(before)
    for k,v in before.items():
        assert torch.equal(v,result.state_dict()[k]),k
    x=torch.ones(2,8,9,7)
    torch.testing.assert_close(result(x),result.forward_split(x),rtol=0,atol=1e-6)


@pytest.mark.parametrize("entry",[False,[],{},{"mode":"other"},{"mode":"frisa","unreviewed":True}])
def test_optional_recipe_rejects_implicit_or_unknown_settings(entry):
    with pytest.raises(ValueError):
        frisa_settings({"frisa":entry})


def test_missing_recipe_keeps_existing_contract_and_candidate_has_fixed_identity():
    assert frisa_settings({}) is None
    assert frisa_settings({"frisa":None}) is None
    assert frisa_settings({"frisa":{"mode":"frisa"}})=={
        "mode":"frisa","version":"eq16_34_k1_d3_v1","p4_index":19,
        "depthwise_kernel":3,"spatial_kernel":1,"initialization_seed":42}

def test_paired_receipt_counts_optimizer_calls_and_processed_frames():
    from car_smart_assist.perception.frisa import training_class
    class Base:
        def __init__(self):
            self.optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.]))],lr=.1)
        def preprocess_batch(self,batch):
            return batch
    cls=training_class(Base,frisa_settings({"frisa":{"mode":"control"}}))
    trainer=cls()
    batch={"im_file":["image_a","image_b"]}
    assert trainer.preprocess_batch(batch) is batch
    trainer.optimizer.step()
    receipt=trainer.frisa_receipt()
    assert (receipt["images"],receipt["batches"],receipt["optimizer_updates"])==(2,1,1)
    assert receipt["transfer"] is None
