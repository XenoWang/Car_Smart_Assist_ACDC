"""Independent training-view geometry, schedule and optimizer receipt oracles."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from car_smart_assist.perception.lowlight_training import load_plan, recipe_identity, training_class, transform_batch


def batch_fixture():
    pixels=torch.full((2,3,8,8),17,dtype=torch.uint8)
    pixels[0,:,2:6]=torch.tensor([0,64,255],dtype=torch.uint8)[:,None,None]
    pixels[1].fill_(128)
    names=[str(Path("one.png").resolve()),str(Path("two.png").resolve())]
    batch={"img":pixels,"im_file":names,"ori_shape":((4,8),(8,8)),"cls":torch.tensor([5,8]),
           "obstacle_roi":torch.ones((2,8,8),dtype=torch.bool)}
    plan={"mode":"lowlight","schedule":[{"one"}],"image_ids":dict(zip(names,("one","two"))),
          "parameters":{"seed":42,"imgsz":8,"exposure":.25,"shot_scale":0,"read_std":0}}
    return batch,plan


def test_training_transform_preserves_padding_labels_and_caller_pixels():
    batch,plan=batch_fixture()
    before=batch["img"].clone()
    altered,selected,applied=transform_batch(batch,plan,0)
    expected=before.clone()
    expected[0,:,2:6]=torch.tensor([0,30,137],dtype=torch.uint8)[:,None,None]
    assert torch.equal(altered["img"],expected) and torch.equal(batch["img"],before)
    assert selected==applied==1
    assert altered["cls"] is batch["cls"] and altered["obstacle_roi"] is batch["obstacle_roi"]
    plan["mode"]="control"
    control,selected,applied=transform_batch(batch,plan,0)
    assert control["img"] is batch["img"] and selected==1 and applied==0


def test_schedule_uses_epoch_and_image_identity_without_global_random_state():
    batch,plan=batch_fixture()
    plan["schedule"].append({"one"})
    plan["parameters"].update(shot_scale=.001,read_std=.001)
    state=np.random.get_state()
    first=transform_batch(batch,plan,0)[0]["img"]
    repeat=transform_batch(batch,plan,0)[0]["img"]
    second=transform_batch(batch,plan,1)[0]["img"]
    assert torch.equal(first,repeat) and not torch.equal(first,second)
    now=np.random.get_state()
    assert state[0]==now[0] and np.array_equal(state[1],now[1]) and state[2:]==now[2:]


def test_budget_receipt_counts_real_steps_and_common_frame_order():
    class Base:
        def __init__(self):
            self.epoch=0
            self.optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.]))],lr=.1)
        def preprocess_batch(self,batch):
            return batch

    batch,plan=batch_fixture()
    trial=training_class(Base,plan)()
    trial.preprocess_batch(batch)
    trial.optimizer.step()
    trial.optimizer.step()
    assert trial.lowlight_budget()["images"]==2
    assert trial.lowlight_budget()["batches"]==1 and trial.lowlight_budget()["optimizer_updates"]==2
    assert trial.lowlight_budget()["selected"]==trial.lowlight_budget()["applied"]==1
    plan["mode"]="control"
    control=training_class(Base,plan)()
    control.preprocess_batch(batch)
    assert control.lowlight_budget()["applied"]==0
    assert control.lowlight_budget()["order_sha256"]==trial.lowlight_budget()["order_sha256"]


def test_recipe_checks_identity_and_training_only_non_night_schedule(tmp_path):
    paths=[f"images/train/{i}.png" for i in range(4)]
    plan={"version":1,"dataset_signature":"data","imgsz":960,"seed":42,"epochs":2,"minimum_mean":80,
          "exposure":.5,"shot_scale":.001,"read_std":.001,"eligible":dict.fromkeys(paths,100),
          "schedule":[[paths[0]],[paths[1]]]}
    path=tmp_path/"plan.json"
    path.write_text(json.dumps(plan),encoding="utf-8")
    cfg={"data":"data.yaml","train":{"epochs":2,"imgsz":960,"seed":42},
         "training_lowlight":{"mode":"lowlight","plan":"plan.json",
                              "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}}
    manifest={"joint_supervision":True,"signature":"data",
              "images":[{"image":p,"split":"train","condition":"fog"} for p in paths]}
    result=load_plan(tmp_path,cfg,manifest)
    assert result["schedule"]==[{paths[0]},{paths[1]}] and result["mode"]=="lowlight"
    manifest["images"][0]["condition"]="night"
    with pytest.raises(ValueError,match="non-night"):
        load_plan(tmp_path,cfg,manifest)
    manifest["images"][0].update(condition="fog",split="val")
    with pytest.raises(ValueError,match="training frames"):
        load_plan(tmp_path,cfg,manifest)
    path.write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError,match="identity"):
        load_plan(tmp_path,cfg,manifest)


@pytest.mark.parametrize("bad",[{},False,[],{"mode":"unexpected","plan":"x","sha256":"0"*64}])
def test_malformed_optional_recipe_fails_closed(bad):
    with pytest.raises(ValueError):
        recipe_identity({"training_lowlight":bad})
    assert recipe_identity({}) is None
