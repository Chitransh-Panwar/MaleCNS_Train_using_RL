import sys, torch
from sb3_contrib import RecurrentPPO
from fly_project.experiments.evaluate import evaluate

model = RecurrentPPO.load(f"fly_project/logs/{sys.argv[1]}/model", device="cpu")
ext = model.policy.features_extractor
orig = ext.gain_raw.data.clone()

def report(tag):
    r = {f"{'test' if t else 'train'}_{'smell_off' if e else 'smell_on'}":
         evaluate(model, test=t, e3=e, n=100) for t in (False, True) for e in (False, True)}
    print(tag, r)

report("trained gains")
ext.gain_raw.data.zero_(); report("gains reset to 1")
g = torch.Generator().manual_seed(0)
ext.gain_raw.data = orig[torch.randperm(orig.numel(), generator=g)]; report("gains shuffled")
ext.gain_raw.data = orig.clone()