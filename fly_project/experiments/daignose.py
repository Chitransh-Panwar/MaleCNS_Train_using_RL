import sys
from sb3_contrib import RecurrentPPO
from fly_project.experiments.evaluate import evaluate, trace

name = sys.argv[1] if len(sys.argv) > 1 else "connectome_seed0"
model = RecurrentPPO.load(f"fly_project/logs/{name}/model", device="cpu")
for det in (True, False):
    for test in (False, True):
        for e3 in (False, True):
            print(f"deterministic={det} test_starts={test} smell_off={e3}: "
                  f"{evaluate(model, test, e3, n=50, deterministic=det):.2f}")
trace(model, deterministic=True)