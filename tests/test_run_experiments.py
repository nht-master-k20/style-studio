import os

from experiments.run_experiments import plan_runs


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("x")


def test_plan_runs_full_pending(tmp_path):
    out = str(tmp_path / "cond")
    runs = plan_runs(["p one", "p two"], ["/s/styleA.jpg", "/s/styleB.jpg"], out)
    assert len(runs) == 4
    pi, prompt, sp, stem = runs[0]
    assert pi == 0 and prompt == "p one" and sp == "/s/styleA.jpg"
    assert stem == os.path.join(out, "p00__styleA")


def test_plan_runs_skips_completed(tmp_path):
    out = str(tmp_path / "cond")
    touch(os.path.join(out, "p00__styleA.jpg"))
    touch(os.path.join(out, "p00__styleA.json"))
    touch(os.path.join(out, "p01__styleA.jpg"))  # thieu .json -> van pending
    runs = plan_runs(["p one", "p two"], ["/s/styleA.jpg"], out)
    stems = [r[3] for r in runs]
    assert os.path.join(out, "p00__styleA") not in stems
    assert os.path.join(out, "p01__styleA") in stems
    assert len(runs) == 1
