import json
import pytest

from ip_adapter.fusion_controller import FusionController


def make_ctrl(**kw):
    ctrl = FusionController(**kw)
    ctrl.register("layer_a")
    ctrl.register("layer_b")
    return ctrl


def report_step(ctrl, step, d_a, d_b):
    ctrl.report("layer_a", step, d_a)
    ctrl.report("layer_b", step, d_b)


def test_active_before_any_report():
    ctrl = make_ctrl()
    assert ctrl.is_active(1)
    assert not ctrl.stopped


def test_stops_when_converged_after_min_steps():
    ctrl = make_ctrl(rho=0.2, min_steps=3)
    report_step(ctrl, 1, 1.0, 1.0)   # baseline
    report_step(ctrl, 2, 0.5, 0.5)   # r=0.5
    report_step(ctrl, 3, 0.1, 0.1)   # r=0.1 < 0.2 va step >= min_steps -> stop
    assert ctrl.stopped
    assert ctrl.stop_step == 3
    assert not ctrl.is_active(4)


def test_no_stop_before_min_steps():
    ctrl = make_ctrl(rho=0.2, min_steps=5)
    report_step(ctrl, 1, 1.0, 1.0)
    report_step(ctrl, 2, 0.01, 0.01)  # r rat thap nhung step < min_steps
    assert not ctrl.stopped
    assert ctrl.is_active(3)


def test_end_fusion_max_cap():
    ctrl = make_ctrl(rho=0.0, end_fusion_max=10)  # rho=0 -> khong bao gio stop theo r
    assert ctrl.is_active(10)
    assert not ctrl.is_active(11)


def test_baseline_is_running_max():
    # do lech tang o dau (layer truoc cross-attn dau tien co d(1)=0) roi giam
    ctrl = make_ctrl(rho=0.2, min_steps=2)
    report_step(ctrl, 1, 0.0, 1.0)
    report_step(ctrl, 2, 2.0, 0.9)   # layer_a ramp len 2.0 -> baseline_a=2.0
    # r(2) = mean(2.0/2.0, 0.9/1.0) = 0.95 -> khong stop
    assert not ctrl.stopped
    report_step(ctrl, 3, 0.2, 0.1)   # r = mean(0.1, 0.1) = 0.1 < 0.2 -> stop
    assert ctrl.stopped
    assert ctrl.stop_step == 3


def test_all_zero_layer_treated_converged():
    ctrl = make_ctrl(rho=0.2, min_steps=1)
    report_step(ctrl, 1, 0.0, 1.0)
    report_step(ctrl, 2, 0.0, 0.1)   # layer_a luon 0 -> r_a=0; r=mean(0, 0.1)=0.05 -> stop
    assert ctrl.stopped


def test_decision_waits_for_all_layers():
    ctrl = make_ctrl(rho=1.0, min_steps=1)  # rho=1.0: stop ngay khi du bao cao
    ctrl.report("layer_a", 1, 0.5)
    assert not ctrl.stopped               # layer_b chua bao cao
    ctrl.report("layer_b", 1, 0.5)
    assert ctrl.stopped                   # du bao cao -> quyet dinh


def test_report_after_stop_is_noop():
    ctrl = make_ctrl(rho=1.0, min_steps=1)
    report_step(ctrl, 1, 0.5, 0.5)
    assert ctrl.stopped
    ctrl.report("layer_a", 2, 9.9)        # khong crash, khong doi trang thai
    assert ctrl.stop_step == 1


def test_reset():
    ctrl = make_ctrl(rho=1.0, min_steps=1)
    report_step(ctrl, 1, 0.5, 0.5)
    assert ctrl.stopped
    ctrl.reset()
    assert not ctrl.stopped
    assert ctrl.stop_step is None
    assert ctrl.history == []
    assert ctrl.is_active(1)
    # layers van giu sau reset
    report_step(ctrl, 1, 0.5, 0.5)
    assert ctrl.stopped


def test_history_records_r_per_step():
    ctrl = make_ctrl(rho=0.0, min_steps=1)
    report_step(ctrl, 1, 1.0, 1.0)
    report_step(ctrl, 2, 0.5, 0.5)
    assert ctrl.history == [(1, 1.0), (2, 0.5)]


def test_to_dict_json_serializable():
    ctrl = make_ctrl(rho=0.2)
    report_step(ctrl, 1, 1.0, 1.0)
    s = json.dumps(ctrl.to_dict())
    assert "stop_step" in s and "r_history" in s
