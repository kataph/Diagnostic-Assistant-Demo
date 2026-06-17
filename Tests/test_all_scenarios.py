"""
Scenario tests for all 135 diagnostic scenarios.

Each scenario has two test functions:
  test_<id>        — full gold-standard action sequence, ending with the repair
                     action; asserts is_system_nominal() after simulate()
  test_<id>_repair — 2-line test_repair oracle check

Access conventions:
  - Battery / PSU-internal components: InvertEnclosure(cube_psu) first
  - ctrl_switch / cable inside ctrl cube (3CC/10CC/DCAC): InvertEnclosure(cube_ctrl)
  - relay / internals in ALS/CS ctrl cube: OpenInspectionPanel(ctrl_panel)
  - load-module components (bulb, diode): InvertEnclosure(cube_load)
  - DCAC: uses prefixed cube IDs (cube_ctrl1, cube_ctrl3, cube_psu1, cube_psu2, cube_load1/2)
  - 10CC:  uses prefixed cube IDs (cube_ctrl1…cube_ctrl8)
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "diagnosable-systems-simulation"))

from diagnosable_systems_simulation.actions.diagnostic_actions import (
    AdjustPotentiometer,
    InspectConnections,
    InvertEnclosure,
    MeasureVoltage,
    ObserveComponent,
    OpenInspectionPanel,
    OpenPeephole,
    ReplaceComponent,
    RotateEnclosure,
    TestContinuity,
    TestDiode,
)
from diagnosable_systems_simulation.actions.fault_actions import (
    ReconnectCable,
    ReverseBattery,
    SwapCablePolarities,
)

from Tests.conftest import assert_system_broken, build_system_for, run_sequence


# Reusable part-ID strings (arbitrary labels consumed as resources)
_BATTERY    = "battery_unit"
_CABLE      = "cable_unit"
_BULB       = "bulb_unit"
_LED        = "led_unit"
_RELAY      = "relay_unit"
_DIODE      = "diode_unit"


# ===========================================================================
# 3 CUBES CHAIN  (scenarios 1–25)
# ===========================================================================

# ── Simple (S) ──────────────────────────────────────────────────────────────

def test_3cc_s1():
    s = build_system_for("3CC.S.1")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery",    "part": _BATTERY}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_s1_repair():
    s = build_system_for("3CC.S.1")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_s2():
    s = build_system_for("3CC.S.2")
    run_sequence(s, [
        (InvertEnclosure, {"subject": "cube_psu"}),
        (MeasureVoltage,  {"subject": "battery"}),
        (ReverseBattery,  {"subject": "battery"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_s2_repair():
    s = build_system_for("3CC.S.2")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_s3():
    s = build_system_for("3CC.S.3")
    run_sequence(s, [
        (MeasureVoltage,                              {"subject": "ctrl_cable_in_pos"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_s3_repair():
    s = build_system_for("3CC.S.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})

def test_3cc_s4():
    s = build_system_for("3CC.S.4")
    run_sequence(s, [
        (OpenPeephole,       {"subject": "load_peephole"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_ctrl"}),
        (InspectConnections, {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,   {"subject": "ctrl_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_s4_repair():
    s = build_system_for("3CC.S.4")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_out_pos"})

def test_3cc_s5():
    s = build_system_for("3CC.S.5")
    run_sequence(s, [
        (OpenPeephole,       {"subject": "load_peephole"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load"}),
        (TestContinuity,     {"subject": "main_bulb"}),
        (ReplaceComponent,   {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_s5_repair():
    s = build_system_for("3CC.S.5")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})

# ── Double (D) ───────────────────────────────────────────────────────────────

def test_3cc_d1():
    s = build_system_for("3CC.D.1")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery", "part": _BATTERY}),
        (ObserveComponent, {"subject": "psu_green_led"}),
        (ObserveComponent, {"subject": "ctrl_red_led"}),
        (MeasureVoltage,   {"subject": "ctrl_cable_in_pos"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_d1_repair():
    s = build_system_for("3CC.D.1")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})
    assert s.test_repair({"battery", "ctrl_cable_in_pos", "ctrl_cable_in_neg"})

def test_3cc_d2():
    s = build_system_for("3CC.D.2")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_psu"}),
        (MeasureVoltage,     {"subject": "battery"}),
        (ReplaceComponent,   {"subject": "battery", "part": _BATTERY}),
        (ObserveComponent,   {"subject": "psu_green_led"}),
        (ObserveComponent,   {"subject": "ctrl_red_led"}),
        (OpenPeephole,       {"subject": "load_peephole"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_ctrl"}),
        (InspectConnections, {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,     {"subject": "ctrl_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_d2_repair():
    s = build_system_for("3CC.D.2")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"ctrl_cable_out_pos"})
    assert s.test_repair({"battery", "ctrl_cable_out_pos"})

def test_3cc_d3():
    s = build_system_for("3CC.D.3")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery", "part": _BATTERY}),
        (ObserveComponent, {"subject": "psu_green_led"}),
        (ObserveComponent, {"subject": "ctrl_red_led"}),
        (OpenPeephole,     {"subject": "load_peephole"}),
        (ObserveComponent, {"subject": "internal_bulb"}),
        (InvertEnclosure,  {"subject": "cube_load"}),
        (TestContinuity,   {"subject": "main_bulb"}),
        (ReplaceComponent, {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_d3_repair():
    s = build_system_for("3CC.D.3")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"main_bulb"})
    assert s.test_repair({"battery", "main_bulb"})

def test_3cc_d4():
    s = build_system_for("3CC.D.4")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_psu"}),
        (MeasureVoltage,     {"subject": "battery"}),
        (ReverseBattery,     {"subject": "battery"}),
        (ObserveComponent,   {"subject": "psu_green_led"}),
        (ObserveComponent,   {"subject": "ctrl_red_led"}),
        (OpenPeephole,       {"subject": "load_peephole"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_ctrl"}),
        (InspectConnections, {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,     {"subject": "ctrl_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_d4_repair():
    s = build_system_for("3CC.D.4")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"ctrl_cable_out_pos"})
    assert s.test_repair({"battery", "ctrl_cable_out_pos"})

def test_3cc_d5():
    s = build_system_for("3CC.D.5")
    run_sequence(s, [
        (MeasureVoltage,     {"subject": "ctrl_cable_in_pos"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
        (ObserveComponent,   {"subject": "psu_green_led"}),
        (ObserveComponent,   {"subject": "ctrl_red_led"}),
        (OpenPeephole,       {"subject": "load_peephole"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_ctrl"}),
        (InspectConnections, {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,     {"subject": "ctrl_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_d5_repair():
    s = build_system_for("3CC.D.5")
    assert_system_broken(s)
    assert not s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})
    assert not s.test_repair({"ctrl_cable_out_pos"})
    assert s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg", "ctrl_cable_out_pos"})

# ── Misleading (M) ───────────────────────────────────────────────────────────

def test_3cc_m1():
    # Reversed battery + open ctrl LED: LED off is misleading (looks like depleted)
    s = build_system_for("3CC.M.1")
    run_sequence(s, [
        (InvertEnclosure, {"subject": "cube_psu"}),
        (MeasureVoltage,  {"subject": "battery"}),
        (ReverseBattery,  {"subject": "battery"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_m1_repair():
    s = build_system_for("3CC.M.1")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_m2():
    # Reversed battery + open load diode + burned lamp
    s = build_system_for("3CC.M.2")
    run_sequence(s, [
        (InvertEnclosure, {"subject": "cube_psu"}),
        (MeasureVoltage,  {"subject": "battery"}),
        (ReverseBattery,  {"subject": "battery"}),
        (ObserveComponent, {"subject": "psu_green_led"}),
        (ObserveComponent, {"subject": "ctrl_red_led"}),
        (OpenPeephole,     {"subject": "load_peephole"}),
        (ObserveComponent, {"subject": "internal_bulb"}),
        (InvertEnclosure,  {"subject": "cube_load"}),
        (TestContinuity,   {"subject": "main_bulb"}),
        (ReplaceComponent, {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_m2_repair():
    s = build_system_for("3CC.M.2")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    # shorted diode bypasses polarity protection, so lamp repair alone restores the circuit
    assert s.test_repair({"main_bulb"})
    assert s.test_repair({"battery", "main_bulb"})

def test_3cc_m3():
    # Crossed cables + open ctrl LED: LED off hides the crossing
    s = build_system_for("3CC.M.3")
    run_sequence(s, [
        (OpenPeephole,       {"subject": "load_peephole"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_ctrl"}),
        (InspectConnections, {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl_cable_in_pos"}),
        (TestContinuity,     {"subject": "ctrl_switch"}),
        (MeasureVoltage,     {"subject": "ctrl_cable_in_pos"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_m3_repair():
    s = build_system_for("3CC.M.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})

def test_3cc_m4():
    # Burned lamp + open supply LED: supply LED off is misleading
    s = build_system_for("3CC.M.4")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (TestContinuity,   {"subject": "psu_green_led"}),
        (ReplaceComponent, {"subject": "psu_green_led", "part": _LED}),
        (ObserveComponent, {"subject": "psu_green_led"}),
        (ObserveComponent, {"subject": "ctrl_red_led"}),
        (OpenPeephole,     {"subject": "load_peephole"}),
        (ObserveComponent, {"subject": "internal_bulb"}),
        (InvertEnclosure,  {"subject": "cube_load"}),
        (TestContinuity,   {"subject": "main_bulb"}),
        (ReplaceComponent, {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_m4_repair():
    s = build_system_for("3CC.M.4")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})

def test_3cc_m5():
    # Burned lamp + burned internal (load) indicator (both must be replaced)
    s = build_system_for("3CC.M.5")
    run_sequence(s, [
        (OpenPeephole,       {"subject": "load_peephole"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_ctrl"}),
        (InspectConnections, {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl_cable_in_pos"}),
        (TestContinuity,     {"subject": "ctrl_switch"}),
        (MeasureVoltage,     {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,     {"subject": "load_cable_pos"}),
        (InvertEnclosure,    {"subject": "cube_load"}),
        (TestContinuity,     {"subject": "main_bulb"}),
        (ReplaceComponent,   {"subject": "main_bulb",     "part": _BULB}),
        (ReplaceComponent,   {"subject": "internal_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_m5_repair():
    s = build_system_for("3CC.M.5")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})           # main function restored — sufficient
    assert not s.test_repair({"internal_bulb"})   # indicator only — main lamp still burned
    assert s.test_repair({"main_bulb", "internal_bulb"})

# ── Limited Observability (L) ─────────────────────────────────────────────────

def test_3cc_l1():
    s = build_system_for("3CC.L.1")
    run_sequence(s, [
        (MeasureVoltage,   {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,   {"subject": "psu_cable_pos"}),
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery", "part": _BATTERY}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_l1_repair():
    s = build_system_for("3CC.L.1")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_l2():
    s = build_system_for("3CC.L.2")
    run_sequence(s, [
        (MeasureVoltage,  {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,  {"subject": "psu_cable_pos"}),
        (InvertEnclosure, {"subject": "cube_psu"}),
        (MeasureVoltage,  {"subject": "battery"}),
        (ReverseBattery,  {"subject": "battery"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_l2_repair():
    s = build_system_for("3CC.L.2")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_l3():
    s = build_system_for("3CC.L.3")
    run_sequence(s, [
        (MeasureVoltage,   {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,   {"subject": "psu_cable_pos"}),
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_l3_repair():
    s = build_system_for("3CC.L.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})

def test_3cc_l4():
    s = build_system_for("3CC.L.4")
    run_sequence(s, [
        (MeasureVoltage,     {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,     {"subject": "ctrl_cable_out_pos"}),
        (InvertEnclosure,    {"subject": "cube_ctrl"}),
        (InspectConnections, {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,   {"subject": "ctrl_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_l4_repair():
    s = build_system_for("3CC.L.4")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_out_pos"})

def test_3cc_l5():
    s = build_system_for("3CC.L.5")
    run_sequence(s, [
        (MeasureVoltage,     {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,     {"subject": "ctrl_cable_out_pos"}),
        (MeasureVoltage,     {"subject": "load_cable_pos"}),
        (InvertEnclosure,    {"subject": "cube_load"}),
        (InspectConnections, {"subject": "load_cable_pos"}),
        (TestContinuity,     {"subject": "main_bulb"}),
        (ReplaceComponent,   {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_l5_repair():
    s = build_system_for("3CC.L.5")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})

# ── Intermittent (I) ─────────────────────────────────────────────────────────

def test_3cc_i1():
    s = build_system_for("3CC.I.1")
    run_sequence(s, [
        (InspectConnections, {"subject": "psu_cable_pos"}),
        (ReplaceComponent,   {"subject": "psu_cable_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_i1_repair():
    s = build_system_for("3CC.I.1")
    assert_system_broken(s)
    assert s.test_repair({"psu_cable_pos"})

def test_3cc_i2():
    s = build_system_for("3CC.I.2")
    run_sequence(s, [
        (InspectConnections, {"subject": "ctrl_cable_in_pos"}),
        (ReplaceComponent,   {"subject": "ctrl_cable_in_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_i2_repair():
    s = build_system_for("3CC.I.2")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_in_pos"})

def test_3cc_i3():
    s = build_system_for("3CC.I.3")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl"}),
        (InspectConnections, {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl_cable_in_pos"}),
        (ReplaceComponent,   {"subject": "ctrl_cable_out_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_i3_repair():
    s = build_system_for("3CC.I.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_out_pos"})

def test_3cc_i4():
    s = build_system_for("3CC.I.4")
    run_sequence(s, [
        (InspectConnections, {"subject": "load_cable_pos"}),
        (ReplaceComponent,   {"subject": "load_cable_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_i4_repair():
    s = build_system_for("3CC.I.4")
    assert_system_broken(s)
    assert s.test_repair({"load_cable_pos"})

def test_3cc_i5():
    s = build_system_for("3CC.I.5")
    run_sequence(s, [
        (InspectConnections, {"subject": "load_cable_neg"}),
        (ReplaceComponent,   {"subject": "load_cable_neg", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_i5_repair():
    s = build_system_for("3CC.I.5")
    assert_system_broken(s)
    assert s.test_repair({"load_cable_neg"})

# ===========================================================================
# 10 CUBES CHAIN  (scenarios 26–50)
# ===========================================================================

# ── Simple (S) ──────────────────────────────────────────────────────────────

def test_10cc_s1():
    s = build_system_for("10CC.S.1")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery", "part": _BATTERY}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_s1_repair():
    s = build_system_for("10CC.S.1")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_10cc_s2():
    s = build_system_for("10CC.S.2")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl3"}),
        (InspectConnections, {"subject": "ctrl3_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl3_cable_out_pos"}),
        (ReconnectCable,   {"subject": "ctrl3_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_s2_repair():
    s = build_system_for("10CC.S.2")
    assert_system_broken(s)
    assert s.test_repair({"ctrl3_cable_in_pos"})

def test_10cc_s3():
    s = build_system_for("10CC.S.3")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl6"}),
        (InspectConnections, {"subject": "ctrl6_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl6_cable_out_pos"}),
        (ReconnectCable,   {"subject": "ctrl6_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_s3_repair():
    s = build_system_for("10CC.S.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl6_cable_in_pos"})

def test_10cc_s4():
    s = build_system_for("10CC.S.4")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl8"}),
        (InspectConnections, {"subject": "ctrl8_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl8_cable_out_pos"}),
        (ReconnectCable,   {"subject": "ctrl8_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_s4_repair():
    s = build_system_for("10CC.S.4")
    assert_system_broken(s)
    assert s.test_repair({"ctrl8_cable_in_pos"})

def test_10cc_s5():
    s = build_system_for("10CC.S.5")
    run_sequence(s, [
        (OpenPeephole,       {"subject": "load_peephole"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,  {"subject": "cube_load"}),
        (TestContinuity,   {"subject": "main_bulb"}),
        (ReplaceComponent, {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_s5_repair():
    s = build_system_for("10CC.S.5")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})

# ── Double (D) ───────────────────────────────────────────────────────────────

def test_10cc_d1():
    s = build_system_for("10CC.D.1")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_psu"}),
        (MeasureVoltage,     {"subject": "battery"}),
        (ReplaceComponent,   {"subject": "battery", "part": _BATTERY}),
        (ObserveComponent,   {"subject": "psu_green_led"}),
        (ObserveComponent,   {"subject": "ctrl1_green_led"}),
        (ObserveComponent,   {"subject": "ctrl2_green_led"}),
        (ObserveComponent,   {"subject": "ctrl3_green_led"}), # further LED observations could be skipped 
        (ObserveComponent,   {"subject": "ctrl4_green_led"}),
        (ObserveComponent,   {"subject": "ctrl5_green_led"}),
        (ObserveComponent,   {"subject": "ctrl6_green_led"}),
        (ObserveComponent,   {"subject": "ctrl7_green_led"}),
        (ObserveComponent,   {"subject": "ctrl8_green_led"}),
        (InvertEnclosure,    {"subject": "cube_ctrl3"}),
        (InspectConnections, {"subject": "ctrl3_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl3_cable_out_pos"}),
        (ReconnectCable,     {"subject": "ctrl3_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_d1_repair():
    s = build_system_for("10CC.D.1")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"ctrl3_cable_in_pos"})
    assert s.test_repair({"battery", "ctrl3_cable_in_pos"})

def test_10cc_d2():
    s = build_system_for("10CC.D.2")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_psu"}),
        (MeasureVoltage,     {"subject": "battery"}),
        (ReplaceComponent,   {"subject": "battery", "part": _BATTERY}),
        (ObserveComponent,   {"subject": "psu_green_led"}),
        (ObserveComponent,   {"subject": "ctrl1_green_led"}),
        (ObserveComponent,   {"subject": "ctrl2_green_led"}),
        (ObserveComponent,   {"subject": "ctrl3_green_led"}),
        (ObserveComponent,   {"subject": "ctrl4_green_led"}),
        (ObserveComponent,   {"subject": "ctrl5_green_led"}),
        (ObserveComponent,   {"subject": "ctrl6_green_led"}),
        (ObserveComponent,   {"subject": "ctrl7_green_led"}),
        (ObserveComponent,   {"subject": "ctrl8_green_led"}),
        (InvertEnclosure,    {"subject": "cube_ctrl8"}),
        (InspectConnections, {"subject": "ctrl8_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl8_cable_out_pos"}),
        (ReconnectCable,     {"subject": "ctrl8_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_d2_repair():
    s = build_system_for("10CC.D.2")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"ctrl8_cable_in_pos"})
    assert s.test_repair({"battery", "ctrl8_cable_in_pos"})

def test_10cc_d3():
    s = build_system_for("10CC.D.3")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl3"}),
        (InspectConnections, {"subject": "ctrl3_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl3_cable_out_pos"}),
        (ReconnectCable,     {"subject": "ctrl3_cable_in_pos"}),
        (ObserveComponent,   {"subject": "psu_green_led"}),
        (ObserveComponent,   {"subject": "ctrl1_green_led"}),
        (ObserveComponent,   {"subject": "ctrl2_green_led"}),
        (ObserveComponent,   {"subject": "ctrl3_green_led"}),
        (ObserveComponent,   {"subject": "ctrl4_green_led"}),
        (ObserveComponent,   {"subject": "ctrl5_green_led"}),
        (ObserveComponent,   {"subject": "ctrl6_green_led"}),
        (ObserveComponent,   {"subject": "ctrl7_green_led"}),
        (ObserveComponent,   {"subject": "ctrl8_green_led"}),
        (InvertEnclosure,    {"subject": "cube_ctrl8"}),
        (InspectConnections, {"subject": "ctrl8_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl8_cable_out_pos"}),
        (ReconnectCable,     {"subject": "ctrl8_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_d3_repair():
    s = build_system_for("10CC.D.3")
    assert_system_broken(s)
    assert not s.test_repair({"ctrl3_cable_in_pos"})
    assert not s.test_repair({"ctrl8_cable_in_pos"})
    assert s.test_repair({"ctrl3_cable_in_pos", "ctrl8_cable_in_pos"})

def test_10cc_d4():
    s = build_system_for("10CC.D.4")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl3"}),
        (InspectConnections, {"subject": "ctrl3_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl3_cable_out_pos"}),
        (ReconnectCable,     {"subject": "ctrl3_cable_in_pos"}),
        (ObserveComponent,   {"subject": "psu_green_led"}),
        (ObserveComponent,   {"subject": "ctrl1_green_led"}),
        (ObserveComponent,   {"subject": "ctrl2_green_led"}),
        (ObserveComponent,   {"subject": "ctrl3_green_led"}),
        (ObserveComponent,   {"subject": "ctrl4_green_led"}),
        (ObserveComponent,   {"subject": "ctrl5_green_led"}),
        (ObserveComponent,   {"subject": "ctrl6_green_led"}),
        (ObserveComponent,   {"subject": "ctrl7_green_led"}),
        (ObserveComponent,   {"subject": "ctrl8_green_led"}),
        (OpenPeephole,       {"subject": "load_peephole"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load"}),
        (TestContinuity,     {"subject": "main_bulb"}),
        (ReplaceComponent,   {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_d4_repair():
    s = build_system_for("10CC.D.4")
    assert_system_broken(s)
    assert not s.test_repair({"ctrl3_cable_in_pos"})
    assert not s.test_repair({"main_bulb"})
    assert s.test_repair({"ctrl3_cable_in_pos", "main_bulb"})

def test_10cc_d5():
    s = build_system_for("10CC.D.5")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl8"}),
        (InspectConnections, {"subject": "ctrl8_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl8_cable_out_pos"}),
        (ReconnectCable,     {"subject": "ctrl8_cable_in_pos"}),
        (ObserveComponent,   {"subject": "psu_green_led"}),
        (ObserveComponent,   {"subject": "ctrl1_green_led"}),
        (ObserveComponent,   {"subject": "ctrl2_green_led"}),
        (ObserveComponent,   {"subject": "ctrl3_green_led"}),
        (ObserveComponent,   {"subject": "ctrl4_green_led"}),
        (ObserveComponent,   {"subject": "ctrl5_green_led"}),
        (ObserveComponent,   {"subject": "ctrl6_green_led"}),
        (ObserveComponent,   {"subject": "ctrl7_green_led"}),
        (ObserveComponent,   {"subject": "ctrl8_green_led"}),
        (OpenPeephole,       {"subject": "load_peephole"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load"}),
        (TestContinuity,     {"subject": "main_bulb"}),
        (ReplaceComponent,   {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_d5_repair():
    s = build_system_for("10CC.D.5")
    assert_system_broken(s)
    assert not s.test_repair({"ctrl8_cable_in_pos"})
    assert not s.test_repair({"main_bulb"})
    assert s.test_repair({"ctrl8_cable_in_pos", "main_bulb"})

# ── Misleading (M) ───────────────────────────────────────────────────────────

def test_10cc_m1():
    # Reversed battery + reversed supply LED (LED shows ON misleadingly)
    s = build_system_for("10CC.M.1")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl1"}),
        (InspectConnections, {"subject": "ctrl1_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (TestContinuity,     {"subject": "ctrl1_switch"}),
        (MeasureVoltage,     {"subject": "ctrl1_cable_in_pos"}),
        (InvertEnclosure,    {"subject": "cube_psu"}),
        (MeasureVoltage,     {"subject": "battery"}),
        (ReplaceComponent,   {"subject": "battery", "part": _BATTERY}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_m1_repair():
    s = build_system_for("10CC.M.1")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_10cc_m2():
    # Disconnected switch 3 + ctrl2 LED open (LED shows off one module too early)
    s = build_system_for("10CC.M.2")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl2"}),
        (InspectConnections, {"subject": "ctrl2_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl2_cable_out_pos"}),
        (TestContinuity,     {"subject": "ctrl2_switch"}),
        (MeasureVoltage,     {"subject": "ctrl2_cable_out_pos"}),
        (InvertEnclosure,    {"subject": "cube_ctrl3"}),
        (InspectConnections, {"subject": "ctrl3_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl3_cable_out_pos"}),
        (ReconnectCable,   {"subject": "ctrl3_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_m2_repair():
    s = build_system_for("10CC.M.2")
    assert_system_broken(s)
    assert s.test_repair({"ctrl3_cable_in_pos"})

def test_10cc_m3():
    s = build_system_for("10CC.M.3")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl5"}),
        (InspectConnections, {"subject": "ctrl5_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl5_cable_out_pos"}),
        (TestContinuity,     {"subject": "ctrl5_switch"}),
        (MeasureVoltage,     {"subject": "ctrl5_cable_out_pos"}),
        (InvertEnclosure,    {"subject": "cube_ctrl6"}),
        (InspectConnections, {"subject": "ctrl6_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl6_cable_out_pos"}),
        (ReconnectCable,   {"subject": "ctrl6_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_m3_repair():
    s = build_system_for("10CC.M.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl6_cable_in_pos"})

def test_10cc_m4():
    s = build_system_for("10CC.M.4")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl7"}),
        (InspectConnections, {"subject": "ctrl7_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl7_cable_out_pos"}),
        (TestContinuity,     {"subject": "ctrl7_switch"}),
        (MeasureVoltage,     {"subject": "ctrl7_cable_out_pos"}),
        (InvertEnclosure,    {"subject": "cube_ctrl8"}),
        (InspectConnections, {"subject": "ctrl8_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl8_cable_out_pos"}),
        (ReconnectCable,   {"subject": "ctrl8_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_m4_repair():
    s = build_system_for("10CC.M.4")
    assert_system_broken(s)
    assert s.test_repair({"ctrl8_cable_in_pos"})

def test_10cc_m5():
    # Burned lamp + burned load indicator (both must be replaced)
    s = build_system_for("10CC.M.5")
    run_sequence(s, [
        (OpenPeephole,       {"subject": "load_peephole"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_ctrl8"}),
        (InspectConnections, {"subject": "ctrl8_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl8_cable_in_pos"}),
        (MeasureVoltage,     {"subject": "load_cable_pos"}),
        (InvertEnclosure,    {"subject": "cube_load"}),
        (TestContinuity,     {"subject": "main_bulb"}),
        (ReplaceComponent,   {"subject": "main_bulb",     "part": _BULB}),
        (ReplaceComponent,   {"subject": "internal_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_m5_repair():
    s = build_system_for("10CC.M.5")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})           # main function restored — sufficient
    assert not s.test_repair({"internal_bulb"})   # indicator only — main lamp still burned
    assert s.test_repair({"main_bulb", "internal_bulb"})

# ── Limited Observability (L) ─────────────────────────────────────────────────

def test_10cc_l1():
    s = build_system_for("10CC.L.1")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery", "part": _BATTERY}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_l1_repair():
    s = build_system_for("10CC.L.1")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_10cc_l2():
    s = build_system_for("10CC.L.2")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl3"}),
        (MeasureVoltage,     {"subject": "ctrl3_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl3_cable_in_pos"}),
        (ReconnectCable,   {"subject": "ctrl3_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_l2_repair():
    s = build_system_for("10CC.L.2")
    assert_system_broken(s)
    assert s.test_repair({"ctrl3_cable_in_pos"})

def test_10cc_l3():
    s = build_system_for("10CC.L.3")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl6"}),
        (MeasureVoltage,     {"subject": "ctrl6_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl6_cable_in_pos"}),
        (ReconnectCable,   {"subject": "ctrl6_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_l3_repair():
    s = build_system_for("10CC.L.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl6_cable_in_pos"})

def test_10cc_l4():
    s = build_system_for("10CC.L.4")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl8"}),
        (MeasureVoltage,     {"subject": "ctrl8_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl8_cable_in_pos"}),
        (ReconnectCable,   {"subject": "ctrl8_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_l4_repair():
    s = build_system_for("10CC.L.4")
    assert_system_broken(s)
    assert s.test_repair({"ctrl8_cable_in_pos"})

def test_10cc_l5():
    s = build_system_for("10CC.L.5")
    run_sequence(s, [
        (MeasureVoltage,   {"subject": "ctrl8_cable_out_pos"}),
        (InvertEnclosure,  {"subject": "cube_load"}),
        (TestContinuity,   {"subject": "main_bulb"}),
        (ReplaceComponent, {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_l5_repair():
    s = build_system_for("10CC.L.5")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})

# ── Intermittent (I) ─────────────────────────────────────────────────────────

def test_10cc_i1():
    s = build_system_for("10CC.I.1")
    run_sequence(s, [
        (InspectConnections, {"subject": "psu_cable_pos"}),
        (ReplaceComponent,   {"subject": "psu_cable_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_i1_repair():
    s = build_system_for("10CC.I.1")
    assert_system_broken(s)
    assert s.test_repair({"psu_cable_pos"})

def test_10cc_i2():
    s = build_system_for("10CC.I.2")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl3"}),
        (InspectConnections, {"subject": "ctrl3_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl3_cable_out_pos"}),
        (ReplaceComponent,   {"subject": "ctrl3_cable_in_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_i2_repair():
    s = build_system_for("10CC.I.2")
    assert_system_broken(s)
    assert s.test_repair({"ctrl3_cable_in_pos"})

def test_10cc_i3():
    s = build_system_for("10CC.I.3")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl6"}),
        (InspectConnections, {"subject": "ctrl6_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl6_cable_out_pos"}),
        (ReplaceComponent,   {"subject": "ctrl6_cable_in_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_i3_repair():
    s = build_system_for("10CC.I.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl6_cable_in_pos"})

def test_10cc_i4():
    s = build_system_for("10CC.I.4")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl8"}),
        (InspectConnections, {"subject": "ctrl8_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl8_cable_out_pos"}),
        (ReplaceComponent,   {"subject": "ctrl8_cable_in_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_i4_repair():
    s = build_system_for("10CC.I.4")
    assert_system_broken(s)
    assert s.test_repair({"ctrl8_cable_in_pos"})

def test_10cc_i5():
    s = build_system_for("10CC.I.5")
    run_sequence(s, [
        (InspectConnections, {"subject": "load_cable_neg"}),
        (ReplaceComponent,   {"subject": "load_cable_neg", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_10cc_i5_repair():
    s = build_system_for("10CC.I.5")
    assert_system_broken(s)
    assert s.test_repair({"load_cable_neg"})

# ===========================================================================
# DOUBLY CONNECTED ASYMMETRIC CHAINS  (scenarios 51–75)
# ===========================================================================

# ── Simple (S) ──────────────────────────────────────────────────────────────

def test_dcac_s1():
    s = build_system_for("DCAC.S.1")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl1"}),
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl1_cable_in_pos"}),
        (ReconnectCable,   {"subject": "ctrl1_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_s1_repair():
    s = build_system_for("DCAC.S.1")
    assert_system_broken(s)
    assert s.test_repair({"ctrl1_cable_out_pos"})

def test_dcac_s2():
    s = build_system_for("DCAC.S.2")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl3"}),
        (InspectConnections, {"subject": "ctrl3_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl3_cable_out_pos"}),
        (ReconnectCable,   {"subject": "ctrl3_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_s2_repair():
    s = build_system_for("DCAC.S.2")
    assert_system_broken(s)
    assert s.test_repair({"ctrl3_cable_in_pos"})

def test_dcac_s3():
    s = build_system_for("DCAC.S.3")
    run_sequence(s, [
        (OpenPeephole,       {"subject": "load1_load_peephole"}),
        (ObserveComponent,   {"subject": "load1_internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_ctrl1"}),
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl1_cable_in_pos"}),
        (TestContinuity,     {"subject": "ctrl1_switch"}),
        (InvertEnclosure,    {"subject": "cube_load1"}),
        (InspectConnections, {"subject": "load1_load_cable_pos"}),
        (TestDiode,          {"subject": "load1_load_diode"}),
        (ReplaceComponent,   {"subject": "load1_load_diode", "part": _DIODE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_s3_repair():
    s = build_system_for("DCAC.S.3")
    assert_system_broken(s)
    assert s.test_repair({"load1_load_diode"})

def test_dcac_s4():
    s = build_system_for("DCAC.S.4")
    run_sequence(s, [
        (OpenPeephole,       {"subject": "load1_load_peephole"}),
        (ObserveComponent,   {"subject": "load1_internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load1"}),
        (TestContinuity,     {"subject": "load1_main_bulb"}),
        (ReplaceComponent,   {"subject": "load1_main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_s4_repair():
    s = build_system_for("DCAC.S.4")
    assert_system_broken(s)
    assert s.test_repair({"load1_main_bulb"})

def test_dcac_s5():
    s = build_system_for("DCAC.S.5")
    run_sequence(s, [
        (OpenPeephole,       {"subject": "load2_load_peephole"}),
        (ObserveComponent,   {"subject": "load2_internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load2"}),
        (TestContinuity,     {"subject": "load2_main_bulb"}),
        (ReplaceComponent,   {"subject": "load2_main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_s5_repair():
    s = build_system_for("DCAC.S.5")
    assert_system_broken(s)
    assert s.test_repair({"load2_main_bulb"})

# ── Double (D) ───────────────────────────────────────────────────────────────

def test_dcac_d1():
    s = build_system_for("DCAC.D.1")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu1"}),
        (MeasureVoltage,   {"subject": "psu1_battery"}),
        (ReplaceComponent, {"subject": "psu1_battery", "part": _BATTERY}),
        (InvertEnclosure,  {"subject": "cube_psu2"}),
        (MeasureVoltage,   {"subject": "psu2_battery"}),
        (ReplaceComponent, {"subject": "psu2_battery", "part": _BATTERY}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_d1_repair():
    s = build_system_for("DCAC.D.1")
    assert_system_broken(s)
    assert s.test_repair({"psu1_battery", "psu2_battery"})

def test_dcac_d2():
    s = build_system_for("DCAC.D.2")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_psu2"}),
        (MeasureVoltage,     {"subject": "psu2_battery"}),
        (ReplaceComponent,   {"subject": "psu2_battery", "part": _BATTERY}),
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (InvertEnclosure,    {"subject": "cube_ctrl1"}),
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl1_cable_in_pos"}),
        (ReconnectCable,     {"subject": "ctrl1_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_d2_repair():
    s = build_system_for("DCAC.D.2")
    assert_system_broken(s)
    assert not s.test_repair({"psu2_battery"})
    assert s.test_repair({"psu2_battery", "ctrl1_cable_out_pos"})

def test_dcac_d3():
    s = build_system_for("DCAC.D.3")
    run_sequence(s, [
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl2_cable_out_pos"}),
        (InvertEnclosure,    {"subject": "cube_ctrl1"}),
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl1_cable_in_pos"}),
        (ReconnectCable,     {"subject": "ctrl1_cable_out_pos"}),
        (InvertEnclosure,    {"subject": "cube_ctrl2"}),
        (InspectConnections, {"subject": "ctrl2_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl2_cable_in_pos"}),
        (ReconnectCable,     {"subject": "ctrl2_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_d3_repair():
    s = build_system_for("DCAC.D.3")
    assert_system_broken(s)
    assert not s.test_repair({"ctrl2_cable_out_pos"})
    assert s.test_repair({"ctrl1_cable_out_pos", "ctrl2_cable_out_pos"})

def test_dcac_d4():
    s = build_system_for("DCAC.D.4")
    run_sequence(s, [
        (InspectConnections, {"subject": "ctrl3_cable_in_pos"}),
        (InspectConnections, {"subject": "load1_load_cable_pos"}),
        (InvertEnclosure,    {"subject": "cube_ctrl3"}),
        (InspectConnections, {"subject": "ctrl3_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl3_cable_out_pos"}),
        (ReconnectCable,     {"subject": "ctrl3_cable_in_pos"}),
        (OpenPeephole,       {"subject": "load1_load_peephole"}),
        (ObserveComponent,   {"subject": "load1_internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load1"}),
        (InspectConnections, {"subject": "load1_load_cable_pos"}),
        (TestContinuity,     {"subject": "load1_main_bulb"}),
        (ReplaceComponent,   {"subject": "load1_main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_d4_repair():
    s = build_system_for("DCAC.D.4")
    assert_system_broken(s)
    assert not s.test_repair({"ctrl3_cable_in_pos"})
    assert not s.test_repair({"load1_main_bulb"})
    assert s.test_repair({"ctrl3_cable_in_pos", "load1_main_bulb"})

def test_dcac_d5():
    s = build_system_for("DCAC.D.5")
    run_sequence(s, [
        (OpenPeephole,       {"subject": "load1_load_peephole"}),
        (ObserveComponent,   {"subject": "load1_internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load1"}),
        (TestContinuity,     {"subject": "load1_main_bulb"}),
        (ReplaceComponent,   {"subject": "load1_main_bulb", "part": _BULB}),
        (OpenPeephole,       {"subject": "load2_load_peephole"}),
        (ObserveComponent,   {"subject": "load2_internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load2"}),
        (TestContinuity,     {"subject": "load2_main_bulb"}),
        (ReplaceComponent,   {"subject": "load2_main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_d5_repair():
    s = build_system_for("DCAC.D.5")
    assert_system_broken(s)
    assert not s.test_repair({"load1_main_bulb"})
    assert not s.test_repair({"load2_main_bulb"})
    assert s.test_repair({"load1_main_bulb", "load2_main_bulb"})

# ── Misleading (M) ───────────────────────────────────────────────────────────

def test_dcac_m1():
    s = build_system_for("DCAC.M.1")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_psu1"}),
        (MeasureVoltage,     {"subject": "psu1_battery"}),
        (TestContinuity,     {"subject": "psu1_psu_green_led"}),
        (ReplaceComponent,   {"subject": "psu1_psu_green_led", "part": _LED}),
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (InvertEnclosure,    {"subject": "cube_ctrl1"}),
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl1_cable_in_pos"}),
        (ReconnectCable,     {"subject": "ctrl1_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_m1_repair():
    s = build_system_for("DCAC.M.1")
    assert_system_broken(s)
    assert s.test_repair({"ctrl1_cable_out_pos"})

def test_dcac_m2():
    s = build_system_for("DCAC.M.2")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_psu2"}),
        (MeasureVoltage,     {"subject": "psu2_battery"}),
        (TestContinuity,     {"subject": "psu2_psu_green_led"}),
        (ReplaceComponent,   {"subject": "psu2_psu_green_led", "part": _LED}),
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl2_cable_out_pos"}),
        (InvertEnclosure,    {"subject": "cube_ctrl1"}),
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl1_cable_in_pos"}),
        (ReconnectCable,     {"subject": "ctrl1_cable_out_pos"}),
        (InvertEnclosure,    {"subject": "cube_ctrl2"}),
        (InspectConnections, {"subject": "ctrl2_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl2_cable_in_pos"}),
        (ReconnectCable,     {"subject": "ctrl2_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_m2_repair():
    s = build_system_for("DCAC.M.2")
    assert_system_broken(s)
    assert not s.test_repair({"ctrl2_cable_out_pos"})
    assert s.test_repair({"ctrl1_cable_out_pos", "ctrl2_cable_out_pos"})

def test_dcac_m3():
    # Burned lamp1 + open load1 indicator lamp (both are Bulbs, both must be replaced)
    s = build_system_for("DCAC.M.3")
    run_sequence(s, [
        (OpenPeephole,       {"subject": "load1_load_peephole"}),
        (ObserveComponent,   {"subject": "load1_internal_bulb"}),
        (MeasureVoltage,     {"subject": "ctrl1_cable_out_pos"}),
        (MeasureVoltage,     {"subject": "load1_load_cable_pos"}),
        (InvertEnclosure,    {"subject": "cube_load1"}),
        (TestContinuity,     {"subject": "load1_main_bulb"}),
        (ReplaceComponent,   {"subject": "load1_main_bulb",     "part": _BULB}),
        (ReplaceComponent,   {"subject": "load1_internal_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_m3_repair():
    s = build_system_for("DCAC.M.3")
    assert_system_broken(s)
    assert s.test_repair({"load1_main_bulb"})           # main function restored — sufficient
    assert not s.test_repair({"load1_internal_bulb"})   # indicator only — main lamp still burned
    assert s.test_repair({"load1_main_bulb", "load1_internal_bulb"})

def test_dcac_m4():
    # Burned lamp2 + open supply2 LED: supply LED off is misleading
    s = build_system_for("DCAC.M.4")
    run_sequence(s, [
        (OpenPeephole,       {"subject": "load2_load_peephole"}),
        (ObserveComponent,   {"subject": "load2_internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load2"}),
        (TestContinuity,     {"subject": "load2_main_bulb"}),
        (ReplaceComponent,   {"subject": "load2_main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_m4_repair():
    s = build_system_for("DCAC.M.4")
    assert_system_broken(s)
    assert s.test_repair({"load2_main_bulb"})

def test_dcac_m5():
    # Burned lamp2 + ctrl3 LED open (suggests open in ctrl3, but lamp2 is actually dead)
    s = build_system_for("DCAC.M.5")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl3"}),
        (InspectConnections, {"subject": "ctrl3_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl3_cable_out_pos"}),
        (TestContinuity,     {"subject": "ctrl3_switch"}),
        (MeasureVoltage,     {"subject": "ctrl3_cable_out_pos"}),
        (InvertEnclosure,    {"subject": "cube_psu2"}),
        (TestContinuity,     {"subject": "ctrl3_green_led"}),
        (ReplaceComponent,   {"subject": "ctrl3_green_led", "part": _LED}),
        (OpenPeephole,       {"subject": "load2_load_peephole"}),
        (ObserveComponent,   {"subject": "load2_internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load2"}),
        (TestContinuity,     {"subject": "load2_main_bulb"}),
        (ReplaceComponent,   {"subject": "load2_main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_m5_repair():
    s = build_system_for("DCAC.M.5")
    assert_system_broken(s)
    assert s.test_repair({"load2_main_bulb"})

# ── Limited Observability (L) ─────────────────────────────────────────────────

def test_dcac_l1():
    s = build_system_for("DCAC.L.1")
    run_sequence(s, [
        (MeasureVoltage,     {"subject": "ctrl1_cable_out_pos"}),
        (InvertEnclosure,    {"subject": "cube_ctrl1"}),
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl1_cable_in_pos"}),
        (ReconnectCable,   {"subject": "ctrl1_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_l1_repair():
    s = build_system_for("DCAC.L.1")
    assert_system_broken(s)
    assert s.test_repair({"ctrl1_cable_out_pos"})

def test_dcac_l2():
    s = build_system_for("DCAC.L.2")
    run_sequence(s, [
        (MeasureVoltage,     {"subject": "ctrl3_cable_out_pos"}),
        (InvertEnclosure,    {"subject": "cube_ctrl3"}),
        (InspectConnections, {"subject": "ctrl3_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl3_cable_out_pos"}),
        (ReconnectCable,   {"subject": "ctrl3_cable_in_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_l2_repair():
    s = build_system_for("DCAC.L.2")
    assert_system_broken(s)
    assert s.test_repair({"ctrl3_cable_in_pos"})

def test_dcac_l3():
    s = build_system_for("DCAC.L.3")
    run_sequence(s, [
        (MeasureVoltage,     {"subject": "ctrl1_cable_out_pos"}),
        (MeasureVoltage,     {"subject": "load1_load_cable_pos"}),
        (OpenPeephole,       {"subject": "load1_load_peephole"}),
        (ObserveComponent,   {"subject": "load1_internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load1"}),
        (InspectConnections, {"subject": "load1_load_cable_pos"}),
        (TestDiode,          {"subject": "load1_load_diode"}),
        (ReplaceComponent,   {"subject": "load1_load_diode", "part": _DIODE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_l3_repair():
    s = build_system_for("DCAC.L.3")
    assert_system_broken(s)
    assert s.test_repair({"load1_load_diode"})

def test_dcac_l4():
    s = build_system_for("DCAC.L.4")
    run_sequence(s, [
        (MeasureVoltage,     {"subject": "ctrl1_cable_out_pos"}),
        (MeasureVoltage,     {"subject": "load1_load_cable_pos"}),
        (OpenPeephole,       {"subject": "load1_load_peephole"}),
        (ObserveComponent,   {"subject": "load1_internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load1"}),
        (TestContinuity,     {"subject": "load1_main_bulb"}),
        (ReplaceComponent,   {"subject": "load1_main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_l4_repair():
    s = build_system_for("DCAC.L.4")
    assert_system_broken(s)
    assert s.test_repair({"load1_main_bulb"})

def test_dcac_l5():
    s = build_system_for("DCAC.L.5")
    run_sequence(s, [
        (MeasureVoltage,     {"subject": "ctrl3_cable_out_pos"}),
        (MeasureVoltage,     {"subject": "load2_load_cable_pos"}),
        (OpenPeephole,       {"subject": "load2_load_peephole"}),
        (ObserveComponent,   {"subject": "load2_internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load2"}),
        (TestContinuity,     {"subject": "load2_main_bulb"}),
        (ReplaceComponent,   {"subject": "load2_main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_l5_repair():
    s = build_system_for("DCAC.L.5")
    assert_system_broken(s)
    assert s.test_repair({"load2_main_bulb"})

# ── Intermittent (I) ─────────────────────────────────────────────────────────

def test_dcac_i1():
    s = build_system_for("DCAC.I.1")
    run_sequence(s, [
        (InspectConnections, {"subject": "psu1_psu_cable_pos"}),
        (ReplaceComponent,   {"subject": "psu1_psu_cable_pos", "part": _CABLE}),
        (InspectConnections, {"subject": "psu2_psu_cable_pos"}),
        (ReplaceComponent,   {"subject": "psu2_psu_cable_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_i1_repair():
    s = build_system_for("DCAC.I.1")
    assert_system_broken(s)
    assert s.test_repair({"psu1_psu_cable_pos", "psu2_psu_cable_pos"})

def test_dcac_i2():
    s = build_system_for("DCAC.I.2")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl1"}),
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl1_cable_in_pos"}),
        (ReplaceComponent,   {"subject": "ctrl1_cable_out_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_i2_repair():
    s = build_system_for("DCAC.I.2")
    assert_system_broken(s)
    assert s.test_repair({"ctrl1_cable_out_pos"})

def test_dcac_i3():
    s = build_system_for("DCAC.I.3")
    run_sequence(s, [
        (InvertEnclosure,    {"subject": "cube_ctrl1"}),
        (InspectConnections, {"subject": "ctrl1_cable_out_pos"}),
        (InspectConnections, {"subject": "ctrl1_cable_in_pos"}),
        (ReplaceComponent,   {"subject": "ctrl1_cable_out_pos", "part": _CABLE}),
        (InvertEnclosure,    {"subject": "cube_ctrl3"}),
        (InspectConnections, {"subject": "ctrl3_cable_in_pos"}),
        (InspectConnections, {"subject": "ctrl3_cable_out_pos"}),
        (ReplaceComponent,   {"subject": "ctrl3_cable_in_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_i3_repair():
    s = build_system_for("DCAC.I.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl1_cable_out_pos", "ctrl3_cable_in_pos"})

def test_dcac_i4():
    s = build_system_for("DCAC.I.4")
    run_sequence(s, [
        (InspectConnections, {"subject": "load1_load_cable_neg"}),
        (ReplaceComponent,   {"subject": "load1_load_cable_neg", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_i4_repair():
    s = build_system_for("DCAC.I.4")
    assert_system_broken(s)
    assert s.test_repair({"load1_load_cable_neg"})

def test_dcac_i5():
    s = build_system_for("DCAC.I.5")
    run_sequence(s, [
        (InspectConnections, {"subject": "load2_load_cable_neg"}),
        (ReplaceComponent,   {"subject": "load2_load_cable_neg", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_dcac_i5_repair():
    s = build_system_for("DCAC.I.5")
    assert_system_broken(s)
    assert s.test_repair({"load2_load_cable_neg"})

# ===========================================================================
# 3 CUBES CHAIN WITH AMBIENT SENSOR  (scenarios 76–105)
# ALS uses OpenInspectionPanel(ctrl_panel) instead of InvertEnclosure for
# relay/internal access. Battery still requires InvertEnclosure(cube_psu).
# ===========================================================================

# ── Simple (S) ──────────────────────────────────────────────────────────────

def test_3cc_as_s1():
    s = build_system_for("3CC-AS.S.1")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery", "part": _BATTERY}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_s1_repair():
    s = build_system_for("3CC-AS.S.1")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_as_s2():
    s = build_system_for("3CC-AS.S.2")
    run_sequence(s, [
        (InvertEnclosure, {"subject": "cube_psu"}),
        (MeasureVoltage,  {"subject": "battery"}),
        (ReverseBattery,  {"subject": "battery"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_s2_repair():
    s = build_system_for("3CC-AS.S.2")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_as_s3():
    s = build_system_for("3CC-AS.S.3")
    run_sequence(s, [
        (MeasureVoltage,   {"subject": "ctrl_cable_in_pos"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_s3_repair():
    s = build_system_for("3CC-AS.S.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})

def test_3cc_as_s4():
    # Disconnected relay cable
    s = build_system_for("3CC-AS.S.4")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel,    {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,      {"subject": "ctrl_cable_out_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_s4_repair():
    s = build_system_for("3CC-AS.S.4")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_out_neg"})

def test_3cc_as_s5():
    s = build_system_for("3CC-AS.S.5")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load"}),
        (TestContinuity,     {"subject": "main_bulb"}),
        (ReplaceComponent,   {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_s5_repair():
    s = build_system_for("3CC-AS.S.5")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})

# ── Double (D) ───────────────────────────────────────────────────────────────

def test_3cc_as_d1():
    s = build_system_for("3CC-AS.D.1")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery", "part": _BATTERY}),
        (ObserveComponent, {"subject": "psu_green_led"}),
        (MeasureVoltage,   {"subject": "ctrl_cable_in_pos"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_d1_repair():
    s = build_system_for("3CC-AS.D.1")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})
    assert s.test_repair({"battery", "ctrl_cable_in_pos", "ctrl_cable_in_neg"})

def test_3cc_as_d2():
    s = build_system_for("3CC-AS.D.2")
    run_sequence(s, [
        (InvertEnclosure,     {"subject": "cube_psu"}),
        (MeasureVoltage,      {"subject": "battery"}),
        (ReplaceComponent,    {"subject": "battery", "part": _BATTERY}),
        (ObserveComponent,    {"subject": "psu_green_led"}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,      {"subject": "ctrl_cable_out_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_d2_repair():
    s = build_system_for("3CC-AS.D.2")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"ctrl_cable_out_neg"})
    assert s.test_repair({"battery", "ctrl_cable_out_neg"})

def test_3cc_as_d3():
    s = build_system_for("3CC-AS.D.3")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery", "part": _BATTERY}),
        (ObserveComponent, {"subject": "psu_green_led"}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,  {"subject": "cube_load"}),
        (TestContinuity,   {"subject": "main_bulb"}),
        (ReplaceComponent, {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_d3_repair():
    s = build_system_for("3CC-AS.D.3")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"main_bulb"})
    assert s.test_repair({"battery", "main_bulb"})

def test_3cc_as_d4():
    s = build_system_for("3CC-AS.D.4")
    run_sequence(s, [
        (InvertEnclosure,     {"subject": "cube_psu"}),
        (MeasureVoltage,      {"subject": "battery"}),
        (ReverseBattery,      {"subject": "battery"}),
        (ObserveComponent,    {"subject": "psu_green_led"}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,      {"subject": "ctrl_cable_out_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_d4_repair():
    s = build_system_for("3CC-AS.D.4")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"ctrl_cable_out_neg"})
    assert s.test_repair({"battery", "ctrl_cable_out_neg"})

def test_3cc_as_d5():
    s = build_system_for("3CC-AS.D.5")
    run_sequence(s, [
        (MeasureVoltage,      {"subject": "ctrl_cable_in_pos"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
        (ObserveComponent,    {"subject": "psu_green_led"}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,      {"subject": "ctrl_cable_out_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_d5_repair():
    s = build_system_for("3CC-AS.D.5")
    assert_system_broken(s)
    assert not s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})
    assert not s.test_repair({"ctrl_cable_out_neg"})
    assert s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg", "ctrl_cable_out_neg"})

# ── Misleading (M) ───────────────────────────────────────────────────────────

def test_3cc_as_m1():
    s = build_system_for("3CC-AS.M.1")
    run_sequence(s, [
        (InvertEnclosure, {"subject": "cube_psu"}),
        (MeasureVoltage,  {"subject": "battery"}),
        (ReverseBattery,  {"subject": "battery"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_m1_repair():
    s = build_system_for("3CC-AS.M.1")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_as_m2():
    s = build_system_for("3CC-AS.M.2")
    run_sequence(s, [
        (InvertEnclosure, {"subject": "cube_psu"}),
        (MeasureVoltage,  {"subject": "battery"}),
        (ReverseBattery,  {"subject": "battery"}),
        (ObserveComponent, {"subject": "psu_green_led"}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,  {"subject": "cube_load"}),
        (TestContinuity,   {"subject": "main_bulb"}),
        (ReplaceComponent, {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_m2_repair():
    s = build_system_for("3CC-AS.M.2")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    # shorted diode bypasses polarity protection, so lamp repair alone restores the circuit
    assert s.test_repair({"main_bulb"})
    assert s.test_repair({"battery", "main_bulb"})

def test_3cc_as_m3():
    s = build_system_for("3CC-AS.M.3")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (TestContinuity,      {"subject": "ctrl_relay"}),
        (MeasureVoltage,      {"subject": "ctrl_cable_in_pos"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_m3_repair():
    s = build_system_for("3CC-AS.M.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})

def test_3cc_as_m4():
    s = build_system_for("3CC-AS.M.4")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (TestContinuity,   {"subject": "psu_green_led"}),
        (ReplaceComponent, {"subject": "psu_green_led", "part": _LED}),
        (ObserveComponent, {"subject": "psu_green_led"}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,  {"subject": "cube_load"}),
        (TestContinuity,   {"subject": "main_bulb"}),
        (ReplaceComponent, {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_m4_repair():
    s = build_system_for("3CC-AS.M.4")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})

def test_3cc_as_m5():
    # Burned lamp + burned load indicator (both must be replaced)
    s = build_system_for("3CC-AS.M.5")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,    {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (TestContinuity,      {"subject": "ctrl_relay"}),
        (MeasureVoltage,      {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,      {"subject": "load_cable_pos"}),
        (InvertEnclosure,     {"subject": "cube_load"}),
        (TestContinuity,      {"subject": "main_bulb"}),
        (ReplaceComponent,    {"subject": "main_bulb",     "part": _BULB}),
        (ReplaceComponent,    {"subject": "internal_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_m5_repair():
    s = build_system_for("3CC-AS.M.5")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})           # main function restored — sufficient
    assert not s.test_repair({"internal_bulb"})   # indicator only — main lamp still burned
    assert s.test_repair({"main_bulb", "internal_bulb"})

# ── Limited Observability (L) ─────────────────────────────────────────────────

def test_3cc_as_l1():
    s = build_system_for("3CC-AS.L.1")
    run_sequence(s, [
        (MeasureVoltage,   {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,   {"subject": "psu_cable_pos"}),
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery", "part": _BATTERY}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_l1_repair():
    s = build_system_for("3CC-AS.L.1")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_as_l2():
    s = build_system_for("3CC-AS.L.2")
    run_sequence(s, [
        (MeasureVoltage,  {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,  {"subject": "psu_cable_pos"}),
        (InvertEnclosure, {"subject": "cube_psu"}),
        (MeasureVoltage,  {"subject": "battery"}),
        (ReverseBattery,  {"subject": "battery"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_l2_repair():
    s = build_system_for("3CC-AS.L.2")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_as_l3():
    s = build_system_for("3CC-AS.L.3")
    run_sequence(s, [
        (MeasureVoltage,   {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,   {"subject": "psu_cable_pos"}),
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_l3_repair():
    s = build_system_for("3CC-AS.L.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})

def test_3cc_as_l4():
    s = build_system_for("3CC-AS.L.4")
    run_sequence(s, [
        (MeasureVoltage,      {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,      {"subject": "ctrl_cable_out_pos"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,      {"subject": "ctrl_cable_out_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_l4_repair():
    s = build_system_for("3CC-AS.L.4")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_out_neg"})

def test_3cc_as_l5():
    s = build_system_for("3CC-AS.L.5")
    run_sequence(s, [
        (MeasureVoltage,     {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,     {"subject": "ctrl_cable_out_pos"}),
        (MeasureVoltage,     {"subject": "load_cable_pos"}),
        (InvertEnclosure,    {"subject": "cube_load"}),
        (InspectConnections, {"subject": "load_cable_pos"}),
        (TestContinuity,     {"subject": "main_bulb"}),
        (ReplaceComponent,   {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_l5_repair():
    s = build_system_for("3CC-AS.L.5")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})

# ── Intermittent (I) ─────────────────────────────────────────────────────────

def test_3cc_as_i1():
    s = build_system_for("3CC-AS.I.1")
    run_sequence(s, [
        (InspectConnections, {"subject": "psu_cable_pos"}),
        (ReplaceComponent,   {"subject": "psu_cable_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_i1_repair():
    s = build_system_for("3CC-AS.I.1")
    assert_system_broken(s)
    assert s.test_repair({"psu_cable_pos"})

def test_3cc_as_i2():
    s = build_system_for("3CC-AS.I.2")
    run_sequence(s, [
        (InspectConnections, {"subject": "ctrl_cable_in_pos"}),
        (ReplaceComponent,   {"subject": "ctrl_cable_in_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_i2_repair():
    s = build_system_for("3CC-AS.I.2")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_in_pos"})

def test_3cc_as_i3():
    s = build_system_for("3CC-AS.I.3")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (ReplaceComponent,    {"subject": "ctrl_relay", "part": _RELAY}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_i3_repair():
    s = build_system_for("3CC-AS.I.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_relay"})

def test_3cc_as_i4():
    s = build_system_for("3CC-AS.I.4")
    run_sequence(s, [
        (InspectConnections, {"subject": "load_cable_pos"}),
        (ReplaceComponent,   {"subject": "load_cable_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_i4_repair():
    s = build_system_for("3CC-AS.I.4")
    assert_system_broken(s)
    assert s.test_repair({"load_cable_pos"})

def test_3cc_as_i5():
    s = build_system_for("3CC-AS.I.5")
    run_sequence(s, [
        (InspectConnections, {"subject": "load_cable_neg"}),
        (ReplaceComponent,   {"subject": "load_cable_neg", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_i5_repair():
    s = build_system_for("3CC-AS.I.5")
    assert_system_broken(s)
    assert s.test_repair({"load_cable_neg"})

# ── Unforeseen Interaction (U) ────────────────────────────────────────────────

def test_3cc_as_u1():
    s = build_system_for("3CC-AS.U.1")
    run_sequence(s, [
        (AdjustPotentiometer(0.0), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.25), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.5), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.75), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(1.0), {"subject": "ctrl_sensitivity_pot"}),
        (RotateEnclosure,     {"subject": "cube_ctrl"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_u1_repair():
    s = build_system_for("3CC-AS.U.1")
    assert_system_broken(s)
    assert s.test_repair({"cube_ctrl"})

def test_3cc_as_u2():
    s = build_system_for("3CC-AS.U.2")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,      {"subject": "ctrl_cable_out_neg"}),
        (AdjustPotentiometer(0.0), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.25), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.5), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.75), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(1.0), {"subject": "ctrl_sensitivity_pot"}),
        (RotateEnclosure,     {"subject": "cube_ctrl"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_u2_repair():
    s = build_system_for("3CC-AS.U.2")
    assert_system_broken(s)
    assert not s.test_repair({"ctrl_cable_out_neg"})
    assert not s.test_repair({"cube_ctrl"})
    assert s.test_repair({"ctrl_cable_out_neg", "cube_ctrl"})

def test_3cc_as_u3():
    s = build_system_for("3CC-AS.U.3")
    run_sequence(s, [
        (AdjustPotentiometer(0.0), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.25), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.5), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.75), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(1.0), {"subject": "ctrl_sensitivity_pot"}),
        (RotateEnclosure,     {"subject": "cube_ctrl"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_u3_repair():
    s = build_system_for("3CC-AS.U.3")
    assert_system_broken(s)
    assert s.test_repair({"cube_ctrl"})

def test_3cc_as_u4():
    s = build_system_for("3CC-AS.U.4")
    run_sequence(s, [
        (AdjustPotentiometer(0.0), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.25), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.5), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.75), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(1.0), {"subject": "ctrl_sensitivity_pot"}),
        (RotateEnclosure,     {"subject": "cube_ctrl"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_u4_repair():
    s = build_system_for("3CC-AS.U.4")
    assert_system_broken(s)
    assert s.test_repair({"cube_ctrl"})

def test_3cc_as_u5():
    s = build_system_for("3CC-AS.U.5")
    run_sequence(s, [
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (ReplaceComponent,    {"subject": "ctrl_cable_out_pos", "part": _CABLE}),
        (AdjustPotentiometer(0.0), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.25), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.5), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(0.75), {"subject": "ctrl_sensitivity_pot"}),
        (AdjustPotentiometer(1.0), {"subject": "ctrl_sensitivity_pot"}),
        (RotateEnclosure,     {"subject": "cube_ctrl"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_as_u5_repair():
    s = build_system_for("3CC-AS.U.5")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_out_pos", "cube_ctrl"})

# ===========================================================================
# 3 CUBES CHAIN WITH CURRENT SENSOR  (scenarios 106–135)
# CS uses OpenInspectionPanel(ctrl_panel) for relay access, same as ALS.
# ===========================================================================

# ── Simple (S) ──────────────────────────────────────────────────────────────

def test_3cc_cs_s1():
    s = build_system_for("3CC-CS.S.1")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery", "part": _BATTERY}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_s1_repair():
    s = build_system_for("3CC-CS.S.1")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_cs_s2():
    s = build_system_for("3CC-CS.S.2")
    run_sequence(s, [
        (InvertEnclosure, {"subject": "cube_psu"}),
        (MeasureVoltage,  {"subject": "battery"}),
        (ReverseBattery,  {"subject": "battery"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_s2_repair():
    s = build_system_for("3CC-CS.S.2")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_cs_s3():
    s = build_system_for("3CC-CS.S.3")
    run_sequence(s, [
        (MeasureVoltage,   {"subject": "ctrl_cable_in_pos"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_s3_repair():
    s = build_system_for("3CC-CS.S.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})

def test_3cc_cs_s4():
    s = build_system_for("3CC-CS.S.4")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,      {"subject": "ctrl_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_s4_repair():
    s = build_system_for("3CC-CS.S.4")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_out_pos"})

def test_3cc_cs_s5():
    s = build_system_for("3CC-CS.S.5")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,    {"subject": "cube_load"}),
        (TestContinuity,     {"subject": "main_bulb"}),
        (ReplaceComponent,   {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_s5_repair():
    s = build_system_for("3CC-CS.S.5")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})

# ── Double (D) ───────────────────────────────────────────────────────────────

def test_3cc_cs_d1():
    s = build_system_for("3CC-CS.D.1")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery", "part": _BATTERY}),
        (ObserveComponent, {"subject": "psu_green_led"}),
        (ObserveComponent, {"subject": "ctrl_green_led"}),
        (MeasureVoltage,   {"subject": "ctrl_cable_in_pos"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_d1_repair():
    s = build_system_for("3CC-CS.D.1")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})
    assert s.test_repair({"battery", "ctrl_cable_in_pos", "ctrl_cable_in_neg"})

def test_3cc_cs_d2():
    s = build_system_for("3CC-CS.D.2")
    run_sequence(s, [
        (InvertEnclosure,     {"subject": "cube_psu"}),
        (MeasureVoltage,      {"subject": "battery"}),
        (ReplaceComponent,    {"subject": "battery", "part": _BATTERY}),
        (ObserveComponent,    {"subject": "psu_green_led"}),
        (ObserveComponent,    {"subject": "ctrl_green_led"}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,      {"subject": "ctrl_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_d2_repair():
    s = build_system_for("3CC-CS.D.2")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"ctrl_cable_out_pos"})
    assert s.test_repair({"battery", "ctrl_cable_out_pos"})

def test_3cc_cs_d3():
    s = build_system_for("3CC-CS.D.3")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery", "part": _BATTERY}),
        (ObserveComponent, {"subject": "psu_green_led"}),
        (ObserveComponent, {"subject": "ctrl_green_led"}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,  {"subject": "cube_load"}),
        (TestContinuity,   {"subject": "main_bulb"}),
        (ReplaceComponent, {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_d3_repair():
    s = build_system_for("3CC-CS.D.3")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"main_bulb"})
    assert s.test_repair({"battery", "main_bulb"})

def test_3cc_cs_d4():
    s = build_system_for("3CC-CS.D.4")
    run_sequence(s, [
        (InvertEnclosure,     {"subject": "cube_psu"}),
        (MeasureVoltage,      {"subject": "battery"}),
        (ReverseBattery,      {"subject": "battery"}),
        (ObserveComponent,    {"subject": "psu_green_led"}),
        (ObserveComponent,    {"subject": "ctrl_green_led"}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,      {"subject": "ctrl_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_d4_repair():
    s = build_system_for("3CC-CS.D.4")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    assert not s.test_repair({"ctrl_cable_out_pos"})
    assert s.test_repair({"battery", "ctrl_cable_out_pos"})

def test_3cc_cs_d5():
    s = build_system_for("3CC-CS.D.5")
    run_sequence(s, [
        (MeasureVoltage,      {"subject": "ctrl_cable_in_pos"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
        (ObserveComponent,    {"subject": "psu_green_led"}),
        (ObserveComponent,    {"subject": "ctrl_green_led"}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,      {"subject": "ctrl_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_d5_repair():
    s = build_system_for("3CC-CS.D.5")
    assert_system_broken(s)
    assert not s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})
    assert not s.test_repair({"ctrl_cable_out_pos"})
    assert s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg", "ctrl_cable_out_pos"})

# ── Misleading (M 1–5) ───────────────────────────────────────────────────────

def test_3cc_cs_m1():
    s = build_system_for("3CC-CS.M.1")
    run_sequence(s, [
        (InvertEnclosure, {"subject": "cube_psu"}),
        (MeasureVoltage,  {"subject": "battery"}),
        (ReverseBattery,  {"subject": "battery"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_m1_repair():
    s = build_system_for("3CC-CS.M.1")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_cs_m2():
    s = build_system_for("3CC-CS.M.2")
    run_sequence(s, [
        (InvertEnclosure, {"subject": "cube_psu"}),
        (MeasureVoltage,  {"subject": "battery"}),
        (ReverseBattery,  {"subject": "battery"}),
        (ObserveComponent, {"subject": "psu_green_led"}),
        (ObserveComponent, {"subject": "ctrl_green_led"}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,  {"subject": "cube_load"}),
        (TestContinuity,   {"subject": "main_bulb"}),
        (ReplaceComponent, {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_m2_repair():
    s = build_system_for("3CC-CS.M.2")
    assert_system_broken(s)
    assert not s.test_repair({"battery"})
    # shorted diode bypasses polarity protection, so lamp repair alone restores the circuit
    assert s.test_repair({"main_bulb"})
    assert s.test_repair({"battery", "main_bulb"})

def test_3cc_cs_m3():
    s = build_system_for("3CC-CS.M.3")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (TestContinuity,      {"subject": "ctrl_relay"}),
        (MeasureVoltage,      {"subject": "ctrl_cable_in_pos"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_m3_repair():
    s = build_system_for("3CC-CS.M.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})

def test_3cc_cs_m4():
    s = build_system_for("3CC-CS.M.4")
    run_sequence(s, [
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (TestContinuity,   {"subject": "psu_green_led"}),
        (ReplaceComponent, {"subject": "psu_green_led", "part": _LED}),
        (ObserveComponent, {"subject": "psu_green_led"}),
        (ObserveComponent, {"subject": "ctrl_green_led"}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (InvertEnclosure,  {"subject": "cube_load"}),
        (TestContinuity,   {"subject": "main_bulb"}),
        (ReplaceComponent, {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_m4_repair():
    s = build_system_for("3CC-CS.M.4")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})

def test_3cc_cs_m5():
    # Burned lamp + burned load indicator (both must be replaced)
    s = build_system_for("3CC-CS.M.5")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,    {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (TestContinuity,      {"subject": "ctrl_relay"}),
        (MeasureVoltage,      {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,      {"subject": "load_cable_pos"}),
        (InvertEnclosure,     {"subject": "cube_load"}),
        (TestContinuity,      {"subject": "main_bulb"}),
        (ReplaceComponent,    {"subject": "main_bulb",     "part": _BULB}),
        (ReplaceComponent,    {"subject": "internal_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_m5_repair():
    s = build_system_for("3CC-CS.M.5")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})           # main function restored — sufficient
    assert not s.test_repair({"internal_bulb"})   # indicator only — main lamp still burned
    assert s.test_repair({"main_bulb", "internal_bulb"})

# ── Limited Observability (L) ─────────────────────────────────────────────────

def test_3cc_cs_l1():
    s = build_system_for("3CC-CS.L.1")
    run_sequence(s, [
        (MeasureVoltage,   {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,   {"subject": "psu_cable_pos"}),
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (ReplaceComponent, {"subject": "battery", "part": _BATTERY}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_l1_repair():
    s = build_system_for("3CC-CS.L.1")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_cs_l2():
    s = build_system_for("3CC-CS.L.2")
    run_sequence(s, [
        (MeasureVoltage,  {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,  {"subject": "psu_cable_pos"}),
        (InvertEnclosure, {"subject": "cube_psu"}),
        (MeasureVoltage,  {"subject": "battery"}),
        (ReverseBattery,  {"subject": "battery"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_l2_repair():
    s = build_system_for("3CC-CS.L.2")
    assert_system_broken(s)
    assert s.test_repair({"battery"})

def test_3cc_cs_l3():
    s = build_system_for("3CC-CS.L.3")
    run_sequence(s, [
        (MeasureVoltage,   {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,   {"subject": "psu_cable_pos"}),
        (InvertEnclosure,  {"subject": "cube_psu"}),
        (MeasureVoltage,   {"subject": "battery"}),
        (SwapCablePolarities(port_name="p"), {"cable_a": "ctrl_cable_in_pos", "cable_b": "ctrl_cable_in_neg"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_l3_repair():
    s = build_system_for("3CC-CS.L.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})

def test_3cc_cs_l4():
    s = build_system_for("3CC-CS.L.4")
    run_sequence(s, [
        (MeasureVoltage,      {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,      {"subject": "ctrl_cable_out_pos"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (ReconnectCable,      {"subject": "ctrl_cable_out_pos"}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_l4_repair():
    s = build_system_for("3CC-CS.L.4")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_out_pos"})

def test_3cc_cs_l5():
    s = build_system_for("3CC-CS.L.5")
    run_sequence(s, [
        (MeasureVoltage,     {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,     {"subject": "ctrl_cable_out_pos"}),
        (MeasureVoltage,     {"subject": "load_cable_pos"}),
        (InvertEnclosure,    {"subject": "cube_load"}),
        (InspectConnections, {"subject": "load_cable_pos"}),
        (TestContinuity,     {"subject": "main_bulb"}),
        (ReplaceComponent,   {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_l5_repair():
    s = build_system_for("3CC-CS.L.5")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})

# ── Intermittent (I) ─────────────────────────────────────────────────────────

def test_3cc_cs_i1():
    s = build_system_for("3CC-CS.I.1")
    run_sequence(s, [
        (InspectConnections, {"subject": "psu_cable_pos"}),
        (ReplaceComponent,   {"subject": "psu_cable_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_i1_repair():
    s = build_system_for("3CC-CS.I.1")
    assert_system_broken(s)
    assert s.test_repair({"psu_cable_pos"})

def test_3cc_cs_i2():
    s = build_system_for("3CC-CS.I.2")
    run_sequence(s, [
        (InspectConnections, {"subject": "ctrl_cable_in_pos"}),
        (ReplaceComponent,   {"subject": "ctrl_cable_in_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_i2_repair():
    s = build_system_for("3CC-CS.I.2")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_cable_in_pos"})

def test_3cc_cs_i3():
    s = build_system_for("3CC-CS.I.3")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (ReplaceComponent,    {"subject": "ctrl_relay", "part": _RELAY}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_i3_repair():
    s = build_system_for("3CC-CS.I.3")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_relay"})

def test_3cc_cs_i4():
    s = build_system_for("3CC-CS.I.4")
    run_sequence(s, [
        (InspectConnections, {"subject": "load_cable_pos"}),
        (ReplaceComponent,   {"subject": "load_cable_pos", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_i4_repair():
    s = build_system_for("3CC-CS.I.4")
    assert_system_broken(s)
    assert s.test_repair({"load_cable_pos"})

def test_3cc_cs_i5():
    s = build_system_for("3CC-CS.I.5")
    run_sequence(s, [
        (InspectConnections, {"subject": "load_cable_neg"}),
        (ReplaceComponent,   {"subject": "load_cable_neg", "part": _CABLE}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_i5_repair():
    s = build_system_for("3CC-CS.I.5")
    assert_system_broken(s)
    assert s.test_repair({"load_cable_neg"})

# ── CS-specific Misleading (M.6–M.10) ────────────────────────────────────────

def test_3cc_cs_m6():
    # Relay failed permanently open
    s = build_system_for("3CC-CS.M.6")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,      {"subject": "ctrl_relay"}),
        (ReplaceComponent,    {"subject": "ctrl_relay", "part": _RELAY}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_m6_repair():
    s = build_system_for("3CC-CS.M.6")
    assert_system_broken(s)
    assert s.test_repair({"ctrl_relay"})

def test_3cc_cs_m7():
    # Shorted lamp → relay opens
    s = build_system_for("3CC-CS.M.7")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,      {"subject": "ctrl_relay"}),
        (ReplaceComponent,    {"subject": "ctrl_relay", "part": _RELAY}),
        (InvertEnclosure,     {"subject": "cube_load"}),
        (TestContinuity,      {"subject": "main_bulb"}),
        (ReplaceComponent,    {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_m7_repair():
    s = build_system_for("3CC-CS.M.7")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})

def test_3cc_cs_m8():
    # Shorted internal_bulb indicator → relay opens
    s = build_system_for("3CC-CS.M.8")
    run_sequence(s, [
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,      {"subject": "ctrl_relay"}),
        (ReplaceComponent,    {"subject": "ctrl_relay", "part": _RELAY}),
        (InvertEnclosure,     {"subject": "cube_load"}),
        (TestContinuity,      {"subject": "main_bulb"}),
        (TestContinuity,      {"subject": "internal_bulb"}),
        (ReplaceComponent,    {"subject": "internal_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_m8_repair():
    s = build_system_for("3CC-CS.M.8")
    assert_system_broken(s)
    assert s.test_repair({"internal_bulb"})

def test_3cc_cs_m9():
    # Shorted lamp + open supply LED
    s = build_system_for("3CC-CS.M.9")
    run_sequence(s, [
        (InvertEnclosure,     {"subject": "cube_psu"}),
        (MeasureVoltage,      {"subject": "battery"}),
        (TestContinuity,      {"subject": "psu_green_led"}),
        (ReplaceComponent,    {"subject": "psu_green_led", "part": _LED}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,      {"subject": "ctrl_relay"}),
        (ReplaceComponent,    {"subject": "ctrl_relay", "part": _RELAY}),
        (InvertEnclosure,     {"subject": "cube_load"}),
        (TestContinuity,      {"subject": "main_bulb"}),
        (ReplaceComponent,    {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_m9_repair():
    s = build_system_for("3CC-CS.M.9")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})

def test_3cc_cs_m10():
    # Shorted lamp (no indicators) → relay opens
    s = build_system_for("3CC-CS.M.10")
    run_sequence(s, [
        (MeasureVoltage,      {"subject": "ctrl_cable_in_pos"}),
        (OpenInspectionPanel, {"subject": "load_panel"}),
        (ObserveComponent,   {"subject": "internal_bulb"}),
        (MeasureVoltage,      {"subject": "ctrl_cable_out_pos"}),
        (OpenInspectionPanel, {"subject": "ctrl_panel"}),
        (InspectConnections,  {"subject": "ctrl_cable_out_pos"}),
        (InspectConnections,  {"subject": "ctrl_cable_in_pos"}),
        (MeasureVoltage,      {"subject": "ctrl_relay"}),
        (ReplaceComponent,    {"subject": "ctrl_relay", "part": _RELAY}),
        (InvertEnclosure,     {"subject": "cube_load"}),
        (TestContinuity,      {"subject": "main_bulb"}),
        (ReplaceComponent,    {"subject": "main_bulb", "part": _BULB}),
    ])
    s.simulate(); assert s.is_system_nominal()

def test_3cc_cs_m10_repair():
    s = build_system_for("3CC-CS.M.10")
    assert_system_broken(s)
    assert s.test_repair({"main_bulb"})
