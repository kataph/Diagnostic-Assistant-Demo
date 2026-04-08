"""
Tests for TestControlSubchain action.

Scenarios tested:
  1. All 8 modules healthy → lamp ON for any sub-range
  2. Faulty module (ctrl3 disconnected) inside the range → lamp OFF
  3. Faulty module outside the range → lamp ON (fault bypassed)
  4. Wiring is restored after the test (no lasting side-effects)
  5. Single-module subchain (start == end)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "diagnosable-systems-simulation"))

import pytest

from diagnosable_systems_simulation.actions.diagnostic_actions import TestControlSubchain
from diagnosable_systems_simulation.actions.fault_actions import DisconnectCable
from diagnosable_systems_simulation.electrical_simulation.backend.spice import PySpiceBackend
from diagnosable_systems_simulation.systems.ten_cubes.factory import build_ten_cubes_system


@pytest.fixture
def healthy_system():
    return build_ten_cubes_system(extra_tools={"multimeter"})


def _apply(system, action, targets):
    result = system.apply_action(action, targets)
    assert result.success, f"Action failed: {result.message}"
    return result


def _run_subchain(system, start_cube, end_cube):
    action = TestControlSubchain(start_module_id=start_cube, end_module_id=end_cube)
    result = system.apply_action(action, {})
    assert result.success, f"TestControlSubchain failed: {result.message}"
    return result.observation.properties


def _lamp_on(props) -> bool:
    for p in props:
        if p.name == "lamp_on":
            return p.value
    raise AssertionError("lamp_on property not found in observation")


def _node_snapshot(system):
    """Return {component_id: {port_name: node_id}} for all components."""
    snap = {}
    for cid, comp in system.all_components().items():
        snap[cid] = {p.name: p.node_id for p in comp.ports}
    return snap


class TestHealthySystem:
    def test_full_chain_lamp_on(self, healthy_system):
        props = _run_subchain(healthy_system, "cube_ctrl1", "cube_ctrl8")
        assert _lamp_on(props) is True

    def test_first_half_lamp_on(self, healthy_system):
        props = _run_subchain(healthy_system, "cube_ctrl1", "cube_ctrl4")
        assert _lamp_on(props) is True

    def test_second_half_lamp_on(self, healthy_system):
        props = _run_subchain(healthy_system, "cube_ctrl5", "cube_ctrl8")
        assert _lamp_on(props) is True

    def test_single_module_lamp_on(self, healthy_system):
        props = _run_subchain(healthy_system, "cube_ctrl3", "cube_ctrl3")
        assert _lamp_on(props) is True


class TestFaultyModuleInRange:
    def test_fault_inside_range_lamp_off(self, healthy_system):
        # Disconnect ctrl3's input positive cable (n port inside cube)
        cable = healthy_system.component("ctrl3_cable_in_pos")
        _apply(healthy_system, DisconnectCable(port_names=["n"]), {"subject": cable})

        # Subchain includes ctrl3 → lamp should be OFF
        props = _run_subchain(healthy_system, "cube_ctrl1", "cube_ctrl5")
        assert _lamp_on(props) is False

    def test_fault_outside_range_lamp_on(self, healthy_system):
        # Disconnect ctrl6's input positive cable
        cable = healthy_system.component("ctrl6_cable_in_pos")
        _apply(healthy_system, DisconnectCable(port_names=["n"]), {"subject": cable})

        # Subchain ctrl1–ctrl5 does NOT include ctrl6 → lamp should be ON
        props = _run_subchain(healthy_system, "cube_ctrl1", "cube_ctrl5")
        assert _lamp_on(props) is True


class TestNoSideEffects:
    def test_wiring_restored_after_test(self, healthy_system):
        before = _node_snapshot(healthy_system)
        _run_subchain(healthy_system, "cube_ctrl3", "cube_ctrl6")
        after = _node_snapshot(healthy_system)
        assert before == after, "Node topology changed after TestControlSubchain"

    def test_simulation_result_restored(self, healthy_system):
        # Simulate normally first
        result_before = healthy_system.simulate()
        lit_before = frozenset(result_before.emitting_light)

        _run_subchain(healthy_system, "cube_ctrl2", "cube_ctrl5")

        result_after = healthy_system.simulate()
        lit_after = frozenset(result_after.emitting_light)
        assert lit_before == lit_after, "Emitting-light set changed after subchain test"

    def test_wiring_restored_after_faulty_subchain(self, healthy_system):
        cable = healthy_system.component("ctrl4_cable_in_pos")
        _apply(healthy_system, DisconnectCable(port_names=["n"]), {"subject": cable})

        before = _node_snapshot(healthy_system)
        _run_subchain(healthy_system, "cube_ctrl3", "cube_ctrl6")
        after = _node_snapshot(healthy_system)
        assert before == after
