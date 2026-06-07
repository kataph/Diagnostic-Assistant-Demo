"""
Fault injection registry for all 135 diagnostic scenarios.

Structure
---------
FAULT_FNS_FOR_INJECTIONS maps each scenario number (1–135) to a 4-tuple:
    (scenario_id, fault_fns, system_config_fn, world_context)

where
  scenario_id    -- semantic id from SCENARIOS_MASTER.csv (for redundancy check)
  fault_fns      -- list of callables (DiagnosableSystem) -> None, or None if
                    the scenario has no simulation support
  system_config_fn -- single callable applied BEFORE fault_fns to modify system
                    configuration (e.g. remove all LED indicators); None if no alteration
  world_context  -- WorldContext (tools_in_hand) for this scenario

load_scenarios_from_csv() reads SCENARIOS_MASTER.csv and constructs the full
list of Scenario objects consumed by saboteur implementations.
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Optional

from diagnosable_systems_simulation.actions.fault_actions import (
    DegradeComponent, DisconnectCable, ForceSwitch, ReconnectCable, ShortCircuit,
)
from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph
from diagnosable_systems_simulation.electrical_simulation.results import SimulationResult
from diagnosable_systems_simulation.electrical_simulation.solver import PhysicalCoupling, SimulationRunner
from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem
from diagnosable_systems_simulation.world.context import WorldContext

from environment_classes import (
    FaultFn, RootCauseDescription, Scenario, SymptomDescription, SymptomDescriptions,
)

_CSV_PATH = Path(__file__).resolve().parent.parent / "SCENARIOS_MASTER.csv"

_CSV_SYSTEM_MAP = {
    "3 cubes chain":                      "3_cubes",
    "10 cubes chain":                     "10_cubes",
    "Doubly connected asymmetric chains": "asymmetric_chains",
    "3 cubes chain with ambient sensor":  "ambient_light_sensor",
    "3 cubes chain with current sensor":  "current_sensor",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _apply(sys: DiagnosableSystem, action, targets: dict) -> None:
    result = sys.apply_action(action, targets)
    if not result.success:
        raise RuntimeError(
            f"Fault injection failed [{action.action_id}]: {result.message}"
        )


# ---------------------------------------------------------------------------
# LooseConnectionCoupling
# ---------------------------------------------------------------------------

class LooseConnectionCoupling(PhysicalCoupling):
    """
    Models an intermittent open circuit on a single cable port.

    On each simulation step, the port is randomly disconnected with
    probability *p* (default 0.5). When disconnected the port is
    immediately reconnected before the next step so the simulation
    can converge with either an open or closed connection.

    The coupling also marks the system context so that
    ServiceAgentSpiceSim.decide_finish() does not declare a false
    success when the connection happens to be closed during a check.
    """

    def __init__(self, component_id: str, port_name: str, p: float = 0.5) -> None:
        self.component_id = component_id
        self.port_name = port_name
        self.p = p
        self._currently_disconnected = False
        self._saved_node: Optional[int] = None

    def apply(self, result: SimulationResult, graph: CircuitGraph, context: WorldContext) -> bool:
        context.extra["has_loose_connection"] = True

        if not graph.has_component(self.component_id):
            return False

        comp = graph.get_component(self.component_id)
        port = next((p for p in comp.ports if p.name == self.port_name), None)
        if port is None:
            return False

        if self._currently_disconnected:
            # Reconnect
            if self._saved_node is not None:
                from diagnosable_systems_simulation.electrical_simulation.circuit import CircuitGraph as _CG
                graph.reconnect_port(self.component_id, self.port_name, self._saved_node)
            self._currently_disconnected = False
            self._saved_node = None
            return True
        else:
            # Randomly disconnect
            if random.random() < self.p and port.is_connected():
                self._saved_node = port.node_id
                graph.disconnect_port(self.component_id, self.port_name)
                self._currently_disconnected = True
                return True
            return False


def _add_loose_connection(sys: DiagnosableSystem, component_id: str, port_name: str, p: float = 0.5) -> None:
    """Attach a LooseConnectionCoupling and flag the context."""
    coupling = LooseConnectionCoupling(component_id, port_name, p=p)
    sys._runner.couplings.append(coupling)
    sys.context.extra["has_loose_connection"] = True


# ---------------------------------------------------------------------------
# Generic primitives
# ---------------------------------------------------------------------------

def _deplete_battery(sys: DiagnosableSystem) -> None:
    _apply(sys, DegradeComponent({"voltage": 0.0}), {"subject": sys.component("battery")})


def _invert_battery(sys: DiagnosableSystem) -> None:
    _apply(sys, DegradeComponent({"voltage": -12.0}), {"subject": sys.component("battery")})


def _burn_main_bulb(sys: DiagnosableSystem) -> None:
    _apply(sys, DegradeComponent({"resistance": 1e9}), {"subject": sys.component("main_bulb")})


def _cross_psu_ctrl_cables(sys: DiagnosableSystem) -> None:
    in_pos = sys.component("ctrl_cable_in_pos")
    in_neg = sys.component("ctrl_cable_in_neg")
    in_pos_p_node = in_pos.port("p").node_id
    in_neg_p_node = in_neg.port("p").node_id
    _apply(sys, DisconnectCable(port_names=["p"]), {"subject": in_pos})
    _apply(sys, DisconnectCable(port_names=["p"]), {"subject": in_neg})
    _apply(sys, ReconnectCable({"p": in_neg_p_node}), {"subject": in_pos})
    _apply(sys, ReconnectCable({"p": in_pos_p_node}), {"subject": in_neg})


def _disconnect_ctrl_cable_out_pos(sys: DiagnosableSystem) -> None:
    _apply(sys, DisconnectCable(port_names=["p"]), {"subject": sys.component("ctrl_cable_out_pos")})


def _force_switch_open(sys: DiagnosableSystem) -> None:
    sw = sys.component("ctrl_switch")
    _apply(sys, DegradeComponent({"resistance": sw.roff}), {"subject": sw})


def open_component(component_id: str) -> FaultFn:
    """Return a fault function that sets the named component to open-circuit resistance.
    No-op if the component does not exist in the system (e.g. already absent)."""
    def _fn(sys: DiagnosableSystem) -> None:
        if component_id not in sys.all_components():
            return
        _apply(sys, DegradeComponent({"resistance": 1e9}), {"subject": sys.component(component_id)})
    _fn.__name__ = f"open_{component_id}"
    return _fn


def short_component(component_id: str) -> FaultFn:
    """Return a fault function that short-circuits the named component."""
    def _fn(sys: DiagnosableSystem) -> None:
        _apply(sys, DegradeComponent({"resistance": 0.01}), {"subject": sys.component(component_id)})
    _fn.__name__ = f"short_{component_id}"
    return _fn


def loose_connection(component_id: str, port_name: str, p: float = 0.5) -> FaultFn:
    """Return a fault function that makes a port intermittently open.
    No-op if the component does not exist."""
    def _fn(sys: DiagnosableSystem) -> None:
        if component_id not in sys.all_components():
            return
        _add_loose_connection(sys, component_id, port_name, p=p)
    _fn.__name__ = f"loose_{component_id}_{port_name}"
    return _fn


# ---------------------------------------------------------------------------
# 3_cubes specific
# ---------------------------------------------------------------------------

def _short_psu_output_and_discharge(sys: DiagnosableSystem) -> None:
    cable_pos = sys.component("psu_cable_pos")
    cable_neg = sys.component("psu_cable_neg")
    psu_pos_node = cable_pos.port("p").node_id
    gnd_node = cable_neg.port("p").node_id
    _apply(sys, ShortCircuit(psu_pos_node, gnd_node, "psu_output_short"), {"start": cable_pos, "end": cable_neg})
    _apply(sys, DegradeComponent({"voltage": 0.0}), {"subject": sys.component("battery")})


def _3cubes_remove_all_indicators(sys: DiagnosableSystem) -> None:
    for cid in ["psu_green_led", "psu_green_resistor", "ctrl_red_led", "ctrl_red_resistor"]:
        if cid in sys.all_components():
            sys.remove_component(cid)


# ---------------------------------------------------------------------------
# 10_cubes specific
# ---------------------------------------------------------------------------

def _disconnect_ctrl3_cable_in_pos(sys: DiagnosableSystem) -> None:
    _apply(sys, DisconnectCable(port_names=["n"]), {"subject": sys.component("ctrl3_cable_in_pos")})


def _disconnect_ctrl6_cable_in_pos(sys: DiagnosableSystem) -> None:
    _apply(sys, DisconnectCable(port_names=["n"]), {"subject": sys.component("ctrl6_cable_in_pos")})


def _disconnect_ctrl8_cable_in_pos(sys: DiagnosableSystem) -> None:
    _apply(sys, DisconnectCable(port_names=["n"]), {"subject": sys.component("ctrl8_cable_in_pos")})


def _10cubes_remove_ctrl_leds(sys: DiagnosableSystem) -> None:
    for i in range(1, 9):
        for cid in [f"ctrl{i}_green_led", f"ctrl{i}_green_resistor"]:
            if cid in sys.all_components():
                sys.remove_component(cid)


# 10CC.M.1: reversed battery + supply LED appears on (reversed LED polarity)
def _10cubes_reverse_psu_led(sys: DiagnosableSystem) -> None:
    """Swap anode/cathode node IDs of psu_green_led so it lights when battery polarity is reversed."""
    led = sys.component("psu_green_led")
    anode = led.port("anode")
    cathode = led.port("cathode")
    anode.node_id, cathode.node_id = cathode.node_id, anode.node_id


# ---------------------------------------------------------------------------
# asymmetric_chains (DCAC) specific
# ---------------------------------------------------------------------------

def _ac_deplete_psu1(sys: DiagnosableSystem) -> None:
    _apply(sys, DegradeComponent({"voltage": 0.0}), {"subject": sys.component("psu1_battery")})


def _ac_deplete_psu2(sys: DiagnosableSystem) -> None:
    _apply(sys, DegradeComponent({"voltage": 0.0}), {"subject": sys.component("psu2_battery")})


def _ac_disconnect_ctrl1_cable_out_pos(sys: DiagnosableSystem) -> None:
    _apply(sys, DisconnectCable(port_names=["n"]), {"subject": sys.component("ctrl1_cable_out_pos")})


def _ac_disconnect_ctrl2_cable_out_pos(sys: DiagnosableSystem) -> None:
    _apply(sys, DisconnectCable(port_names=["n"]), {"subject": sys.component("ctrl2_cable_out_pos")})


def _ac_disconnect_ctrl3_cable_in_pos(sys: DiagnosableSystem) -> None:
    _apply(sys, DisconnectCable(port_names=["n"]), {"subject": sys.component("ctrl3_cable_in_pos")})


def _ac_burn_load1_main_bulb(sys: DiagnosableSystem) -> None:
    _apply(sys, DegradeComponent({"resistance": 1e9}), {"subject": sys.component("load1_main_bulb")})


def _ac_burn_load2_main_bulb(sys: DiagnosableSystem) -> None:
    _apply(sys, DegradeComponent({"resistance": 1e9}), {"subject": sys.component("load2_main_bulb")})


def _dcac_remove_all_indicators(sys: DiagnosableSystem) -> None:
    for prefix in ["psu1_psu", "psu2_psu", "ctrl1", "ctrl2", "ctrl3"]:
        for suffix in ["green_led", "green_resistor"]:
            cid = f"{prefix}_{suffix}"
            if cid in sys.all_components():
                sys.remove_component(cid)


# ---------------------------------------------------------------------------
# ambient_light_sensor (ALS) specific
# ---------------------------------------------------------------------------

def _stack_modules(sys: DiagnosableSystem) -> None:
    sys.context.extra["als_feedback"] = True


def _als_disconnect_relay_cable(sys: DiagnosableSystem) -> None:
    """Force the ALS relay permanently open, breaking the 0V return path."""
    _apply(sys, ForceSwitch(is_closed=False), {"subject": sys.component("ctrl_relay")})


# ALS crossed cables: same as 3-cubes
_als_cross_psu_ctrl_cables = _cross_psu_ctrl_cables


def _als_remove_all_indicators(sys: DiagnosableSystem) -> None:
    for cid in ["psu_green_led", "psu_green_resistor"]:
        if cid in sys.all_components():
            sys.remove_component(cid)
    # ALS has no ctrl_red_led — nothing else to remove for indicators


# ---------------------------------------------------------------------------
# current_sensor (CS) specific
# ---------------------------------------------------------------------------

def _cs_disconnect_relay_cable(sys: DiagnosableSystem) -> None:
    """Force the current-sensor relay permanently open, breaking the 0V return path."""
    _apply(sys, ForceSwitch(is_closed=False), {"subject": sys.component("ctrl_relay")})


def _cs_force_relay_open(sys: DiagnosableSystem) -> None:
    _apply(sys, ForceSwitch(is_closed=False), {"subject": sys.component("ctrl_relay")})


def _cs_remove_all_indicators(sys: DiagnosableSystem) -> None:
    for cid in ["psu_green_led", "psu_green_resistor", "ctrl_green_led", "ctrl_green_resistor"]:
        if cid in sys.all_components():
            sys.remove_component(cid)


# CS crossed cables: same underlying function
_cs_cross_psu_ctrl_cables = _cross_psu_ctrl_cables


# ---------------------------------------------------------------------------
# Pre-built lambda-like fault functions (closures over open_component etc.)
# ---------------------------------------------------------------------------

# 3CC specific
_open_ctrl_red_led      = open_component("ctrl_red_led")
_open_psu_green_led     = open_component("psu_green_led")
_open_internal_bulb     = open_component("internal_bulb")
_open_load_diode        = open_component("load_diode")

# 10CC specific
_open_ctrl2_green_led   = open_component("ctrl2_green_led")
_open_ctrl5_green_led   = open_component("ctrl5_green_led")
_open_ctrl7_green_led   = open_component("ctrl7_green_led")

# DCAC specific
_open_psu1_green_led    = open_component("psu1_psu_green_led")
_open_psu2_green_led    = open_component("psu2_psu_green_led")
_open_load1_diode       = open_component("load1_load_diode")
_open_load1_internal    = open_component("load1_internal_bulb")
_open_ctrl3_green_led   = open_component("ctrl3_green_led")

# CS specific
_short_main_bulb        = short_component("main_bulb")
_short_internal_bulb    = short_component("internal_bulb")

# Loose connections — 3CC (cable component_id, port that can go loose)
_loose_3cc_battery_cable        = loose_connection("psu_cable_pos", "p")
_loose_3cc_psu_ctrl             = loose_connection("ctrl_cable_in_pos", "p")
_loose_3cc_switch_cable         = loose_connection("ctrl_cable_out_pos", "p")
_loose_3cc_ctrl_load            = loose_connection("load_cable_pos", "p")
_loose_3cc_lamp_cable           = loose_connection("load_cable_neg", "n")

# Loose connections — 10CC
_loose_10cc_battery_cable       = loose_connection("psu_cable_pos", "p")
_loose_10cc_switch3_cable       = loose_connection("ctrl3_cable_in_pos", "n")
_loose_10cc_switch6_cable       = loose_connection("ctrl6_cable_in_pos", "n")
_loose_10cc_switch8_cable       = loose_connection("ctrl8_cable_in_pos", "n")
_loose_10cc_lamp_cable          = loose_connection("load_cable_neg", "n")

# Loose connections — DCAC
_loose_dcac_battery1_cable      = loose_connection("psu1_psu_cable_pos", "p")
_loose_dcac_battery2_cable      = loose_connection("psu2_psu_cable_pos", "p")
_loose_dcac_switch1_cable       = loose_connection("ctrl1_cable_out_pos", "n")
_loose_dcac_switch3_cable       = loose_connection("ctrl3_cable_in_pos", "n")
_loose_dcac_lamp1_cable         = loose_connection("load1_load_cable_neg", "n")
_loose_dcac_lamp2_cable         = loose_connection("load2_load_cable_neg", "n")

# Loose connections — ALS
_loose_als_battery_cable        = loose_connection("psu_cable_pos", "p")
_loose_als_psu_ctrl             = loose_connection("ctrl_cable_in_pos", "p")
_loose_als_relay_cable          = loose_connection("ctrl_relay", "n")
_loose_als_ctrl_load            = loose_connection("load_cable_pos", "p")
_loose_als_lamp_cable           = loose_connection("load_cable_neg", "n")
_loose_als_switch_cable         = loose_connection("ctrl_cable_out_pos", "p")

# Loose connections — CS (same cable IDs as ALS minus ctrl_switch)
_loose_cs_battery_cable         = loose_connection("psu_cable_pos", "p")
_loose_cs_psu_ctrl              = loose_connection("ctrl_cable_in_pos", "p")
_loose_cs_relay_cable           = loose_connection("ctrl_relay", "n")
_loose_cs_ctrl_load             = loose_connection("load_cable_pos", "p")
_loose_cs_lamp_cable            = loose_connection("load_cable_neg", "n")

# Multimeter available by default for all scenarios
_MT = WorldContext(tools_in_hand={"multimeter"})
_NO_TOOLS = WorldContext(tools_in_hand=set())

# ---------------------------------------------------------------------------
# FAULT_FNS_FOR_INJECTIONS
# Maps scenario_number -> (scenario_id, fault_fns, system_config_fn, world_context)
# ---------------------------------------------------------------------------

FAULT_FNS_FOR_INJECTIONS: dict[int, tuple[
    str,
    Optional[list[FaultFn]],
    Optional[FaultFn],
    WorldContext,
]] = {
    # ── 3 cubes chain ────────────────────────────────────────────────────
    1:  ("3CC.S.1",  [_deplete_battery],                                    None,                       _MT),
    2:  ("3CC.S.2",  [_invert_battery],                                     None,                       _MT),
    3:  ("3CC.S.3",  [_cross_psu_ctrl_cables],                              None,                       _MT),
    4:  ("3CC.S.4",  [_disconnect_ctrl_cable_out_pos],                      None,                       _MT),
    5:  ("3CC.S.5",  [_burn_main_bulb],                                     None,                       _MT),
    6:  ("3CC.D.1",  [_deplete_battery, _cross_psu_ctrl_cables],            None,                       _MT),
    7:  ("3CC.D.2",  [_deplete_battery, _disconnect_ctrl_cable_out_pos],    None,                       _MT),
    8:  ("3CC.D.3",  [_deplete_battery, _burn_main_bulb],                   None,                       _MT),
    9:  ("3CC.D.4",  [_invert_battery, _disconnect_ctrl_cable_out_pos],     None,                       _MT),
    10: ("3CC.D.5",  [_cross_psu_ctrl_cables, _disconnect_ctrl_cable_out_pos], None,                   _MT),
    11: ("3CC.M.1",  [_invert_battery, _open_ctrl_red_led],                 None,                       _MT),
    12: ("3CC.M.2",  [_invert_battery, _open_load_diode, _burn_main_bulb],  None,                       _MT),
    13: ("3CC.M.3",  [_cross_psu_ctrl_cables, _open_ctrl_red_led],         None,                       _MT),
    14: ("3CC.M.4",  [_burn_main_bulb, _open_psu_green_led],               None,                       _MT),
    15: ("3CC.M.5",  [_burn_main_bulb, _open_internal_bulb],               None,                       _MT),
    16: ("3CC.L.1",  [_deplete_battery],                                    _3cubes_remove_all_indicators, _MT),
    17: ("3CC.L.2",  [_invert_battery],                                     _3cubes_remove_all_indicators, _MT),
    18: ("3CC.L.3",  [_cross_psu_ctrl_cables],                             _3cubes_remove_all_indicators, _MT),
    19: ("3CC.L.4",  [_disconnect_ctrl_cable_out_pos],                     _3cubes_remove_all_indicators, _MT),
    20: ("3CC.L.5",  [_burn_main_bulb],                                    _3cubes_remove_all_indicators, _MT),
    21: ("3CC.I.1",  [_loose_3cc_battery_cable],                            None,                       _MT),
    22: ("3CC.I.2",  [_loose_3cc_psu_ctrl],                                 None,                       _MT),
    23: ("3CC.I.3",  [_loose_3cc_switch_cable],                             None,                       _MT),
    24: ("3CC.I.4",  [_loose_3cc_ctrl_load],                                None,                       _MT),
    25: ("3CC.I.5",  [_loose_3cc_lamp_cable],                               None,                       _MT),
    # ── 10 cubes chain ───────────────────────────────────────────────────
    26: ("10CC.S.1", [_deplete_battery],                                    None,                       _MT),
    27: ("10CC.S.2", [_disconnect_ctrl3_cable_in_pos],                      None,                       _MT),
    28: ("10CC.S.3", [_disconnect_ctrl6_cable_in_pos],                      None,                       _MT),
    29: ("10CC.S.4", [_disconnect_ctrl8_cable_in_pos],                      None,                       _MT),
    30: ("10CC.S.5", [_burn_main_bulb],                                     None,                       _MT),
    31: ("10CC.D.1", [_deplete_battery, _disconnect_ctrl3_cable_in_pos],   None,                       _MT),
    32: ("10CC.D.2", [_deplete_battery, _disconnect_ctrl8_cable_in_pos],   None,                       _MT),
    33: ("10CC.D.3", [_disconnect_ctrl3_cable_in_pos, _disconnect_ctrl8_cable_in_pos], None,            _MT),
    34: ("10CC.D.4", [_burn_main_bulb, _disconnect_ctrl3_cable_in_pos],    None,                       _MT),
    35: ("10CC.D.5", [_burn_main_bulb, _disconnect_ctrl8_cable_in_pos],    None,                       _MT),
    36: ("10CC.M.1", [_invert_battery, _10cubes_reverse_psu_led],          None,                       _MT),
    37: ("10CC.M.2", [_disconnect_ctrl3_cable_in_pos, _open_ctrl2_green_led], None,                    _MT),
    38: ("10CC.M.3", [_disconnect_ctrl6_cable_in_pos, _open_ctrl5_green_led], None,                    _MT),
    39: ("10CC.M.4", [_disconnect_ctrl8_cable_in_pos, _open_ctrl7_green_led], None,                    _MT),
    40: ("10CC.M.5", [_burn_main_bulb, _open_internal_bulb],               None,                       _MT),
    41: ("10CC.L.1", [_deplete_battery],                                   _10cubes_remove_ctrl_leds,   _MT),
    42: ("10CC.L.2", [_disconnect_ctrl3_cable_in_pos],                     _10cubes_remove_ctrl_leds,   _MT),
    43: ("10CC.L.3", [_disconnect_ctrl6_cable_in_pos],                     _10cubes_remove_ctrl_leds,   _MT),
    44: ("10CC.L.4", [_disconnect_ctrl8_cable_in_pos],                     _10cubes_remove_ctrl_leds,   _MT),
    45: ("10CC.L.5", [_burn_main_bulb],                                    _10cubes_remove_ctrl_leds,   _MT),
    46: ("10CC.I.1", [_loose_10cc_battery_cable],                           None,                       _MT),
    47: ("10CC.I.2", [_loose_10cc_switch3_cable],                           None,                       _MT),
    48: ("10CC.I.3", [_loose_10cc_switch6_cable],                           None,                       _MT),
    49: ("10CC.I.4", [_loose_10cc_switch8_cable],                           None,                       _MT),
    50: ("10CC.I.5", [_loose_10cc_lamp_cable],                              None,                       _MT),
    # ── Doubly connected asymmetric chains (DCAC) ─────────────────────────
    51: ("DCAC.S.1", [_ac_disconnect_ctrl1_cable_out_pos],                  None,                       _MT),
    52: ("DCAC.S.2", [_ac_disconnect_ctrl3_cable_in_pos],                   None,                       _MT),
    53: ("DCAC.S.3", [_open_load1_diode],                                   None,                       _MT),
    54: ("DCAC.S.4", [_ac_burn_load1_main_bulb],                            None,                       _MT),
    55: ("DCAC.S.5", [_ac_burn_load2_main_bulb],                            None,                       _MT),
    56: ("DCAC.D.1", [_ac_deplete_psu1, _ac_deplete_psu2],                 None,                       _MT),
    57: ("DCAC.D.2", [_ac_deplete_psu2, _ac_disconnect_ctrl1_cable_out_pos], None,                     _MT),
    58: ("DCAC.D.3", [_ac_disconnect_ctrl1_cable_out_pos, _ac_disconnect_ctrl2_cable_out_pos], None,   _MT),
    59: ("DCAC.D.4", [_ac_disconnect_ctrl3_cable_in_pos, _ac_burn_load1_main_bulb], None,              _MT),
    60: ("DCAC.D.5", [_ac_burn_load1_main_bulb, _ac_burn_load2_main_bulb], None,                       _MT),
    61: ("DCAC.M.1", [_ac_disconnect_ctrl1_cable_out_pos, _open_psu1_green_led], None,                 _MT),
    62: ("DCAC.M.2", [_ac_disconnect_ctrl1_cable_out_pos, _ac_disconnect_ctrl2_cable_out_pos, _open_psu2_green_led], None, _MT),
    63: ("DCAC.M.3", [_ac_burn_load1_main_bulb, _open_load1_internal],     None,                       _MT),
    64: ("DCAC.M.4", [_ac_burn_load2_main_bulb, _open_psu2_green_led],     None,                       _MT),
    65: ("DCAC.M.5", [_ac_burn_load2_main_bulb, _open_ctrl3_green_led],    None,                       _MT),
    66: ("DCAC.L.1", [_ac_disconnect_ctrl1_cable_out_pos],                  _dcac_remove_all_indicators, _MT),
    67: ("DCAC.L.2", [_ac_disconnect_ctrl3_cable_in_pos],                   _dcac_remove_all_indicators, _MT),
    68: ("DCAC.L.3", [_open_load1_diode],                                   _dcac_remove_all_indicators, _MT),
    69: ("DCAC.L.4", [_ac_burn_load1_main_bulb],                            _dcac_remove_all_indicators, _MT),
    70: ("DCAC.L.5", [_ac_burn_load2_main_bulb],                            _dcac_remove_all_indicators, _MT),
    71: ("DCAC.I.1", [_loose_dcac_battery1_cable, _loose_dcac_battery2_cable], None,                   _MT),
    72: ("DCAC.I.2", [_loose_dcac_switch1_cable],                           None,                       _MT),
    73: ("DCAC.I.3", [_loose_dcac_switch1_cable, _loose_dcac_switch3_cable], None,                     _MT),
    74: ("DCAC.I.4", [_loose_dcac_lamp1_cable],                             None,                       _MT),
    75: ("DCAC.I.5", [_loose_dcac_lamp2_cable],                             None,                       _MT),
    # ── 3 cubes chain with ambient sensor ────────────────────────────────
    76: ("3CC-AS.S.1", [_deplete_battery],                                  None,                       _MT),
    77: ("3CC-AS.S.2", [_invert_battery],                                   None,                       _MT),
    78: ("3CC-AS.S.3", [_als_cross_psu_ctrl_cables],                        None,                       _MT),
    79: ("3CC-AS.S.4", [_als_disconnect_relay_cable],                       None,                       _MT),
    80: ("3CC-AS.S.5", [_burn_main_bulb],                                   None,                       _MT),
    81: ("3CC-AS.D.1", [_deplete_battery, _als_cross_psu_ctrl_cables],      None,                       _MT),
    82: ("3CC-AS.D.2", [_deplete_battery, _als_disconnect_relay_cable],     None,                       _MT),
    83: ("3CC-AS.D.3", [_deplete_battery, _burn_main_bulb],                 None,                       _MT),
    84: ("3CC-AS.D.4", [_invert_battery, _als_disconnect_relay_cable],      None,                       _MT),
    85: ("3CC-AS.D.5", [_als_cross_psu_ctrl_cables, _als_disconnect_relay_cable], None,                 _MT),
    86: ("3CC-AS.M.1", [_invert_battery, _open_ctrl_red_led],               None,                       _MT),
    87: ("3CC-AS.M.2", [_invert_battery, _open_load_diode, _burn_main_bulb], None,                     _MT),
    88: ("3CC-AS.M.3", [_als_cross_psu_ctrl_cables, _open_ctrl_red_led],   None,                       _MT),
    89: ("3CC-AS.M.4", [_burn_main_bulb, _open_psu_green_led],             None,                       _MT),
    90: ("3CC-AS.M.5", [_burn_main_bulb, _open_internal_bulb],             None,                       _MT),
    91: ("3CC-AS.L.1", [_deplete_battery],                                  _als_remove_all_indicators,  _MT),
    92: ("3CC-AS.L.2", [_invert_battery],                                   _als_remove_all_indicators,  _MT),
    93: ("3CC-AS.L.3", [_als_cross_psu_ctrl_cables],                        _als_remove_all_indicators,  _MT),
    94: ("3CC-AS.L.4", [_als_disconnect_relay_cable],                       _als_remove_all_indicators,  _MT),
    95: ("3CC-AS.L.5", [_burn_main_bulb],                                   _als_remove_all_indicators,  _MT),
    96: ("3CC-AS.I.1", [_loose_als_battery_cable],                          None,                       _MT),
    97: ("3CC-AS.I.2", [_loose_als_psu_ctrl],                               None,                       _MT),
    98: ("3CC-AS.I.3", [_loose_als_relay_cable],                            None,                       _MT),
    99: ("3CC-AS.I.4", [_loose_als_ctrl_load],                              None,                       _MT),
    100: ("3CC-AS.I.5", [_loose_als_lamp_cable],                            None,                       _MT),
    101: ("3CC-AS.U.1", [_stack_modules],                                   None,                       _MT),
    102: ("3CC-AS.U.2", [_stack_modules, _als_disconnect_relay_cable],      None,                       _MT),
    103: ("3CC-AS.U.3", [_stack_modules, _open_psu_green_led],              None,                       _MT),
    104: ("3CC-AS.U.4", [_stack_modules],                                   _als_remove_all_indicators,  _MT),
    105: ("3CC-AS.U.5", [_stack_modules, _loose_als_switch_cable],          None,                       _MT),
    # ── 3 cubes chain with current sensor ────────────────────────────────
    106: ("3CC-CS.S.1", [_deplete_battery],                                 None,                       _MT),
    107: ("3CC-CS.S.2", [_invert_battery],                                  None,                       _MT),
    108: ("3CC-CS.S.3", [_cs_cross_psu_ctrl_cables],                        None,                       _MT),
    109: ("3CC-CS.S.4", [_cs_disconnect_relay_cable],                       None,                       _MT),
    110: ("3CC-CS.S.5", [_burn_main_bulb],                                  None,                       _MT),
    111: ("3CC-CS.D.1", [_deplete_battery, _cs_cross_psu_ctrl_cables],      None,                       _MT),
    112: ("3CC-CS.D.2", [_deplete_battery, _cs_disconnect_relay_cable],     None,                       _MT),
    113: ("3CC-CS.D.3", [_deplete_battery, _burn_main_bulb],                None,                       _MT),
    114: ("3CC-CS.D.4", [_invert_battery, _cs_disconnect_relay_cable],      None,                       _MT),
    115: ("3CC-CS.D.5", [_cs_cross_psu_ctrl_cables, _cs_disconnect_relay_cable], None,                  _MT),
    116: ("3CC-CS.M.1", [_invert_battery, _open_ctrl_red_led],              None,                       _MT),
    117: ("3CC-CS.M.2", [_invert_battery, _open_load_diode, _burn_main_bulb], None,                    _MT),
    118: ("3CC-CS.M.3", [_cs_cross_psu_ctrl_cables, _open_ctrl_red_led],   None,                       _MT),
    119: ("3CC-CS.M.4", [_burn_main_bulb, _open_psu_green_led],            None,                       _MT),
    120: ("3CC-CS.M.5", [_burn_main_bulb, _open_internal_bulb],            None,                       _MT),
    121: ("3CC-CS.L.1", [_deplete_battery],                                 _cs_remove_all_indicators,   _MT),
    122: ("3CC-CS.L.2", [_invert_battery],                                  _cs_remove_all_indicators,   _MT),
    123: ("3CC-CS.L.3", [_cs_cross_psu_ctrl_cables],                        _cs_remove_all_indicators,   _MT),
    124: ("3CC-CS.L.4", [_cs_disconnect_relay_cable],                       _cs_remove_all_indicators,   _MT),
    125: ("3CC-CS.L.5", [_burn_main_bulb],                                  _cs_remove_all_indicators,   _MT),
    126: ("3CC-CS.I.1", [_loose_cs_battery_cable],                          None,                       _MT),
    127: ("3CC-CS.I.2", [_loose_cs_psu_ctrl],                               None,                       _MT),
    128: ("3CC-CS.I.3", [_loose_cs_relay_cable],                            None,                       _MT),
    129: ("3CC-CS.I.4", [_loose_cs_ctrl_load],                              None,                       _MT),
    130: ("3CC-CS.I.5", [_loose_cs_lamp_cable],                             None,                       _MT),
    131: ("3CC-CS.M.6",  [_cs_force_relay_open],                            None,                       _MT),
    132: ("3CC-CS.M.7",  [_short_main_bulb],                                None,                       _MT),
    133: ("3CC-CS.M.8",  [_short_internal_bulb],                            None,                       _MT),
    134: ("3CC-CS.M.9",  [_short_main_bulb, _open_psu_green_led],           None,                       _MT),
    135: ("3CC-CS.M.10", [_short_main_bulb],                                _cs_remove_all_indicators,   _MT),
}


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _parse_observations(obs_str: str) -> SymptomDescriptions:
    """Split semicolon-separated observation string into SymptomDescriptions."""
    parts = [p.strip() for p in obs_str.split(";") if p.strip()]
    return SymptomDescriptions([SymptomDescription(p) for p in parts])


def load_scenarios_from_csv(csv_path: Path = _CSV_PATH) -> list[Scenario]:
    scenarios: list[Scenario] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row or not row[0].strip():
                break
            num = int(row[0].strip())
            if num > 135:
                break
            scenario_id  = row[1].strip()
            system_csv   = row[2].strip()
            system_config_str = row[3].strip()
            fault_label  = row[6].strip()
            description  = row[7].strip()
            obs_str      = row[8].strip()

            system_name = _CSV_SYSTEM_MAP.get(system_csv, system_csv)

            entry = FAULT_FNS_FOR_INJECTIONS.get(num)
            if entry is None:
                raise ValueError(f"No FAULT_FNS_FOR_INJECTIONS entry for scenario {num}")

            reg_id, fault_fns, system_config_fn, world_ctx = entry
            assert reg_id == scenario_id, (
                f"Scenario {num}: id mismatch — CSV={scenario_id!r}, registry={reg_id!r}"
            )

            root_cause = RootCauseDescription(
                root_cause_description_proper=fault_label,
                symptoms_descriptions=_parse_observations(obs_str),
                notes=description or None,
            )

            scenarios.append(Scenario(
                number=num,
                scenario_id=scenario_id,
                system_name=system_name,
                system_config=system_config_str,
                root_cause=root_cause,
                fault_fns=fault_fns,
                world_context=world_ctx,
                system_config_fn=system_config_fn,
            ))
    return scenarios


SCENARIOS: list[Scenario] = load_scenarios_from_csv()
