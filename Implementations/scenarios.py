"""
Central repository of diagnostic scenarios.

Each Scenario bundles:
  - a unique integer id
  - the system name ("3_cubes", "10_cubes", "ambient_light_sensor")
  - the RootCauseDescription used by the fixed-scenario and spice-sim saboteurs
  - an optional list of FaultFn callables that apply the fault to a live
    DiagnosableSystem instance (None = simulation not yet implemented)

FaultFn signature:  (DiagnosableSystem) -> None
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from diagnosable_systems_simulation.actions.fault_actions import (
    DegradeComponent, DisconnectCable, ForceSwitch, ReconnectCable, ShortCircuit,
)
from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem, WorldContext

from environment_classes import RootCauseDescription, SymptomDescription, SymptomDescriptions

FaultFn = Callable[[DiagnosableSystem], None]


@dataclass
class Scenario:
    id: int
    system_name: str
    root_cause: RootCauseDescription
    world_context: WorldContext
    fault_fns: Optional[list[FaultFn]] = field(default=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply(sys: DiagnosableSystem, action, targets: dict) -> None:
    """Apply a fault action and raise if the system rejects it."""
    result = sys.apply_action(action, targets)
    if not result.success:
        raise RuntimeError(
            f"Fault injection failed [{action.action_id}]: {result.message}"
        )


# ---------------------------------------------------------------------------
# Fault functions — 3-cubes system
# ---------------------------------------------------------------------------

def _disconnect_ctrl_cable_out_pos(sys: DiagnosableSystem) -> None:
    """Detach the switch-side connector (port 'p') of the ctrl→load positive cable."""
    _apply(sys, DisconnectCable(port_names=["p"]), {"subject": sys.component("ctrl_cable_out_pos")})


def _burn_main_bulb(sys: DiagnosableSystem) -> None:
    """Model burned lamp filament as open-circuit resistance."""
    _apply(sys, DegradeComponent({"resistance": 1e9}), {"subject": sys.component("main_bulb")})


def _deplete_battery(sys: DiagnosableSystem) -> None:
    _apply(sys, DegradeComponent({"voltage": 0.0}), {"subject": sys.component("psu_source")})


def _invert_battery(sys: DiagnosableSystem) -> None:
    """Inverted polarity → negative supply voltage."""
    _apply(sys, DegradeComponent({"voltage": -12.0}), {"subject": sys.component("psu_source")})


def _force_switch_open(sys: DiagnosableSystem) -> None:
    """
    Internal open circuit in the switch.

    The switch contact is broken internally: it conducts no current regardless
    of its mechanical position.  This is modelled by fixing the resistance at
    roff (the switch's own open-circuit value) via a fault overlay — NOT by
    forcing is_closed=False.

    As a result the switch LOOKS closed to an observer when the user closes it,
    but voltage/continuity measurements will reveal it never conducts.  This is
    the physically correct representation of an internal contact failure.
    """
    sw = sys.component("ctrl_switch")
    _apply(sys, DegradeComponent({"resistance": sw.roff}), {"subject": sw})


def _cross_psu_ctrl_cables(sys: DiagnosableSystem) -> None:
    """
    Cross the PSU→ctrl cables by swapping their PSU-side ('p' port) connections.

    Before:  ctrl_cable_in_pos.p → psu_pos (12 V),  ctrl_cable_in_neg.p → gnd (0 V)
    After:   ctrl_cable_in_pos.p → gnd (0 V),        ctrl_cable_in_neg.p → psu_pos (12 V)

    Effect on circuit:
      ctrl_in_p net = 0 V  →  switch output = 0 V  →  lamp off
      ctrl_in_n net = 12 V →  red LED anode = 12 V, cathode via R from 0 V  →  red LED ON
      Green LED: directly across psu_source  →  still ON
    """
    in_pos = sys.component("ctrl_cable_in_pos")
    in_neg = sys.component("ctrl_cable_in_neg")
    # Save PSU-side node IDs before any disconnection
    in_pos_p_node = in_pos.port("p").node_id
    in_neg_p_node = in_neg.port("p").node_id
    # Disconnect only the PSU-side port of each cable
    _apply(sys, DisconnectCable(port_names=["p"]), {"subject": in_pos})
    _apply(sys, DisconnectCable(port_names=["p"]), {"subject": in_neg})
    # Reconnect swapped
    _apply(sys, ReconnectCable({"p": in_neg_p_node}), {"subject": in_pos})
    _apply(sys, ReconnectCable({"p": in_pos_p_node}), {"subject": in_neg})


def _short_psu_output_and_discharge(sys: DiagnosableSystem) -> None:
    """
    Short the PSU output cables together, then mark the battery as discharged.

    The short collapses the psu_pos net to ground; combined with the discharged
    battery, replacing the battery alone will not restore function.
    """
    cable_pos = sys.component("psu_cable_pos")
    cable_neg = sys.component("psu_cable_neg")
    psu_pos_node = cable_pos.port("p").node_id   # psu_pos net
    gnd_node = cable_neg.port("p").node_id       # ground net
    _apply(sys, ShortCircuit(psu_pos_node, gnd_node, "psu_output_short"), {})
    _apply(sys, DegradeComponent({"voltage": 0.0}), {"subject": sys.component("psu_source")})


# ---------------------------------------------------------------------------
# Fault functions — 10-cubes system
# ---------------------------------------------------------------------------

def _disconnect_ctrl3_cable_in_pos(sys: DiagnosableSystem) -> None:
    """Detach ctrl3's positive input cable, breaking the circuit from ctrl3 onwards."""
    _apply(sys, DisconnectCable(), {"subject": sys.component("ctrl3_cable_in_pos")})


def _disconnect_ctrl6_cable_in_pos(sys: DiagnosableSystem) -> None:
    """Detach ctrl6's positive input cable, breaking the circuit from ctrl6 onwards."""
    _apply(sys, DisconnectCable(), {"subject": sys.component("ctrl6_cable_in_pos")})


def _remove_all_ctrl_green_leds(sys: DiagnosableSystem) -> None:
    """Simulate removal of all 8 control-module green LEDs by open-circuiting them."""
    for i in range(1, 9):
        _apply(
            sys,
            DegradeComponent({"forward_voltage": 1e9}),
            {"subject": sys.component(f"ctrl{i}_green_led")},
        )


# ---------------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    Scenario(
        id=1, system_name="3_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="One of the cables connected to the switch has been detached",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The lamp does not turn on when the switch is operated"),
                SymptomDescription("The green led on top of the power supply module is on"),
                SymptomDescription("The red led on top of the control module is off"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=[_disconnect_ctrl_cable_out_pos],
    ),
    Scenario(
        id=2, system_name="3_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="Burned lamp filaments",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The lamp does not turn on when the switch is operated"),
                SymptomDescription("The green led on top of the power supply module is on"),
                SymptomDescription("The red led on top of the control module is off"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=[_burn_main_bulb],
    ),
    Scenario(
        id=3, system_name="3_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="Battery is depleted",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The lamp does not turn on when the switch is operated"),
                SymptomDescription("The green led on top of the power supply module is off"),
                SymptomDescription("The red led on top of the control module is off"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=[_deplete_battery],
    ),
    Scenario(
        id=4, system_name="3_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="Battery has been installed incorrectly (it has been installed with inverted polarity)",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The lamp does not turn on when the switch is operated"),
                SymptomDescription("The green led on top of the power supply module is off"),
                SymptomDescription("The red led on top of the control module is on"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=[_invert_battery],
    ),
    Scenario(
        id=5, system_name="3_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="The cables between the control module and the load module (or between the power supply module and the control module) are crossed, resulting in reverse voltage being supplied to the load",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The lamp does not turn on when the switch is operated"),
                SymptomDescription("The green led on top of the power supply module is on"),
                SymptomDescription("The red led on top of the control module is on"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=[_cross_psu_ctrl_cables],
    ),
    Scenario(
        id=6, system_name="3_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="Internal open circuit in the switch: the switch is always open.",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The lamp does not turn on when the switch is operated"),
                SymptomDescription("The green led on top of the power supply module is on"),
                SymptomDescription("The red led on top of the control module is off"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=[_force_switch_open],
    ),
    Scenario(
        id=7, system_name="10_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="Battery exhausted.",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The led on the power supply module is off"),
                SymptomDescription("All the leds on the control modules are off"),
                SymptomDescription("The lamp does not turn on when the battery is inserted in the circuit and all the switches are in the on position"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=[_deplete_battery],
    ),
    Scenario(
        id=8, system_name="10_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="The switch in the control module 3 is detached from one of the corresponding cables.",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The led on the power supply module is on"),
                SymptomDescription("The first two control module leds are on. Starting from the third onwards they are off"),
                SymptomDescription("The lamp does not turn on when a battery is inserted in the circuit and all the switches are in the on position"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=[_disconnect_ctrl3_cable_in_pos],
    ),
    Scenario(
        id=9, system_name="10_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="The switch in the control module 3 is detached from one of the corresponding cables. Also, all the control module leds have been removed.",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The led on the power supply module is on"),
                SymptomDescription("All the leds on the control modules are missing"),
                SymptomDescription("The lamp does not turn on when a battery is inserted in the circuit and all the switches are in the on position"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=[_disconnect_ctrl3_cable_in_pos, _remove_all_ctrl_green_leds],
    ),
    Scenario(
        id=10, system_name="10_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="The switch in the control module 6 is detached from one of the corresponding cables. Also, all the control module leds have been removed.",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The led on the power supply module is on"),
                SymptomDescription("All the leds on the control modules are missing"),
                SymptomDescription("The lamp does not turn on when a battery is inserted in the circuit and all the switches are in the on position"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=[_disconnect_ctrl6_cable_in_pos, _remove_all_ctrl_green_leds],
    ),
    Scenario(
        id=11, system_name="10_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="The switch in the control module 6 is detached from one of the corresponding cables. Also, all the control module leds have been removed. Also, the service agent does not have a multimeter or other tools at its disposal to take electric measurements.",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The led on the power supply module is on"),
                SymptomDescription("All the leds on the control modules are missing"),
                SymptomDescription("The lamp does not turn on when a battery is inserted in the circuit and all the switches are in the on position"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={}),
        fault_fns=[_disconnect_ctrl6_cable_in_pos, _remove_all_ctrl_green_leds],
    ),
    Scenario(
        id=12, system_name="3_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="Detached cable from the switch and, at the same time and independently, The cables between the control module and the load module (or between the power supply module and the control module) are crossed, resulting in reverse voltage being supplied to the load",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The lamp does not turn on when the switch is operated"),
                SymptomDescription("The green led on top of the power supply module is on"),
                SymptomDescription("The red led on top of the control module is on"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=[_cross_psu_ctrl_cables, _disconnect_ctrl_cable_out_pos],
    ),
    Scenario(
        id=13, system_name="10_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="The switch in the control module 3 is detached from one of the corresponding cables. Also, at the same time and independently, battery is exhausted.",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The led on the power supply module is off"),
                SymptomDescription("All the leds on the control modules are off"),
                SymptomDescription("The lamp does not turn on when a battery is inserted in the circuit and all the switches are in the on position"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=[_disconnect_ctrl3_cable_in_pos, _deplete_battery],
    ),
    Scenario(
        id=14, system_name="10_cubes",
        root_cause=RootCauseDescription(
            root_cause_description_proper="The cables that come out from the power supply module are accidently shorted. This discharged the battery violently and now the battery does not supply power anymore. Replacing the battery will not solve the issue.",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The led on the power supply module is off"),
                SymptomDescription("All the leds on the control modules are off"),
                SymptomDescription("The lamp does not turn on when a battery is inserted in the circuit and all the switches are in the on position"),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=[_short_psu_output_and_discharge],
    ),
    Scenario(
        id=15, system_name="ambient_light_sensor",
        root_cause=RootCauseDescription(
            root_cause_description_proper="The lamp turns off about every 20 seconds because the sensor is incorrectly positioned and receives part of the light of the lamp: a sufficient quantity to make the sensor turn off the lamp. This happens about every 20 seconds due to the ambient light sensor inner workings. After the light turns off the sensor records a below-threshold ambient light and turns on the lamp almost immediately. This keeps occurring.",
            symptoms_descriptions=SymptomDescriptions([
                SymptomDescription("The led on the power supply module is on"),
                SymptomDescription("The lamp turns off about every 20 seconds for about half a second. Then it turns on again. This keeps happening."),
            ])
        ),
        world_context=WorldContext(tools_in_hand={'multimeter'}),
        fault_fns=None,
    ),
]
