"""
DiagnosticAssistantRandomTrajectory — mock diagnostic assistant for protocol testing.

At setup() time, looks up the system name from Configuration.SYSTEM_NAME and
randomly selects one of five fixed diagnostic action sequences for that system.
During the scenario, suggest_action() pops from the selected sequence until
exhausted, then returns None (triggering timeout/surrender).

No LLM calls are made. The fixed sequences use component IDs that are valid
in each system, and action.description is left None so execute_action() will
call action.get_name() = "type -> target", which the mock NL interface parses
correctly.
"""
from __future__ import annotations

import random
from typing import Optional

from environment_classes import (
    DiagnosticAction, DiagnosticAssistant, DiagnosticFaultHypothesis,
    HypothesisVerificationResult, Observation, SystemDescription,
)
from configuration import Configuration

_DA = lambda t, tgt: DiagnosticAction(type=t, target=tgt)

# Five fixed trajectories per system. Component IDs must exist in the
# corresponding system's all_components() dict.
FIXED_TRAJECTORIES: dict[str, list[list[DiagnosticAction]]] = {
    "3_cubes": [
        [_DA("Observe", "battery"),      _DA("Test",    "ctrl_switch"),       _DA("Replace", "main_bulb")],
        [_DA("Observe", "main_bulb"),    _DA("Test",    "psu_cable_pos"),     _DA("Replace", "battery")],
        [_DA("Observe", "psu_green_led"),_DA("Test",    "ctrl_cable_in_pos"), _DA("Observe", "internal_bulb")],
        [_DA("Test",    "battery"),      _DA("Observe", "ctrl_switch"),       _DA("Replace", "ctrl_switch")],
        [_DA("Observe", "psu_green_led"),_DA("Observe", "ctrl_red_led"),      _DA("Test",    "battery")],
    ],
    "ambient_light_sensor": [
        [_DA("Observe", "battery"),      _DA("Test",    "ctrl_relay"),        _DA("Replace", "main_bulb")],
        [_DA("Observe", "main_bulb"),    _DA("Test",    "psu_cable_pos"),     _DA("Replace", "battery")],
        [_DA("Observe", "psu_green_led"),_DA("Test",    "ctrl_cable_in_pos"), _DA("Observe", "internal_bulb")],
        [_DA("Test",    "battery"),      _DA("Observe", "ctrl_relay"),        _DA("Replace", "ctrl_relay")],
        [_DA("Observe", "psu_green_led"),_DA("Test",    "ctrl_relay"),        _DA("Test",    "battery"), _DA("Test",    "battery"), _DA("Test",    "battery")],
    ],
    "current_sensor": [
        [_DA("Observe", "battery"),      _DA("Test",    "ctrl_relay"),        _DA("Replace", "main_bulb")],
        [_DA("Observe", "main_bulb"),    _DA("Test",    "psu_cable_pos"),     _DA("Replace", "battery")],
        [_DA("Observe", "psu_green_led"),_DA("Test",    "ctrl_cable_in_pos"), _DA("Observe", "internal_bulb")],
        [_DA("Test",    "battery"),      _DA("Observe", "ctrl_relay"),        _DA("Replace", "ctrl_relay")],
        [_DA("Observe", "psu_green_led"),_DA("Test",    "ctrl_relay"),        _DA("Test",    "battery")],
    ],
    "10_cubes": [
        [_DA("Observe", "battery"),       _DA("Test",    "ctrl3_switch"),       _DA("Replace", "main_bulb")],
        [_DA("Observe", "main_bulb"),     _DA("Test",    "ctrl6_switch"),       _DA("Replace", "battery")],
        [_DA("Observe", "psu_green_led"), _DA("Test",    "ctrl3_cable_in_pos"), _DA("Observe", "internal_bulb")],
        [_DA("Test",    "battery"),       _DA("Observe", "ctrl8_switch"),       _DA("Replace", "ctrl6_switch")],
        [_DA("Observe", "psu_green_led"), _DA("Observe", "ctrl1_green_led"),    _DA("Test",    "battery")],
    ],
    "asymmetric_chains": [
        [_DA("Observe", "psu1_battery"),       _DA("Test",    "ctrl1_switch"),        _DA("Replace", "load1_main_bulb")],
        [_DA("Observe", "load1_main_bulb"),    _DA("Test",    "ctrl1_cable_out_pos"), _DA("Replace", "psu1_battery")],
        [_DA("Observe", "psu1_psu_green_led"), _DA("Test",    "ctrl3_cable_in_pos"),  _DA("Observe", "load2_main_bulb")],
        [_DA("Test",    "psu2_battery"),       _DA("Observe", "ctrl3_switch"),        _DA("Replace", "ctrl1_switch")],
        [_DA("Observe", "psu2_psu_green_led"), _DA("Observe", "ctrl1_green_led"),     _DA("Test",    "psu1_battery")],
    ],
}

# Fallback for unknown systems: generic observation + test + replace
_FALLBACK_TRAJECTORIES: list[list[DiagnosticAction]] = [
    [_DA("Observe", "battery"), _DA("Test", "ctrl_switch"), _DA("Replace", "main_bulb")],
    [_DA("Observe", "main_bulb"), _DA("Test", "battery"), _DA("Replace", "battery")],
]


class DiagnosticAssistantRandomTrajectory(DiagnosticAssistant):
    """
    Mock diagnostic assistant that executes a randomly chosen fixed action
    sequence. Used for protocol testing without any LLM calls.
    """

    def __init__(self, description: SystemDescription, configuration: Configuration) -> None:
        super().__init__(description, configuration)
        self._planned_actions: list[DiagnosticAction] = []

    async def setup(self, observations: list[Observation]) -> None:
        system_name = self.configuration.SYSTEM_NAME
        # SYSTEM_NAME from CLI is e.g. "3CubesSystem" — map to system_name key
        name_map = {
            "3CubesSystem":             "3_cubes",
            "10CubesSystem":            "10_cubes",
            "AmbientLightSensorSystem": "ambient_light_sensor",
            "AsymmetricChainsSystem":   "asymmetric_chains",
            "CurrentSensorSystem":      "current_sensor",
        }
        key = name_map.get(system_name, system_name)
        trajectories = FIXED_TRAJECTORIES.get(key, _FALLBACK_TRAJECTORIES)
        chosen = random.choice(trajectories)
        self._planned_actions = list(chosen)  # copy so we can pop
        self.logger.info(
            f"RandomTrajectory assistant selected trajectory: "
            f"{[a.get_name() for a in self._planned_actions]}"
        )

    async def suggest_action(self) -> Optional[DiagnosticAction | DiagnosticFaultHypothesis]:
        if not self._planned_actions:
            self.logger.info("RandomTrajectory assistant: planned actions exhausted, returning None.")
            return None
        action = self._planned_actions.pop(0)
        self.logger.info(f"RandomTrajectory assistant suggests: {action.get_name()}")
        return action
