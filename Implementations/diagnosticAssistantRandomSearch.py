"""
DiagnosticAssistantRandomSearch — random baseline diagnostic assistant.

Ignores all observations and LLM reasoning. At setup() it reads the component
list directly from the simulated system (description.simulated_system.all_components()).
At each suggest_action() call it draws the next component uniformly at random
(without replacement) and immediately hypothesises that component as the root cause.
When all components have been hypothesised it returns None (surrender).

This gives a pure random-search lower bound for comparison against LLM and
neuro-symbolic assistants.
"""
from __future__ import annotations

import random
from typing import Optional

from environment_classes import (
    DiagnosticAssistant,
    DiagnosticFaultHypothesis,
    Observation,
    SystemDescription,
)
from configuration import Configuration


class DiagnosticAssistantRandomSearch(DiagnosticAssistant):
    """
    Random baseline: hypothesises one component at a time in random order,
    without replacement, until all components are exhausted.
    """

    def __init__(self, description: SystemDescription, configuration: Configuration) -> None:
        super().__init__(description, configuration)
        self._remaining_components: list[str] = []

    async def setup(self, observations: list[Observation]) -> None:
        sys_desc = self.state.general_system_description
        if sys_desc is not None and sys_desc.simulated_system is not None:
            components = list(sys_desc.simulated_system.all_components().keys())
        else:
            # Fallback: no system available — nothing to hypothesise
            components = []

        random.shuffle(components)
        self._remaining_components = components
        self.logger.info(
            f"RandomSearch assistant initialised with {len(components)} components: "
            f"{components}"
        )

    async def suggest_action(self) -> Optional[DiagnosticFaultHypothesis]:
        if not self._remaining_components:
            self.logger.info("RandomSearch assistant: all components exhausted, surrendering.")
            return None

        component = self._remaining_components.pop(0)
        self.logger.info(f"RandomSearch assistant hypothesises: {component}")
        return DiagnosticFaultHypothesis(
            suspected_components={component},
            explanation=f"Random search: hypothesising {component} as root cause.",
        )
