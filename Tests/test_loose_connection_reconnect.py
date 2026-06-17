"""
Regression test for the B3 fix: reconnect_cable on a loose-connection cable.

Verifies three things:
1. After _add_loose_connection, the cable has RECONNECTABLE affordance.
2. disconnect_cable on an already-floating port does NOT clobber _orig_connections.
3. reconnect_cable succeeds, removes the LooseConnectionCoupling, and the lamp
   lights up reliably on all subsequent simulate() calls (fault is truly gone).
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "diagnosable-systems-simulation"))

from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system
from diagnosable_systems_simulation.electrical_simulation.couplings import (
    _add_loose_connection,
    LooseConnectionCoupling,
)
from diagnosable_systems_simulation.actions.fault_actions import DisconnectCable, ReconnectCable
from diagnosable_systems_simulation.world.affordances import Affordance


def _build_system():
    sys_ = build_three_cubes_system(extra_tools={"multimeter"})
    sys_.simulate()
    sys_._nominal_emitting_light = sys_.last_result.emitting_light
    return sys_


def test_loose_cable_has_reconnectable_affordance():
    """After _add_loose_connection, the cable must advertise RECONNECTABLE."""
    sys_ = _build_system()
    _add_loose_connection(sys_, "load_cable_pos", "p", p=0.5)
    cable = sys_.component("load_cable_pos")
    assert Affordance.RECONNECTABLE in cable.affordances.all_active(cable, sys_.context), (
        "load_cable_pos should have RECONNECTABLE after _add_loose_connection"
    )


def test_disconnect_does_not_clobber_orig_connections():
    """
    If a loose port is already floating and DisconnectCable is called on it,
    _orig_connections must retain the original node ID saved by _add_loose_connection.
    """
    sys_ = _build_system()
    _add_loose_connection(sys_, "load_cable_pos", "p", p=1.0)  # always disconnected
    sys_.simulate()  # ensure port is floating

    cable = sys_.component("load_cable_pos")
    orig_node = cable._orig_connections.get("p")
    assert orig_node is not None, "_add_loose_connection must save original node ID"

    # Now simulate what the agent does: disconnect the already-floating port
    sys_.apply_action(DisconnectCable(port_names=["p"]), {"subject": cable})

    assert cable._orig_connections.get("p") == orig_node, (
        "DisconnectCable must not overwrite _orig_connections['p'] saved by _add_loose_connection"
    )


def test_reconnect_removes_loose_coupling_and_restores_lamp():
    """
    Full B3 scenario: loose fault injected, disconnect attempted on floating port,
    then reconnect_cable succeeds and the lamp is on reliably afterwards.
    """
    sys_ = _build_system()
    nominal_lit = sys_._nominal_emitting_light
    _add_loose_connection(sys_, "load_cable_pos", "p", p=1.0)  # always disconnected
    sys_.simulate()

    cable = sys_.component("load_cable_pos")

    # Step 1: agent tries to disconnect the already-floating port (no-op on the graph)
    sys_.apply_action(DisconnectCable(port_names=["p"]), {"subject": cable})

    # Step 2: agent reconnects — this is what previously failed
    result = sys_.apply_action(ReconnectCable(), {"subject": cable})
    assert result.success, f"ReconnectCable should succeed: {result.message}"

    # Step 3: no LooseConnectionCoupling should remain on load_cable_pos
    remaining = [
        c for c in sys_._runner.couplings
        if isinstance(c, LooseConnectionCoupling) and c.component_id == "load_cable_pos"
    ]
    assert not remaining, "LooseConnectionCoupling must be removed after ReconnectCable"

    # Step 4: lamp must be on reliably on all subsequent simulate() calls
    for i in range(5):
        r = sys_.simulate()
        assert nominal_lit.issubset(r.emitting_light), (
            f"Lamp should be on after reconnect (iteration {i})"
        )