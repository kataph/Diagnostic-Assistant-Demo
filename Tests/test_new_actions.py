"""
Tests for MoveLED and ShortPorts actions.

Run with:
    python -m pytest Tests/test_new_actions.py -s
or:
    python Tests/test_new_actions.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnosable_systems_simulation.systems.ten_cubes.factory import build_ten_cubes_system
from diagnosable_systems_simulation.electrical_simulation.backend.stub import StubBackend
from diagnosable_systems_simulation.actions.diagnostic_actions import MoveLED, ShortPorts
from Implementations.scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(extra_tools=None):
    """Build 10-cubes system with a StubBackend (no ngspice needed)."""
    return build_ten_cubes_system(
        backend=StubBackend(),
        extra_tools=extra_tools or set(),
    )


# ---------------------------------------------------------------------------
# MoveLED tests
# ---------------------------------------------------------------------------

def test_move_led_to_empty_slot():
    """
    Move psu_green_led (with its resistor) from cube_psu to cube_ctrl1
    after the ctrl1 LED+resistor have been physically removed.

    Expected:
    - psu_green_led.enclosure_id == "cube_ctrl1"
    - psu_green_resistor.enclosure_id == "cube_ctrl1"
    - psu_green_led.port("cathode").node_id == node of ctrl1_cable_in_neg.n
    - psu_green_resistor.port("p").node_id == node of ctrl1_switch.n
    """
    sim = _build()

    # Remove ctrl1 LED + resistor to create an empty slot
    sim.remove_component("ctrl1_green_led")
    sim.remove_component("ctrl1_green_resistor")

    psu_led = sim.component("psu_green_led")
    ctrl1_switch   = sim.component("ctrl1_switch")
    ctrl1_cable_in_neg = sim.component("ctrl1_cable_in_neg")

    # Record target slot node IDs before the move
    tgt_pos_node = ctrl1_switch.port("n").node_id
    tgt_neg_node = ctrl1_cable_in_neg.port("n").node_id

    assert tgt_pos_node is not None, "ctrl1_switch.n should be connected"
    assert tgt_neg_node is not None, "ctrl1_cable_in_neg.n should be connected"

    action = MoveLED(target_module_id="cube_ctrl1")
    result = sim.apply_action(action, {"subject": psu_led})

    print(f"\n[move_led empty slot] {result.message}")
    assert result.success, f"MoveLED failed: {result.message}"

    psu_resistor = sim.component("psu_green_resistor")

    # LED and resistor enclosure should be updated
    assert psu_led.enclosure_id == "cube_ctrl1", (
        f"psu_green_led.enclosure_id should be 'cube_ctrl1', got {psu_led.enclosure_id!r}"
    )
    assert psu_resistor.enclosure_id == "cube_ctrl1", (
        f"psu_green_resistor.enclosure_id should be 'cube_ctrl1', got {psu_resistor.enclosure_id!r}"
    )

    # Resistor.p should be on ctrl1's switch.n node
    assert psu_resistor.port("p").node_id == tgt_pos_node, (
        f"psu_green_resistor.p should be at {tgt_pos_node!r}, "
        f"got {psu_resistor.port('p').node_id!r}"
    )

    # LED.cathode should be on ctrl1's cable_in_neg.n node
    assert psu_led.port("cathode").node_id == tgt_neg_node, (
        f"psu_green_led.cathode should be at {tgt_neg_node!r}, "
        f"got {psu_led.port('cathode').node_id!r}"
    )

    print("[check] enclosure_id and port nodes updated correctly ✓")


def test_move_led_swap():
    """
    Move psu_green_led to cube_ctrl1 when ctrl1 already has a LED.
    The two LEDs should be swapped (only the LEDs move, resistors stay).

    Expected after swap:
    - psu_green_led.enclosure_id == "cube_ctrl1"
    - ctrl1_green_led.enclosure_id == "cube_psu"
    - psu_green_led.port("anode").node_id == ctrl1_green_resistor.port("n").node_id
    - ctrl1_green_led.port("anode").node_id == psu_green_resistor.port("n").node_id
    """
    sim = _build()

    psu_led        = sim.component("psu_green_led")
    ctrl1_led      = sim.component("ctrl1_green_led")
    psu_resistor   = sim.component("psu_green_resistor")
    ctrl1_resistor = sim.component("ctrl1_green_resistor")

    # Record the intermediate mid-nodes (resistor.n = led.anode pre-swap)
    psu_mid_node   = psu_resistor.port("n").node_id
    ctrl1_mid_node = ctrl1_resistor.port("n").node_id

    ctrl1_neg_node = sim.component("ctrl1_cable_in_neg").port("n").node_id
    psu_neg_node   = sim.component("battery").port("neg").node_id

    action = MoveLED(target_module_id="cube_ctrl1")
    result = sim.apply_action(action, {"subject": psu_led})

    print(f"\n[move_led swap] {result.message}")
    assert result.success, f"MoveLED (swap) failed: {result.message}"

    # Enclosure IDs swapped
    assert psu_led.enclosure_id == "cube_ctrl1", (
        f"psu_green_led.enclosure_id should be 'cube_ctrl1', got {psu_led.enclosure_id!r}"
    )
    assert ctrl1_led.enclosure_id == "cube_psu", (
        f"ctrl1_green_led.enclosure_id should be 'cube_psu', got {ctrl1_led.enclosure_id!r}"
    )

    # After swap: psu_led is now connected to ctrl1's resistor.n and ctrl1's neg node
    assert psu_led.port("anode").node_id == ctrl1_mid_node, (
        f"psu_led.anode should be at ctrl1's mid-node {ctrl1_mid_node!r}, "
        f"got {psu_led.port('anode').node_id!r}"
    )
    assert psu_led.port("cathode").node_id == ctrl1_neg_node, (
        f"psu_led.cathode should be at ctrl1's neg node {ctrl1_neg_node!r}, "
        f"got {psu_led.port('cathode').node_id!r}"
    )

    # ctrl1_led is now at psu's mid-node and psu's neg node
    assert ctrl1_led.port("anode").node_id == psu_mid_node, (
        f"ctrl1_led.anode should be at psu's mid-node {psu_mid_node!r}, "
        f"got {ctrl1_led.port('anode').node_id!r}"
    )
    assert ctrl1_led.port("cathode").node_id == psu_neg_node, (
        f"ctrl1_led.cathode should be at psu's neg node {psu_neg_node!r}, "
        f"got {ctrl1_led.port('cathode').node_id!r}"
    )

    print("[check] LED swap circuit connections correct ✓")


# ---------------------------------------------------------------------------
# ShortPorts tests
# ---------------------------------------------------------------------------

def test_short_ports_adds_bridge():
    """
    ShortPorts between psu_green_led and ctrl1_green_led should insert a
    0.01 Ω resistor in the circuit graph between their (auto-picked) nodes.
    """
    sim = _build()

    psu_led   = sim.component("psu_green_led")
    ctrl1_led = sim.component("ctrl1_green_led")

    node_a = psu_led.port("anode").node_id
    node_b = ctrl1_led.port("anode").node_id

    action = ShortPorts()
    result = sim.apply_action(action, {"source": psu_led, "sink": ctrl1_led})

    print(f"\n[short_ports] {result.message}")
    assert result.success, f"ShortPorts failed: {result.message}"

    # The inserted bridge should be in the netlist
    short_id = f"_short_{psu_led.component_id}_{ctrl1_led.component_id}"
    assert sim.graph.has_component(short_id), (
        f"Short resistor '{short_id}' not found in circuit graph"
    )

    # Its port nodes should be the two original anode nodes
    nodes = sim.graph.nodes_of(short_id)
    assert set(nodes.values()) == {node_a, node_b}, (
        f"Short resistor connects {set(nodes.values())!r}, expected {{{node_a!r}, {node_b!r}}}"
    )
    print(f"[check] bridge '{short_id}' inserted correctly between {node_a} ↔ {node_b} ✓")


def test_short_ports_duplicate_rejected():
    """
    Attempting to add a second short between the same two components
    (same short_id) must fail gracefully with a clear message.
    """
    sim = _build()

    psu_led   = sim.component("psu_green_led")
    ctrl1_led = sim.component("ctrl1_green_led")

    action = ShortPorts()

    result1 = sim.apply_action(action, {"source": psu_led, "sink": ctrl1_led})
    assert result1.success, f"First ShortPorts failed: {result1.message}"

    result2 = sim.apply_action(action, {"source": psu_led, "sink": ctrl1_led})
    print(f"\n[short_ports duplicate] {result2.message}")
    assert not result2.success, "Second ShortPorts should fail (duplicate short)"
    assert "already exists" in result2.message, (
        f"Expected 'already exists' in message, got: {result2.message!r}"
    )
    print("[check] duplicate short rejected correctly ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_move_led_to_empty_slot()
    test_move_led_swap()
    test_short_ports_adds_bridge()
    test_short_ports_duplicate_rejected()
    print("\nAll tests passed.")