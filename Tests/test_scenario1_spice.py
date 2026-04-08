"""
Manual smoke test: scenario 1 on the three-cubes system (SPICE backend).

Steps
-----
1. Build the three-cubes system.
2. Attach a stdout logger to the DiagnosableSystem runner.
3. Inject scenario-1 fault (scenario 1: disconnect port 'p' of ctrl_cable_out_pos).
4. Open the control switch.
5. Observe the main lightbulb and print the result.

Run with:
    python -m pytest tests/test_scenario1_spice.py -s
or simply:
    python tests/test_scenario1_spice.py
"""
import logging
import sys
import os

# Allow imports from the ESWC_2026_Demo root (scenarios.py lives there)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system
from diagnosable_systems_simulation.actions.diagnostic_actions import ObserveComponent, OpenSwitch
from Implementations.scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# Minimal stdout logger
# ---------------------------------------------------------------------------

def _make_stdout_logger(name: str = "SpiceRunner") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_scenario1_open_switch_observe_bulb():
    """
Executing this test will, among other things, print 3 spice netlists. 
The first correspond to the nominal 3 cubes system, the second to the 
system sabotaged by disconnecting the negative switch wire, the third
to the sabotaged system with the switch opened. 
This is the first netlist: 

.title diagnosable_system
Vbattery net_0 0 12.0
Dpsu_green_led net_1 _led_mid_psu_green_led LED_psu_green_led
R_rled_psu_green_led _led_mid_psu_green_led 0 1.0
Rpsu_cable_neg 0 net_3 0.01
Rpsu_green_resistor net_0 net_1 1000.0
Rpsu_cable_pos net_0 net_2 0.01
Rctrl_cable_in_pos net_2 net_4 0.01
Rctrl_cable_in_neg net_3 net_7 0.01
Rctrl_switch net_4 net_5 1e-06
Rctrl_red_resistor net_4 net_6 1000.0
Rctrl_cable_out_pos net_5 net_8 0.01
Dctrl_red_led net_7 _led_mid_ctrl_red_led LED_ctrl_red_led
R_rled_ctrl_red_led _led_mid_ctrl_red_led net_6 1.0
Rctrl_cable_out_neg net_7 net_9 0.01
Rload_cable_pos net_8 net_10 0.01
Rload_cable_neg net_9 net_11 0.01
Dload_diode net_10 net_12 D_load_diode
Rmain_bulb net_12 net_11 120.0
Rinternal_bulb net_12 net_11 500.0
.model LED_psu_green_led D (IS=1e-14 N=1.8 VJ=2.1)
.model LED_ctrl_red_led D (IS=1e-14 N=1.8 VJ=2.0)
.model D_load_diode D (IS=1e-14 N=1.0 VJ=0.7)


This is the third one: 

.title diagnosable_system
Vbattery net_0 0 12.0
Dpsu_green_led net_1 _led_mid_psu_green_led LED_psu_green_led
R_rled_psu_green_led _led_mid_psu_green_led 0 1.0
Rpsu_cable_neg 0 net_3 0.01
Rpsu_green_resistor net_0 net_1 1000.0
Rpsu_cable_pos net_0 net_2 0.01
Rctrl_cable_in_pos net_2 net_4 0.01
Rctrl_cable_in_neg net_3 net_7 0.01
Rctrl_switch net_4 net_5 1000000000.0
Rctrl_red_resistor net_4 net_6 1000.0
Dctrl_red_led net_7 _led_mid_ctrl_red_led LED_ctrl_red_led
R_rled_ctrl_red_led _led_mid_ctrl_red_led net_6 1.0
Rctrl_cable_out_neg net_7 net_9 0.01
Rload_cable_pos net_8 net_10 0.01
Rload_cable_neg net_9 net_11 0.01
Dload_diode net_10 net_12 D_load_diode
Rmain_bulb net_12 net_11 120.0
Rinternal_bulb net_12 net_11 500.0
.model LED_psu_green_led D (IS=1e-14 N=1.8 VJ=2.1)
.model LED_ctrl_red_led D (IS=1e-14 N=1.8 VJ=2.0)
.model D_load_diode D (IS=1e-14 N=1.0 VJ=0.7)


The differences are: 
-the resistance value of the switch: 1GOhm is an approximation for an open
-the deletion of the negative switch wire (it must be deleted as spice does not allow for floating nets)

Another thing to note:
-wires are modelled as 10mOhm resistors. Important: for some reason modelling
the with 1uOhm resistance is troublesome (LTspice instead is able to work with the 1uOhm value)"""

    def _observe_bulb(sim) -> None:
        obs_result = sim.apply_action(
        ObserveComponent(),
        {"subject": sim.component("main_bulb")},
        )
        print(f"[observe bulb]    {obs_result}")
        assert obs_result.success, f"ObserveComponent failed: {obs_result.message}"

        print("\n=== observation record ===")
        if obs_result.observation is not None:
            for prop in obs_result.observation.properties:
                unit = f" {prop.unit}" if prop.unit else ""
                print(f"  {prop.name}: {prop.value}{unit}")
        else:
            print("  (no observation record)")
    # 1. Build the system 
    sim = build_three_cubes_system(extra_tools={"multimeter"})
    
    # 2. Attach a stdout logger so SPICE netlist details appear on the console
    sim.add_logger(_make_stdout_logger())
    
    # 3. Observe the bulb
    _observe_bulb(sim)

    # 4. Inject scenario-1 fault via the canonical fault functions and observe the bulb again
    scenario = next(s for s in SCENARIOS if s.id == 1)
    for fault_fn in scenario.fault_fns:
        fault_fn(sim)
    print(f"\n[fault injection] scenario {scenario.id}: {scenario.root_cause.root_cause_description_proper}")
    _observe_bulb(sim)

    # 4. Open the control switch
    open_result = sim.apply_action(
        OpenSwitch(),
        {"subject": sim.component("ctrl_switch")},
    )
    print(f"[open switch]     {open_result}")
    assert open_result.success, f"OpenSwitch failed: {open_result.message}"

    # 5. Observe the main lightbulb again
    _observe_bulb(sim)
    


if __name__ == "__main__":
    test_scenario1_open_switch_observe_bulb()

    