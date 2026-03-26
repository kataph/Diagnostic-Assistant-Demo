from diagnosable_systems_simulation.actions import ActionCost

SIM_SECONDS_PER_KG_UNIT = 1.0

def sim_cost_to_kg_units(action_cost: ActionCost) -> float:
    return action_cost.time / SIM_SECONDS_PER_KG_UNIT