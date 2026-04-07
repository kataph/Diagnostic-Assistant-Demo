from enum import Enum


class PROMPTS(Enum):
    Saboteur = "You are an expert engineer whose duty is to study the documentation available about an engineered system and to prouce a fault cause model. The fault cause model TODO"
    FTGenerator = """You are an expert engineer whose duty is to study the documentation available about an engineered system and to prouce a fault tree. Fault trees are representations of the various parallel and sequential combinations of faults that can result in the occurrence of a predefined undesired event (top event). The fault tree is constructed by identifying the top event and then determining all the possible causes that could lead to that event, breaking them down into more specific sub-events until reaching basic events that cannot be further subdivided. In your case, the top event is the generic failure of the engineered system. You have to output a json file where the (a) all the events of the fault tree are listed (each with a name and description property) and (b) all the AND and OR gates are in a list containing objects such as 
    {
        "gate": "AND",
        "input_arguments": ["event1", "event2"],
        "output_argument": "event3"
     }. 
    Ensure that the fault tree is comprehensive, logically structured, and accurately reflects the potential failure modes of the system based on the provided documentation. Use your own knowledge and/or web searches about engineering systems to supplement the information from the documentation where necessary."""
    # Remember for each component explicitly the cognitive level needed: top is LLM or human, for RJ
    AnomalousNominalComponentExtractor_agent = """You are given an engineered system description and a list of component identifiers. The system description is a description of certain aspect of the system current behavior. The component identifiers (built using URL addresses) list all the components of the system. 
    
    You must do the following: identify all the components that are mentioned in the current system behavior. Then, for each of those, use the function 'retrieve_component_context_tool' to understand if the current behavior of the component is suggesting that something is anomalous in the system (either in the component itself or somewhere else) or not. If yes, put the component into a 'components_suggesting_anomaly_presence' list, if not put the component into a 'components_suggesting_nominal_behavior' list. If no relevant information about a component can be derived from the description then this component should not be included in either list, so that the union of the two lists may be smaller than the whole list of components. Additionally, the two lists should not have common elements. Please, copy the component identifiers as they are provided.
    
    You MUST use the tool retrieve_component_context_tool. Do not answer if you dont use the tool. 
    
    Current system behavior description:
    {symptom}
    
    Component list:
    {components}
    """
    AnomalousNominalComponentExtractor_agent_v2 = """You will be given a description of an engineered system's current observable behavior and a list of component identifiers. Each component is shown as "LocalName (full_URI)".

Your goal is to classify components into two lists:
- components_suggesting_anomaly_presence: the behavior description suggests this component (or something it interacts with) is NOT functioning normally.
- components_suggesting_nominal_behavior: the behavior description explicitly indicates this component IS functioning normally.
Components for which the description provides no information must be excluded from both lists.

Follow these steps exactly for each component mentioned or strongly implied by the description:
1. Call retrieve_component_context_tool with a natural-language description of the component to understand its normal function and role.
2. Based on the behavior description AND the retrieved context, decide: anomalous, nominal, or unknown.
3. If anomalous or nominal, add the component's full_URI (not the LocalName) to the appropriate output list.

Classification rules:
- anomalous: the description suggests the component is malfunctioning, missing, or that a fault exists at or near it.
- nominal: the description explicitly states the component is working correctly.
- When uncertain, do NOT include the component in either list.
- The two lists must be disjoint.

You MUST call retrieve_component_context_tool for every component you classify. Do not classify any component without first calling the tool.
    """

    AnomalousNominalComponentExtractor_agent_v2_input = """
    Current system behavior description:
    {symptom}

    Component list (format: LocalName (full_URI)):
    {components}
    """
    AnomalousNominalComponentExtractor_call_v2 = """You are given an engineered system description and a list of component identifiers. The system description is compoed of two parts: a general description of the system and also a description of its current behavior. The component identifiers (built using URL addresses) list all the components of the system. 
    
    You must do the following: identify all the components that are mentioned in the current system behavior. Then, for each of those, use the supplied system description to understand if the current behavior of the component is suggestsing that something is anomalous in the system or not (either in the component itself or somewhere else) or not. If yes, put the component into a 'components_suggesting_anomaly_presence' list, if not put the component into a 'components_suggesting_nominal_behavior' list. If no relevant information about a component can be derived from the description then this component should not be included in either list, so that the union of the two lists may be smaller than the whole list of components. Additionally, the two lists should not have common elements. Please, copy the component identifiers as they are provided.
    
    General system description:
    {system_description}
    
    Current system behavior description:
    {symptom}
    
    Component list:
    {components}
    """
    AnomalousNominalComponentExtractor_call = """You are given an engineered system description and a list of component identifiers. The system description is compoed of two parts: a general description of the system and also a description of its current behavior. The component identifiers (built using URL addresses) list all the components of the system. Your task is to output two lists: one containing the subset of components that the current system behavior suggests are behaving in a way that suggests there is something anomalous in the system (either in the component itself or somewhere else), another one for the subset of components that are mentioned in the system description but there is no suggestion of anomalous behavior. If no information about a component can be derived from the description then this component should not be included in either list, so that the union of the lists may be smaller than the whole list of components. Additionally, the two lists should not have common elements. Please, copy the component identifiers as they are provided.
    
    General system description:
    {system_description}
    
    Current system behavior description:
    {symptom}
    
    Component list:
    {components}
    """
