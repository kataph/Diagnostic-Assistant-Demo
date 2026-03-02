import math

from functools import partial, update_wrapper
from logging import Logger, root
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Any, Optional, Literal as LiteralType
from agents import Agent, function_tool 
from rdflib import Graph, URIRef, Literal as LiteralRDF
from rdflib.plugins.sparql.results.csvresults import *
from rdflib.namespace import split_uri
from pprint import pprint as pp


from Utilities.formatting import terminal_uri_parts_gpt, to_one_line
from configuration import Configuration
from environment_classes import AssistantState, DiagnosticAction, DiagnosticActionResult, DiagnosticAssistant, DiagnosticPlan, Observation, empty_sys_descr, diagnosticActionTypes

from Utilities.utils import get_key, get_set
from Utilities.caching import possibly_cached_runner_run
from Utilities.assorted_prompts import PROMPTS
from Utilities.topology import minimum_open_dense_set_gpt_thesis as minimal_dense_set
from Utilities.retrieving_gpt import retrieve_top_chunks
from Utilities.OWL_reasoning import expand_with_hermit
from Utilities.asyncio_utils import async_friendly_input







class HeuristicTestingProcedure(DiagnosticPlan):
    
    test2problem: dict[DiagnosticAction, list[object]]
    
    # outcomeSimplifier = Agent(
    #     name = "outcomeSimplifier", 
    #     instructions = "You will receive in input a diagnostic action outcome. A diagnostic action is a verb-object couple, accompanied by an optional description. Its outcome is a free-text description written by a field service engineer that has executed the action on a real physiscal system. ", 
    #     tools = [function_tool(retrieve_component_context_tool)],
    #     output_type = AnomalousNominalExtractorOutput)
    
    # output: AnomalousNominalExtractorOutput = await possibly_cached_runner_run(agent=anomalousNominalExtractor, input = PROMPTS.AnomalousNominalComponentExtractor_agent_v2_input.value.format(symptom = str(symptom), components = "\n    ".join(str(component) for component in all_components)), cached=configuration.USE_CACHE)
        
    def __bool__(self):
        return len(self.test2problem.keys()) > 0 
    def __len__(self):
        return len(self.test2problem.keys())
    
    def get_information_gain(self, action: DiagnosticAction) -> float:
        """
        Return the expected information gain per unit cost for a diagnostic action.

        The gain is computed as:
            [(A/T)*log2(T/A) + ((T-A)/T))*log2(T/(T-A))]/ cost

        where T is the total number of unique problems and A is the number
        of problems associated with the given action.

        Returns float('inf') if the action covers all remaining problems.
        
        Cost is never supposed to be zero. 
        """
        total_problems = set().union(*self.test2problem.values())
            
        T = total_problems_count = len(total_problems)
        A = associated_problems_count = len(self.test2problem[action])
        B = non_associated_problems_count = total_problems_count - associated_problems_count
        if (total_problems_count == associated_problems_count) or (total_problems_count == non_associated_problems_count):
            return float(0)
        cost = action.get_cost()
        expected_information_gain = ((A/T) * math.log2(T/A)) + ((B/T) * math.log2(T/B))
        return expected_information_gain/cost

    def get_next_action(self, logger: Logger) -> DiagnosticAction | None:
        """
        Return the diagnostic action with the highest information gain.
        Returns None if no actions are available.
        """
        if len(self.test2problem) > 0:
            test2gain = [(action,self.get_information_gain(action), -action.get_cost()) for action in self.test2problem.keys()]
            test2gain.sort(key=lambda x:(x[1], x[2]), reverse=True) #orders by increasing gain and *decreasing* cost. TODO This is O(nlog(n)) and only used for logging. Could be imporved to O(n) 
            logger.debug(f"Sorted actions by gain: {test2gain}")
            return test2gain[0][0]
        else:
            return None
    
    async def update_test_problem_matrix(self, last_action_outcome: DiagnosticActionResult, logger: Logger) -> None:
        """It will reduce the size of the test2problem dictionary depening on the outcome of the last executed action"""
        async def get_simplified_outcome(free_text_outcome: str) -> LiteralType["anomalous", "nominal"]:
            out = await async_friendly_input(f"The action you've executed was {last_action_outcome.action.get_name()} and was aimed to individuate the problem(s) {terminal_uri_parts_gpt(self.test2problem[last_action_outcome.action])}. Did you find in your testing some anomalous behavior that suggests the presence of the aforementioned problems? Write 'anomalous' if so, 'nominal' otherwise\n> ")
            while out not in ["anomalous", "nominal"]:
                out = await async_friendly_input(f"Please, reply with either 'anomalous' or 'nominal'\n> ")
            return out
                
        if not last_action_outcome.simplified_outcome or last_action_outcome.simplified_outcome == "":
            simple_outcome = await get_simplified_outcome(last_action_outcome.outcome)
        else:
            simple_outcome = last_action_outcome.simplified_outcome
        match simple_outcome:
            case 'anomalous': # Anomalous/Y means that anomalous behavior was found by testing: the action is described as a Y/N question, where Y refers to non-nominal behavior, linked to some failure modes, while Nominal/N refers to nominal behavior, and is linked to the complement set of such failure modes. --> 'Y': keep only the linked failure modes; 'N': remove the linked failure modes
                logger.info(f"I am keeping only the problems {terminal_uri_parts_gpt(self.test2problem[last_action_outcome.action])} in my plan and removing the last test")
                # Keeps only problem individuated by test
                self.test2problem = {key: [problem for problem in value if problem in self.test2problem[last_action_outcome.action]] for key, value in self.test2problem.items()}
                # drops tests tha are not linked to problems anymore (if any)
                self.test2problem = {key: value for key, value in self.test2problem.items() if len(value) > 0 }
                # drops the last test
                self.test2problem.pop(last_action_outcome.action, None)
            case 'nominal':
                logger.info(f"I am removing the problems {terminal_uri_parts_gpt(self.test2problem[last_action_outcome.action])} in my plan")
                # Removes problems individuated by test
                self.test2problem = {key: [problem for problem in value if problem not in self.test2problem[last_action_outcome.action]] for key, value in self.test2problem.items()}
                # drops tests that are not linked to problems anymore (at least last_action should be removed)
                self.test2problem = {key: value for key, value in self.test2problem.items() if len(value) > 0 }
            case _:
                raise ValueError(f"Unrecognized value of last_action_outcome.simplified_outcome. Value is {simple_outcome}")
            
class AssistantStateKGO(AssistantState):
    current_pieces_of_evidence: list[set] = Field(default_factory=list)
    current_candidates: set = Field(default_factory=set)
    current_explicit_plan: Optional[HeuristicTestingProcedure] = None


class DiagnosticAssistantEvidenceKGOptimal(DiagnosticAssistant):
    """
    TODO
    """

    def __init__(self, description, configuration):
        super().__init__(description, configuration) # self.state = AssistantState(general_system_description=description)
        self.state = AssistantStateKGO(
            general_system_description = description,
            )

    async def setup(self, observations: list[Observation]) -> None:
        if not self.configuration.ONTOLOGY_PATH: raise ValueError("ONTOLOGY PATH is None, was it an input argument?")
        pieces_of_evidence = await get_pieces_of_evidence_from_many_symptoms(
            ontology_path = self.configuration.KG_PATH, 
            schema_path = self.configuration.ONTOLOGY_PATH, 
            system = self.configuration.SYSTEM_URL, 
            descriptions = [obs.description for obs in observations],
            configuration = self.configuration,
            logger = self.logger)
        pieces_of_evidence = get_qualitative_pieces_of_evidence_from_quantitative(pieces_of_evidence)
        self.state.current_pieces_of_evidence = pieces_of_evidence
        self.state.current_candidates = minimal_dense_set(pieces_of_evidence, self.logger)
        self.logger.info(f"Starting pieces of evidence: {terminal_uri_parts_gpt(self.state.current_pieces_of_evidence)}") 
        self.logger.info(f"Starting candidates: {terminal_uri_parts_gpt(self.state.current_candidates)}")
        self.state.current_explicit_plan = self._create_testing_procedure()
        self.logger.info(f"Created initial testing procedure: {self.state.current_explicit_plan}")

    async def record_outcome(self, last_outcome) -> None:
        self.state.diagnostic_scenario_memory.append(last_outcome) # increases diagnostic memory
        await self.state.current_explicit_plan.update_test_problem_matrix(last_outcome, self.logger)

    
    async def suggest_action(self) -> Optional[DiagnosticAction]:
        self.logger.debug(f"Current explicit plan is {self.state.current_explicit_plan}")
        
        if next_action := self.state.current_explicit_plan.get_next_action(self.logger):
            self.logger.info(f"Got next action from current plan: {next_action.get_name()}")
            return next_action
        else: # plan is currently exhausted, must try instantiating a new plan
            # will discard current candidates and try again 
            self.logger.warning("Current plan exhausted... I will remove the current candidates from each piece of evidence, try to generate a new plan and call my suggest_action method again.")
            for i, piece in enumerate(self.state.current_pieces_of_evidence):
                self.logger.debug(f"piece number {i} was of length {len(piece)}...")
                piece.difference_update(set(self.state.current_candidates))
                self.logger.debug(f"... now it is of length {len(piece)}.")
            self.state.current_pieces_of_evidence = [piece for piece in self.state.current_pieces_of_evidence if len(piece) > 0]
            self.state.current_candidates = minimal_dense_set(self.state.current_pieces_of_evidence, self.logger)
            self.logger.info(f"Starting pieces of evidence: {terminal_uri_parts_gpt(self.state.current_pieces_of_evidence)} --- Starting candidates: {terminal_uri_parts_gpt(self.state.current_candidates)}")
            self.state.current_explicit_plan = self._create_testing_procedure()
            if not self.state.current_explicit_plan: # could be because no more pieces of evidence, but also because no more info from kg
                self.logger.error("Could not generate a plan... Out of ideas: either inject additional knowledge in the knowledge base or try with a new list of observations")
                return None
            return await self.suggest_action() # Will never return None at this point since explicit plan contains something...
            
    #     # Incorporate last outcome if provided
    #     if last_outcome is not None:
    #         self._state.outcomes.append(last_outcome)
    #         if self._state.plan is not None:
    #             self._update_plan_after_outcome(last_outcome)

    #     plan = self._state.plan
    #     if not plan or plan.status != PlanStatus.ONGOING:
    #         return None

    #     if plan.current_index >= len(plan.actions):
    #         plan.status = PlanStatus.EXHAUSTED
    #         return None

    #     return plan.actions[plan.current_index]

    # def finish_session(self, root_cause: Optional[RootCauseDescription]) -> None:
    #     self._state.user_finished = True
    #     if root_cause:
    #         self._state.user_confirmed_root_cause = RootCauseHypothesis(
    #             component=root_cause.component,
    #             failure_mode=root_cause.failure_mode,
    #             confidence=1.0,
    #             proposed_by="user",
    #         )

    # # ---- internal planning logic specific to this concrete assistant ----

    def _create_testing_procedure(self) -> HeuristicTestingProcedure:
        if len(self.state.current_candidates) == 0:
            self.logger.info("Cannot generate a new HeuristicTestingProcedure: candidate list is empty")
            return None
        
        test_problem_cost = get_finest_problems_tests_from_components(
            ontology_path=self.configuration.KG_PATH, 
            schema_path=self.configuration.ONTOLOGY_PATH,
            subjects=set(URIRef(component_str) for component_str in self.state.current_candidates),
            expand=False)
        self.logger.debug(f"""I queried to get test_problem_cost matrix. I queried with
                    ontology_path={self.configuration.KG_PATH},
                    schema_path={self.configuration.ONTOLOGY_PATH},
                    subjects={set(self.configuration.ONTOLOGY_NAMESPACE[component_str] for component_str in self.state.current_candidates)},
                    expand={False},
                    
                    And got test_problem_cost = {test_problem_cost}
                        """)
        def from_URIRef_to_diagnostic_action(uri: URIRef) -> DiagnosticAction:
            type, target, description = get_diagnostic_action_properties(ontology_path=self.configuration.KG_PATH, subject=uri)
            return DiagnosticAction(type=type, target=target, description=description)
        
        test2problem = {}
        for row in test_problem_cost:
            test = from_URIRef_to_diagnostic_action(row[0])
            if test not in test2problem:
                test2problem.update({test: []})
            test2problem[test].append(problem:=row[1])
    
        self.logger.debug(f"Returning test2problem = {test2problem}")
        return HeuristicTestingProcedure(test2problem=test2problem)

    # def _update_plan_after_outcome(self, outcome: DiagnosticActionResult) -> None:
    #     plan = self._state.plan
    #     if not plan or plan.status != PlanStatus.ONGOING:
    #         return

    #     # Advance index if expected action matches outcome.action
    #     if (
    #         plan.current_index < len(plan.actions)
    #         and plan.actions[plan.current_index] == outcome.action
    #     ):
    #         plan.current_index += 1

    #     # Example: no complex re-planning here, but you could adapt
    #     if plan.current_index >= len(plan.actions):
    #         plan.status = PlanStatus.EXHAUSTED
            
            
    # qual_ev = []
    # while True:
    #     pieces_of_evidence = get_pieces_of_evidence_from_many_symptoms(
    #         ontology_path, 
    #         System, 
    #         descriptions=descriptions)
    #     new_qual_ev=get_qualitative_pieces_of_evidence_from_quantitative(pieces_of_evidence)
    #     print(f"{'qual evidence pieces from this turn':*^120}")
    #     pp(new_qual_ev)
    #     qual_ev += new_qual_ev
    #     print(f"{'total evidence pieces from all turns':*^120}")
    #     pp(qual_ev)
    #     print(f"{'minimal dense set from evidence pieces':*^120}")
    #     pp(minimal_belief:=minimal_dense_set(qual_ev)) #belief according to the logix aybu''ke O''zgu''n (Baltag A. Bezhanishvili, Smets, S.) (for computation)/ Evidence logic pacuit eric, johan van benden older
    #     print('get minimal')
    #     print(f"Analyze any/some/all of the above components and report again your observations")
            
            
            
            




#################################################
#### BLOCK FOR QUERYING ONTOLOGY
#################################################
# ZORRO = Namespace("http://www.example.org/zorro/")

def query_ontology_with_subject_object_query(ontology_path: str, schema_path: str, query: str, subject: URIRef, expand: bool = False) -> set[URIRef]:
    if not isinstance(subject, URIRef):
        raise ValueError(f"Got in input {subject} of type {type(subject)} instead of URIRef!")
    
    if expand:
        graph = expand_with_hermit(ontology_path, schema_path)
    else:
        graph = Graph().parse(ontology_path)

    results = graph.query(
        query,
        initBindings={"subject": subject}
    )

    return {row.object for row in results}

def query_ontology_with_subjects_object_query(ontology_path: str, schema_path:str, query: str, subjects: set[URIRef], expand: bool = False) -> set[URIRef]:
    results = set()
    for subject in subjects:
        results.update(query_ontology_with_subject_object_query(ontology_path, schema_path, query, subject, expand))
    return results

def get_diagnostic_action_properties(ontology_path: str, subject: URIRef) -> tuple[diagnosticActionTypes, str, str]:
    query = """
    PREFIX : <http://www.example.org/zorro/>

    SELECT DISTINCT ?type ?target (STR(?desc) as ?description)
    WHERE {
        ?subject rdf:type ?type . ?type rdfs:subClassOf :DiagnosticAction .
        OPTIONAL {?subject :hasTarget ?target}
        OPTIONAL {?subject rdfs:comment ?desc}
    }
    """
    graph = Graph().parse(ontology_path)

    results = graph.query(
        query,
        initBindings={"subject": subject}
    )

    if len(results) != 1:
        raise ValueError(f"Zero or more than one line of attributes returned for action {subject}. It should be just one. They are: {results.serialize(format='csv')}\n\n Inputs are {ontology_path} --- {subject}")
    result = list(results)[0]
    return (split_uri(str(result.type))[1], str(result.target), result.description)

def get_component_closure(ontology_path: str, schema_path: str, subject: URIRef) -> set[URIRef]:
    """Given ontology path and an entity of the ontology, retrieves all the entities of the ontology that are (in)directly related to the entity by the parthood relation (the entity itself is included). This method assumes that the component partonomy is a tree! It will not work otherwise. """
    
    query = """
    PREFIX : <http://www.example.org/zorro/>

    SELECT DISTINCT ?object
    WHERE {
        ?subject (^:hasSubComponent)* ?top .
        ?top :hasSubComponent* ?object .
        FILTER NOT EXISTS {?supertop :hasSubComponent ?top}
    }
    """

    return query_ontology_with_subject_object_query(ontology_path, schema_path, query, subject)

def get_leaf_components(ontology_path: str, schema_path: str, subject: URIRef) -> set[URIRef]:
    query = """
    PREFIX : <http://www.example.org/zorro/>

    SELECT DISTINCT ?bottom
    WHERE {
        ?subject (^:hasSubComponent)* ?top .
        ?top :hasSubComponent* ?bottom .
        FILTER NOT EXISTS {?supertop :hasSubComponent ?top}
        FILTER NOT EXISTS {?bottom :hasSubComponent ?superbottom}
    }
    """
    return query_ontology_with_subject_object_query(ontology_path, schema_path, query, subject)

def get_subcomponents(ontology_path: str, schema_path: str, system: URIRef) -> set[URIRef]:
    """Given ontology path and an entity of the ontology, retrieves all the subcomponents of the entity (the entity itself is excluded)"""
    
    query = """
    PREFIX : <http://www.example.org/zorro/>

    SELECT DISTINCT ?object
    WHERE {
        ?subject :hasSubComponent+ ?object .
    }
    """
    return query_ontology_with_subject_object_query(ontology_path, schema_path, query, system)
    

def supercomponents(ontology_path: str, schema_path: str, system: URIRef) -> set[URIRef]:
    """Given ontology path and an entity of the ontology, retrieves all the supercomponents of the entity (the entity itself is excluded)"""
    
    query = """
    PREFIX : <http://www.example.org/zorro/>

    SELECT DISTINCT ?object
    WHERE {
        ?subject (^:hasSubComponent)+ ?object .
    }
    """
    return query_ontology_with_subject_object_query(ontology_path, schema_path, query, system)


#################################################
#### BLOCK FOR LLM(+RETRIEVING)-BASED COMPONENTS EXTRACTION AND CLASSIFICATION FROM TEXT
#################################################

# given a symptom description, a knowledge graph, and a machine (node in the graph), extracts the list of that machine components from graph, then calls an LLM to get three subsets: the set of components that the symptom suggests are behaving anomalously, those that are behaving nominally, and those that no information is known about from the symptom description. 

class AnomalousNominalExtractorOutput(BaseModel):
  components_suggesting_anomaly_presence: list[str]
  components_suggesting_nominal_behavior: list[str]
  
# @function_tool # cannot work due to the Logger argument
def retrieve_component_context(component_description: str, logger: Logger, client: OpenAI, folder_path: str, top_k: int, chunk_size: int, chunk_overlap: int, tokenizer_model: str, embed_model: str, cache_path: str) -> str:
    """
    Given a component name or description, returns some text that contextualizes the component. 
    
    :param component_description: A natural language description of the component you wish to retrieve information about
    :type component_description: str
    :return: Some texts describing the component, taken from various sources. They are concatenated into a single string together with the names of their source documents 
    :rtype: str
    """
    logger.info(f"Calling retrieving tool with query '{component_description}'...")
    if not folder_path: raise ValueError("Retrieval folder path is empty, check input arguments...")
    top_chunks = retrieve_top_chunks(query=component_description, client=client, top_k=top_k, folder_path=folder_path, chunk_size = chunk_size, chunk_overlap = chunk_overlap, embed_model=embed_model, tokenizer_model=tokenizer_model, cache_path=cache_path)
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in top_chunks
    )
    logger.info(f"Retrieved context: {to_one_line(context[:100])}...")
    # logger.debug(f"Retrieved context (complete): {context}")
    return context
    
# retrieve_component_context("Switch", 
#                            logger=root, 
#                            client=OpenAI(), 
#                            folder_path="/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Unstructured_knowledge_sources/3_cubes",
#                            chunk_overlap=2,
#                            top_k=4,
#                            chunk_size=20,
#                            cache_path="embeddings_cache.pkl",
#                            embed_model="text-embedding-3-small",
#                            tokenizer_model="cl100k_base",
#                            )
# quit()

async def get_components_behaving_anomalously_nominally_from_one_symptom(ontology_path: str, schema_path: str, system: URIRef, symptom: str, configuration: Configuration, logger: Logger) -> tuple[list[URIRef], list[URIRef]]:
    if not isinstance(symptom, (str, Observation)):
        raise ValueError(f"Input a symptom of type {type(symptom)} instead of str. Raw value: {symptom}")
    
    all_components = get_subcomponents(ontology_path, schema_path, system)
    all_components = list(all_components)
    all_components.sort(key=str)

    if not all_components:
        raise ValueError(f"No component found when querying kg at location {schema_path} with system {system}! Check if there even is such a system in the kg?")
        
        
    logger.info(f"The set of all possible candidate components is: {[split_uri(x)[1] for x in all_components]}")
    prompt = PROMPTS.AnomalousNominalComponentExtractor_agent_v2.value
    
    def retrieve_component_context_tool(component_description: str) -> str:
        """
        Given a component name or description, returns some text that contextualizes the component. 
        
        :param component_description: A natural language description of the component you wish to retrieve information about
        :type component_description: str
        :return: Some texts describing the component, taken from various sources. They are concatenated into a single string together with the names of their source documents 
        :rtype: str
        """
        return retrieve_component_context(
            component_description=component_description,
            logger=logger,
            client=configuration.CLIENT,
            top_k=configuration.TOP_K,
            folder_path=configuration.RETRIEVAL_FOLDER_PATH,
            chunk_size=configuration.CHUNK_SIZE,
            chunk_overlap=configuration.CHUNK_OVERLAP,
            embed_model=configuration.EMBED_MODEL,
            tokenizer_model=configuration.TOKENIZER_MODEL,
            cache_path=configuration.CACHE_PATH,
        )

    anomalousNominalExtractor = Agent(
        name = "anomalousNominalExtractor", 
        instructions = prompt, 
        tools = [function_tool(retrieve_component_context_tool)],
        output_type = AnomalousNominalExtractorOutput,
        model = configuration.NS_ASSISTANT_MODEL)
    
    logger.debug(f"Entering in possibly cached with configuration.USE_CACHE {configuration.USE_CACHE}")
    output: AnomalousNominalExtractorOutput = await possibly_cached_runner_run(agent=anomalousNominalExtractor, input = PROMPTS.AnomalousNominalComponentExtractor_agent_v2_input.value.format(symptom = str(symptom), components = "\n    ".join(str(component) for component in all_components)), cached=configuration.USE_CACHE)
    logger.info(f"from the symptom '{symptom}' found {output}")
    return [URIRef(url_str) for url_str in output.components_suggesting_anomaly_presence], [URIRef(url_str) for url_str in output.components_suggesting_nominal_behavior]
    

#################################################
#### BLOCK FOR SYMPTOM -> CANDIDATES COMPONENTS EVIDENCE REASONING
#################################################

async def get_pieces_of_evidence_from_one_symptom(ontology_path: str, schema_path: str, system: URIRef, description: str, configuration: Configuration, logger: Logger) -> dict[str, float]:
    if not isinstance(description, (str, Observation)):
        raise ValueError(f"Input a symptom of type {type(description)} instead of str, raw value: {description}")
    pieces_of_evidence = dict()
    #TODO we are forcing simple bba remember to also test other possibility
    components_suggesting_anomaly_presence, components_suggesting_nominal_behavior = await get_components_behaving_anomalously_nominally_from_one_symptom(ontology_path, schema_path, system, description, configuration, logger)
    # I act component by component and keep the from-ko and from-ok evidence separate: taking the union would already be a committment on the reasoning algorithm. Taking the complement is already a strong committment, I don't want to make more of those. TODO: think about this
    for component_ko in components_suggesting_anomaly_presence:
        putative_kos_from_ko = get_putative_failed_components_from_component_behaving_anomalously(ontology_path, schema_path, component_ko)
        pieces_of_evidence.update({get_key(putative_kos_from_ko): 0.5})
        logger.info(f"from the component_ko '{component_ko}' dependent compoments were added: putative_kos_from_ko = '{[split_uri(x)[1] for x in putative_kos_from_ko]}'")
    for component_ok in components_suggesting_nominal_behavior:
        putative_kos_from_ok = get_putative_failed_components_from_component_behaving_nominally(ontology_path, schema_path, component_ok)
        pieces_of_evidence.update({get_key(putative_kos_from_ok): 0.5})
        logger.info(f"from the component_ok '{component_ok}' non-dependent component were added: putative_kos_from_ok = '{[split_uri(x)[1] for x in putative_kos_from_ok]}'")
    return pieces_of_evidence

async def get_pieces_of_evidence_from_many_symptoms(ontology_path: str, schema_path: str, system: URIRef, descriptions: list[str], configuration: Configuration, logger: Logger) -> dict[str, float]:
    if not isinstance(descriptions, list):
        raise ValueError(f"Input a symptom of type {type(descriptions)} instead of list[str]")
    pieces_of_evidence = dict()
    # Already, we have to make a commitment on how to combine evidence from multiple symptoms. In this case I take the logic of: if already present do nothing. Alternative logic could be: if already present increase strength. TODO: think about this
    for description in descriptions:
        logger.info(f"I am going to produce evidence from this observation-description: '{description}'")
        pieces_of_evidence.update(await get_pieces_of_evidence_from_one_symptom(ontology_path, schema_path, system, description, configuration, logger))
    return pieces_of_evidence
    
def get_qualitative_pieces_of_evidence_from_quantitative(pieces_of_evidence: dict[str, float]) -> list[set[str]]:
    return [get_set(key) for key in pieces_of_evidence.keys()]

def get_putative_failed_components_from_component_behaving_anomalously(ontology_path: str, schema_path: str, subject: URIRef, expand: bool = False) -> set[URIRef]:
    """
    Given a component URI, return the set of components
    that the subject functionally depends on via:
    hasSubComponent* / hasFunction / dependsOn* / ^hasFunction
    
    Note that starting component is always included
    """

    query = """
    PREFIX : <http://www.example.org/zorro/>

    SELECT DISTINCT ?object
    WHERE {
        ?subject :hasSubComponent*/:hasFunction/(:dependsOn|:monitors)*/^:hasFunction ?object .
    }
    """
    return query_ontology_with_subject_object_query(ontology_path, schema_path, query, subject, expand)
    
def get_putative_failed_components_from_component_behaving_nominally(ontology_path: str, schema_path: str, subject: URIRef, expand: bool = False) -> set[URIRef]:
    """
    Given a component URI, return the complement of the set of components
    that are functionally depended on via:
    hasFunction / dependsOn* / ^hasFunction
    
    Note that starting component is always excluded
    """

    all_components = get_component_closure(ontology_path, schema_path, subject)
    putative_nominal_componets = get_putative_failed_components_from_component_behaving_anomalously(ontology_path, schema_path, subject, expand)
    return all_components.difference(putative_nominal_componets)


#################################################
#### BLOCK FOR CANDIDATE COMPONENTS -> CANDIDATE PROBLEMS REASONING
#################################################

def get_problems_from_component(ontology_path: str, schema_path:str, subject: URIRef, expand: bool = False) -> set[URIRef]:
    """
    Given a component URI, return the set of finest associated problems, related by either failsVia/hasCause* or hasFunction / defines /hasCause* -- in the case reasoning on property chains was not executed. A problem is finest if no cause for it is explicitely listed in the knowledge graph. 
    """

    query = """
    PREFIX : <http://www.example.org/zorro/>

    SELECT DISTINCT ?object
    WHERE {
        ?subject :failsVia|(:hasFunction/:defines) ?object .
        	FILTER NOT EXISTS {?object :hasCause ?finerCause .}
    }
    """
    return query_ontology_with_subject_object_query(ontology_path, schema_path, query, subject, expand)

def get_finest_problems_from_components(ontology_path: str, schema_path: str, subjects: set[URIRef], expand: bool = False) -> set[URIRef]:
    """
    Given a set of component URIs, return the set of associated problems, related by either failsVia/hasCause* or hasFunction / defines /hasCause* -- in the case reasoning on property chains was not executed. A problem is finest if no cause for it is explicitely listed in the knowledge graph.
    """
    
    query = """
    PREFIX : <http://www.example.org/zorro/>

    SELECT DISTINCT ?object
    WHERE {
        ?subject (:failsVia|(:hasFunction/:defines))/:hasCause* ?object .
            FILTER NOT EXISTS {?object :hasCause ?finerCause .}
    }
    """

    query_ontology_with_subjects_object_query(ontology_path, schema_path, query, subjects, expand)

def materialize_cost(input_file, import_file, output_file=None):
    """If no output the input fille will be updated in-place"""
    g = Graph()
    g.parse(input_file)
    g.parse(import_file)

    # Shorter...
    query = """
    PREFIX : <http://www.example.org/zorro/>

    INSERT {?diagnosticActionInstance :hasCost ?cost}
    WHERE {
        ?diagnosticActionInstance rdf:type/rdfs:subClassOf* ?DiagnosticActionClass . 
        ?DiagnosticActionClass rdfs:subClassOf :DiagnosticAction .
        ?DiagnosticActionClass rdfs:subClassOf ?restriction .
        ?restriction rdf:type owl:Restriction .
        ?restriction owl:onProperty :hasCost .
        ?restriction owl:hasValue ?cost .
    }
    """
    g.update(query)

    ## Faster...
    # # http://www.example.org/zorro/DiagnosticAction
    # Z = Namespace("http://www.example.org/zorro/")
    # g.bind("", Z)

    # added = 0
    # print(f"I am before loop")
    # # Step 1: iterate subclasses of DiagnosticAction
    # for s in g.subjects(RDFS.subClassOf, Z.DiagnosticAction):
    #     print(f"I am in {s}")

    #     # Step 2: find owl:Restriction with hasCost value
    #     for restriction in g.objects(s, RDFS.subClassOf):
    #         if (restriction, RDF.type, OWL.Restriction) in g and \
    #            (restriction, OWL.onProperty, Z.hasCost) in g:

    #             cost = g.value(restriction, OWL.hasValue)

    #             if cost is None:
    #                 continue
                
    #             print(f"I found cost {cost}")

    #             # Step 3: find individuals of class s
    #             for i in g.subjects(RDF.type, s):

    #                 triple = (i, Z.hasCost, cost)

    #                 if triple not in g:
    #                     g.add(triple)
    #                     added += 1
    ## print(f"Added {added} materialized hasCost assertions.")

    if output_file is None:
        output_file = input_file

    g.serialize(destination=output_file, format="turtle")

    print(f"Saved to: {output_file}")

    
# materialize_cost("/Users/francescocompagno/Desktop/Work_Units/UvA/Experiments/Naive_failure_simulation/Structured_knowledge_sources/3_cubes/zorro-ontology-3-cubes-abox.ttl", "/Users/francescocompagno/Desktop/Work_Units/UvA/Experiments/Naive_failure_simulation/Structured_knowledge_sources/zorro-ontology-tbox.ttl");quit("quitting...")
    
def get_finest_problems_tests_from_components(ontology_path: str, schema_path: str, subjects: set[URIRef], expand: bool = False) -> list[tuple[URIRef, URIRef, LiteralRDF]]:
    """
    Given a set of component URIs, return the associated problem-test matrix
    """
    
    query = """
    PREFIX : <http://www.example.org/zorro/> 

SELECT DISTINCT ?test ?problem ?cost
WHERE {
    ?subject (:failsVia|(:hasFunction/:defines))/:hasCause* ?problem .
    ?problem :hasTest|:hasRepairAction ?test .
    ?test :hasCost ?cost .
    
    VALUES ?subject { %s }  

    FILTER NOT EXISTS { ?problem :hasCause ?finerCause . }
    
}
ORDER BY ?test ?problem ?cost

    """ % " ".join(f"<{str(s)}>" for s in subjects)
    
    if expand:
        graph = expand_with_hermit(ontology_path, schema_path)
    else:
        graph = Graph().parse(ontology_path)

    results = graph.query(
        query,
    )

    return [(row.test, row.problem, row.cost) for row in results]

# import rdflib
# out = get_finest_problems_tests_from_components(
#     ontology_path="/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Structured_knowledge_sources/3_cubes/zorro-ontology-3-cubes-abox.ttl",
#     schema_path="/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl",
#     subjects={rdflib.term.URIRef('http://www.example.org/zorro/Battery')},
#     expand=False,
# )
# out = get_diagnostic_action_properties(
#     ontology_path="/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Structured_knowledge_sources/3_cubes/zorro-ontology-3-cubes-abox.ttl",
#     subject=rdflib.term.URIRef('http://www.example.org/zorro/InspectBattery'),
# )
# def from_URIRef_to_diagnostic_action(uri: URIRef) -> DiagnosticAction:
#     type, target, description = get_diagnostic_action_properties(
#         ontology_path="/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Structured_knowledge_sources/3_cubes/zorro-ontology-3-cubes-abox.ttl",
#         subject=uri
#     )
#     return DiagnosticAction(type=type, target=target, description=description)
# test2problem = {}
# for row in out:
#     test = from_URIRef_to_diagnostic_action(row[0])
#     if test not in test2problem:
#         test2problem.update({test: []})
#     test2problem[test].append(problem:=row[1])
# print("*"*200)
# print(out)
# print("*"*200)
# quit()

def get_finest_problems_tests_gain_from_components(ontology_path: str, schema_path: str, subjects: set[URIRef], expand: bool = False) -> list[tuple[URIRef]]:
    """
    Given a set of component URIs, return the associated problem-test matrix, with also information gain
    """
    
    query = """
    PREFIX : <http://www.example.org/zorro/>

SELECT ?subject ?problem ?test ?cost ?problemCount
WHERE {
    ?subject (:failsVia|(:hasFunction/:defines))/:hasCause* ?problem .
    ?problem :hasTest|:hasRepairAction ?test .
    ?test :hasCost ?cost .
    
    VALUES ?subject { %(values)s }  

    FILTER NOT EXISTS { ?problem :hasCause ?finerCause . }
    
    # Subquery to count number of problems per test
    {
        SELECT ?test (COUNT(DISTINCT ?problem) AS ?problemCount)
        WHERE {
            ?subject (:failsVia|(:hasFunction/:defines))/:hasCause* ?problem .
            ?problem :hasTest|:hasRepairAction ?test .
            ?test :hasCost ?cost .
            
            VALUES ?subject { %(values)s }  # same subjects as above
            FILTER NOT EXISTS { ?problem :hasCause ?finerCause . }
        }
        GROUP BY ?test
    }
}
ORDER BY ?test ?problem ?cost

    """ % {"values": " ".join(f"<{str(s)}>" for s in subjects)}
    
    if expand:
        graph = expand_with_hermit(ontology_path, schema_path)
    else:
        graph = Graph().parse(ontology_path)

    results = graph.query(
        query,
    )

    totalProblemsCount = len(list({row.problem for row in results}))
    def get_information_gain(totalProblemsCount: int, associatedProblemsCount: int, cost: int):
        return math.log2(totalProblemsCount/(totalProblemsCount-associatedProblemsCount))/cost
    return [(row.problem, row.test, row.cost, row.problemCount, totalProblemsCount, get_information_gain(totalProblemsCount, row.problemCount, row.cost)) for row in results]
# pp(get_finest_problems_tests_from_components("/Users/francescocompagno/Desktop/Work_Units/UvA/Experiments/Naive_failure_simulation/Structured_knowledge_sources/3_cubes/zorro-ontology-3-cubes-abox.ttl", subjects = {ZORRO["Battery"], ZORRO["Switch"]}, expand=False));quit("quitting...")

def get_information_gain_of_diagnostic_action(ontology_path: str, schema_path: str, subjects: set[URIRef], expand: bool = False) -> list[tuple[URIRef,URIRef,URIRef]]:
    """
    Given a set of component URIs, return the associated problem-test matrix
    """
    
    query = """
    PREFIX : <http://www.example.org/zorro/>

    SELECT DISTINCT ?problem ?test ?cost
    WHERE {
        ?subject (:failsVia|(:hasFunction/:defines))/:hasCause* ?problem .
        ?problem :hasTest|:hasRepairAction ?test . ?test :hasCost ?cost
        
        VALUES ?subject { %s }
        
            FILTER NOT EXISTS {?problem :hasCause ?finerCause .}
    }
    ORDER BY ?problem ?test ?cost
    """ % " ".join(f"<{str(s)}>" for s in subjects)
    
    if expand:
        graph = expand_with_hermit(ontology_path, schema_path)
    else:
        graph = Graph().parse(ontology_path)

    results = graph.query(
        query,
    )

    return [(row.problem, row.test, row.cost) for row in results]
# pp(get_finest_problems_tests_from_components("/Users/francescocompagno/Desktop/Work_Units/UvA/Experiments/Naive_failure_simulation/Structured_knowledge_sources/3_cubes/zorro-ontology-3-cubes-abox.ttl", subjects = {ZORRO["Battery"], ZORRO["Switch"]}, expand=False));quit("quitting...")

if __name__ == "__main__":
    # # onotology_path = "/Users/francescocompagno/Desktop/Work_Units/UvA/Experiments/Naive_failure_simulation/zorro2-copy-modified+manual-instances.ttl"
    # ontology_path = "/Users/francescocompagno/Desktop/Work_Units/UvA/Experiments/Naive_failure_simulation/zorro2-copy-modified+manual-instances-expanded.ttl"
    
    # InvertedCablesIndicator = URIRef(base=ZORRO, value="InvertedCablesIndicator") #http://www.example.org/zorro/InvertedCablesIndicator
    # g=Graph().parse(ontology_path)
    # q="DESCRIBE :InvertedCablesIndicator"
    # g.query(q).serialize("tempfile.tmp")
    # visualize(Graph().parse("tempfile.tmp", format="xml"))
    # quit()
    # print("Input: ", InvertedCablesIndicator)
    # results = get_putative_failed_components_from_component_behaving_anomalously(ontology_path, InvertedCablesIndicator, expand = False)
    # print("Output: ", results)
    # print("Input: ", str("The control module led is on"))
    # # InvertedCablesIndicator = URIRef(base=ZORRO, value="System") #http://www.example.org/zorro/System
    # # results = get_components_behaving_anomalously_nominally_from_one_symptom(ontology_path, InvertedCablesIndicator, symptom=str("The control module led is on"))
    # # print("Output: ", results)
    # InvertedCablesIndicator = URIRef(base=ZORRO, value="InvertedCablesIndicator") 
    # Battery = URIRef(base=ZORRO, value="Battery") 
    # System = URIRef(base=ZORRO, value="System") 
    # PowerSupplyModule = URIRef(base=ZORRO, value="PowerSupplyModule") 
    # ControlModule = URIRef(base=ZORRO, value="ControlModule") 
    # LoadModule = URIRef(base=ZORRO, value="LoadModule") 
    
    # print(get_putative_failed_components_from_component_behaving_anomalously(ontology_path, InvertedCablesIndicator))
    # quit()
    # o=get_putative_failed_components_from_component_behaving_anomalously(ontology_path=ontology_path, subject=InvertedCablesIndicator)
    # print(o) 
    # quit()
    # print("Input: ", ontology_path, System, [str("The led on the control module is on")])
    
    # descriptions=[
    #         "The led on the control module is on",
    #         "The battery is working properly"
    #         ]
    # pieces_of_evidence = get_pieces_of_evidence_from_many_symptoms(
    #     ontology_path, 
    #     System, 
    #     descriptions=descriptions)
    # print("Output: ", results)
    # --- SD + minimal dense (qualitative) ---
    # print("\nSD belief model (qualitative reasoning):")
    # possible_candidates = {str(x) for x in get_component_closure(ontology_path, System)}
    # # next lines homogenizes granularity
    # possible_candidates = possible_candidates.difference({str(x) for x in {System, LoadModule, ControlModule, PowerSupplyModule}})
    # result_sd = sd_min_belief_model(possible_candidates, results)
    # print(111111,len(possible_candidates))
    # for x in possible_candidates:
    #     print(x)
    # print(22222, evidences:={
    #     get_key({str(Battery), str(InvertedCablesIndicator)}):0.5,
    #     get_key(possible_candidates.difference({str(Battery)})):0.5
    #     })
    # # result_sd = sd_min_belief_model(possible_candidates, evidences)

    # for k in result_sd:
    #     print(f"belief for {k}")

    # print("not belief for other propositions")
        
        
        
        
    # ontology_path = "/Users/francescocompagno/Desktop/Work_Units/UvA/Experiments/Naive_failure_simulation/zorro2-copy-modified+manual-instances-expanded.ttl"
    # System = URIRef(base=ZORRO, value="System") 
    # qual_ev = []
    # print("Game start...")
    # while True:
    #     inp = ""
    #     descriptions=[]
    #     while inp != "stop":
    #         inp = input("Write observation about system (write 'stop' to end this phase; write 'end' to end game): ")
    #         match inp: #The switched cables indicator is turned on #The battery appears to be working fine
    #             case "end":
    #                 quit('Game terminated, quitting... ')
    #                 break
    #             case "":
    #                 print("Empty input, try again...")
    #                 continue
    #             case "stop":
    #                 print('Stopping phase... ')
    #                 continue
    #         descriptions.append(inp)
    #     print(1212121212)
    #     pieces_of_evidence = get_pieces_of_evidence_from_many_symptoms(
    #         ontology_path, 
    #         System, 
    #         descriptions=descriptions)
    #     new_qual_ev=get_qualitative_pieces_of_evidence_from_quantitative(pieces_of_evidence)
    #     print(f"{'qual evidence pieces from this turn':*^120}")
    #     pp(new_qual_ev)
    #     qual_ev += new_qual_ev
    #     print(f"{'total evidence pieces from all turns':*^120}")
    #     pp(qual_ev)
    #     print(f"{'minimal dense set from evidence pieces':*^120}")
    #     pp(minimal_belief:=minimal_dense_set(qual_ev)) #belief according to the logix aybu''ke O''zgu''n (Baltag A. Bezhanishvili, Smets, S.) (for computation)/ Evidence logic pacuit eric, johan van benden older
    #     print('get minimal')
    #     print(f"Analyze any/some/all of the above components and report again your observations")
    pass
        

        
    
# clear; python -m run_dignostic_scenario --text-input-file /Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Unstructured_knowledge_sources/3_cubes/3_cubes_description.txt --log-level 10 --rounds 5 --kg "/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Structured_knowledge_sources/3_cubes/zorro-ontology-3-cubes-abox.ttl" --system 3CubesSystem --ontology "/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl" --retrieval-folder "/Users/francescocompagno/Desktop/Work_Units/Codebases_to_publish/ESWC_2026_Demo/Knowledge_sources/Unstructured_knowledge_sources/3_cubes"