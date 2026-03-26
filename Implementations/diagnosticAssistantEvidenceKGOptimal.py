import math

from logging import Logger
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, Literal as LiteralType
from agents import Agent, function_tool
from rdflib import Graph, URIRef, Literal as LiteralRDF
from rdflib.plugins.sparql.results.csvresults import *
from rdflib.namespace import split_uri


from configuration import Configuration
from environment_classes import AssistantState, DiagnosticAction, DiagnosticActionResult, DiagnosticAssistant, DiagnosticFaultHypothesis, DiagnosticPlan, HypothesisVerificationResult, Observation, diagnosticActionTypes
from Utilities.formatting import terminal_uri_parts_gpt, to_one_line
from Utilities.utils import get_key, get_set
from Utilities.caching import possibly_cached_runner_run
from Utilities.assorted_prompts import PROMPTS
from Utilities.topology import minimum_open_dense_set_gpt_thesis as minimal_dense_set
from Utilities.retrieving_gpt import retrieve_top_chunks
from Utilities.OWL_reasoning import expand_with_hermit


class HeuristicTestingProcedure(DiagnosticPlan):
    """Represents a greedy testing procedure based on an information heuristic: given a matrix of test/problems (stored as a map test->problems -- test2problem), can suggest the next action and update itself based on the action outcome"""
    test2problem: dict[DiagnosticAction, list[object]]

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

        Returns 0 if the action covers all remaining problems or none of them.

        Cost is never supposed to be zero. 
        """
        total_problems = set().union(*self.test2problem.values())

        T = total_problems_count = len(total_problems)
        A = associated_problems_count = len(self.test2problem[action])
        B = non_associated_problems_count = total_problems_count - associated_problems_count
        if (total_problems_count == associated_problems_count) or (total_problems_count == non_associated_problems_count):
            return float(0)
        cost = action.get_cost()
        expected_information_gain = (
            (A/T) * math.log2(T/A)) + ((B/T) * math.log2(T/B))
        return expected_information_gain/cost

    def get_next_action(self, logger: Logger) -> DiagnosticAction | None:
        """
        Return the diagnostic action with the highest information gain.
        Returns None if no actions are available.
        """
        if len(self.test2problem) > 0:
            test2gain = [(action, self.get_information_gain(
                action), -action.get_cost()) for action in self.test2problem.keys()]
            # orders by decreasing gain and *increasing* cost. TODO This is O(nlog(n)) and only used for logging. Could be imporved to O(n)
            test2gain.sort(key=lambda x: (x[1], x[2]), reverse=True)
            logger.debug(f"Sorted actions by gain: {test2gain}")
            old_action = test2gain[0][0]
            next_action = old_action.model_copy(  # nedded because I froze the model in its definition
                update={
                    "description": old_action.description
                    + "\n\n"
                    + (
                        "This action is to be executed with the goal of individuating the "
                        f"problem(s) {terminal_uri_parts_gpt(self.test2problem[old_action])}. "
                        "While you execute it, please keep a watchful eye for some anomalous "
                        "behavior that suggests the presence of the aforementioned problems. "
                        "If you do find some such suggestions, then write the word 'anomalous' "
                        "as outcome, 'nominal' otherwise. Do not write anything else.\n> "
                    )
                }
            )
            # since I modified the frozen action, I also have to modified the key in the dictionary, otherwise they will not match
            self.test2problem[next_action] = self.test2problem.pop(old_action)
            return next_action
        else:
            return None

    async def update_test_problem_matrix(self, last_action_outcome: DiagnosticActionResult, logger: Logger) -> None:
        """It will reduce the size of the test2problem dictionary depening on the outcome of the last executed action"""
        async def get_simplified_outcome(free_text_outcome: str) -> LiteralType["anomalous", "nominal"]:
            # this old design works but requires human input every time and cannot be automatized. Thus, I inserted (mutata mutatis) the below text into the suggested action. 'anomalous' or 'nominal' should then always be already present in the outcome text.
            # for that to be true, the KGOptimal assistant will modify the description of the actions that are supplied to the service agent
            # out = await async_friendly_input(f"The action you've executed was {last_action_outcome.action.get_name()} and was aimed to individuate the problem(s) {terminal_uri_parts_gpt(self.test2problem[last_action_outcome.action])}. Did you find in your testing some anomalous behavior that suggests the presence of the aforementioned problems? Write 'anomalous' if so, 'nominal' otherwise\n> ")
            # while out not in ["anomalous", "nominal"]:
            #     out = await async_friendly_input(f"Please, reply with either 'anomalous' or 'nominal'\n> ")
            # return out
            free_text_outcome = free_text_outcome.lower()
            anomaly_encountered = 'anomalous' in free_text_outcome
            no_anomaly_encountered = 'nominal' in free_text_outcome
            if anomaly_encountered and no_anomaly_encountered:
                raise ValueError(
                    f"The action outcome {free_text_outcome} contains both the 'anomalous' and the 'nominal' words. It should contain only one of these!")
            if not anomaly_encountered and not no_anomaly_encountered:
                raise ValueError(
                    f"The action outcome {free_text_outcome} contains neither the 'anomalous' nor the 'nominal' words. It should exactly one of these!")
            return 'anomalous' if anomaly_encountered else 'nominal'

        if not last_action_outcome.simplified_outcome:
            simple_outcome = await get_simplified_outcome(last_action_outcome.outcome)
        else:
            simple_outcome = last_action_outcome.simplified_outcome
        match simple_outcome:
            case 'anomalous':  # Anomalous/Y means that anomalous behavior was found by testing: the action is described as a Y/N question, where Y refers to non-nominal behavior, linked to some failure modes, while Nominal/N refers to nominal behavior, and is linked to the complement set of such failure modes. --> 'Y': keep only the linked failure modes; 'N': remove the linked failure modes
                logger.info(
                    f"I am keeping only the problems {terminal_uri_parts_gpt(self.test2problem[last_action_outcome.action])} in my plan and removing the last test")
                # Keeps only problem individuated by test
                self.test2problem = {key: [
                    problem for problem in value if problem in self.test2problem[last_action_outcome.action]] for key, value in self.test2problem.items()}
                # drops tests tha are not linked to problems anymore (if any)
                self.test2problem = {
                    key: value for key, value in self.test2problem.items() if len(value) > 0}
                # drops the last test
                self.test2problem.pop(last_action_outcome.action, None)
            case 'nominal':
                logger.info(
                    f"I am removing the problems {terminal_uri_parts_gpt(self.test2problem[last_action_outcome.action])} in my plan")
                # Removes problems individuated by test
                self.test2problem = {key: [
                    problem for problem in value if problem not in self.test2problem[last_action_outcome.action]] for key, value in self.test2problem.items()}
                # drops tests that are not linked to problems anymore (at least last_action should be removed)
                self.test2problem = {
                    key: value for key, value in self.test2problem.items() if len(value) > 0}
            case _:
                raise ValueError(
                    f"Unrecognized value of last_action_outcome.simplified_outcome. Value is {simple_outcome}")


class AssistantStateKGO(AssistantState):
    """Helper class to keep track of the KGOptimal assistant state"""
    current_pieces_of_evidence: list[set] = Field(default_factory=list)
    current_candidates: set = Field(default_factory=set)
    current_explicit_plan: Optional[HeuristicTestingProcedure] = None


class DiagnosticAssistantEvidenceKGOptimal(DiagnosticAssistant):
    """
    Diagnostic assistant based on greedy information gain heuristic.
    """

    def __init__(self, description, configuration):
        # self.state = AssistantState(general_system_description=description)
        super().__init__(description, configuration)
        self.state = AssistantStateKGO(
            general_system_description=description,
        )

    @property
    def description(self) -> str:
        return super().description + "_" + self.configuration.NS_ASSISTANT_MODEL

    async def setup(self, observations: list[Observation]) -> None:
        if not self.configuration.ONTOLOGY_PATH:
            raise ValueError(
                "ONTOLOGY PATH is None, was it an input argument?")
        pieces_of_evidence = await get_pieces_of_evidence_from_many_symptoms(
            ontology_path=self.configuration.KG_PATH,
            schema_path=self.configuration.ONTOLOGY_PATH,
            system=self.configuration.SYSTEM_URL,
            descriptions=[obs.description for obs in observations],
            configuration=self.configuration,
            logger=self.logger)
        pieces_of_evidence = get_qualitative_pieces_of_evidence_from_quantitative(
            pieces_of_evidence)
        self.state.current_pieces_of_evidence = pieces_of_evidence
        self.state.current_candidates = minimal_dense_set(
            pieces_of_evidence, self.logger)
        self.logger.info(
            f"Starting pieces of evidence: {terminal_uri_parts_gpt(self.state.current_pieces_of_evidence)}")
        self.logger.info(
            f"Starting candidates: {terminal_uri_parts_gpt(self.state.current_candidates)}")
        self.state.current_explicit_plan = self._create_testing_procedure()
        self.logger.info(
            f"Created initial testing procedure: {self.state.current_explicit_plan}")

    async def record_outcome(self, last_outcome) -> None:
        self.state.diagnostic_scenario_memory.append(
            last_outcome)  # increases diagnostic memory
        await self.state.current_explicit_plan.update_test_problem_matrix(last_outcome, self.logger)

    async def suggest_action(self) -> Optional[DiagnosticAction | DiagnosticFaultHypothesis]:
        """
        Return the next DiagnosticAction from the current plan, or a
        DiagnosticFaultHypothesis when the plan is exhausted and recovery
        cannot find new candidates (i.e. the last standing candidates are
        the best available hypothesis for the root cause).
        Returns None only if there are genuinely no candidates and no
        evidence remaining.
        """
        self.logger.debug(
            f"Current explicit plan is {self.state.current_explicit_plan}")

        if self.state.current_explicit_plan and (next_action := self.state.current_explicit_plan.get_next_action(self.logger)):
            self.logger.info(
                f"Got next action from current plan: {next_action.get_name()}")
            return next_action
        else:  # plan is currently exhausted, must try instantiating a new plan
            # Save candidates BEFORE removing them so we can report them as a
            # hypothesis if recovery fails to find anything new.
            exhausted_candidates = set(self.state.current_candidates)
            self.logger.warning(
                "Current plan exhausted... I will remove the current candidates from each piece of evidence, try to generate a new plan and call my suggest_action method again.")
            for i, piece in enumerate(self.state.current_pieces_of_evidence):
                self.logger.debug(
                    f"piece number {i} was of length {len(piece)}...")
                piece.difference_update(set(self.state.current_candidates))
                self.logger.debug(f"... now it is of length {len(piece)}.")
            self.state.current_pieces_of_evidence = [
                piece for piece in self.state.current_pieces_of_evidence if len(piece) > 0]
            self.state.current_candidates = minimal_dense_set(
                self.state.current_pieces_of_evidence, self.logger)
            self.logger.info(
                f"Starting pieces of evidence: {terminal_uri_parts_gpt(self.state.current_pieces_of_evidence)} --- Starting candidates: {terminal_uri_parts_gpt(self.state.current_candidates)}")
            self.state.current_explicit_plan = self._create_testing_procedure()
            if not self.state.current_explicit_plan:
                # No new candidates after recovery. The last exhausted candidates
                # are our best hypothesis; emit it for the service agent to verify.
                if exhausted_candidates:
                    self.logger.info(
                        f"Plan fully exhausted. Emitting fault hypothesis for "
                        f"candidates: {terminal_uri_parts_gpt(exhausted_candidates)}")
                    return DiagnosticFaultHypothesis(
                        suspected_components={str(c) for c in exhausted_candidates},
                        explanation="All diagnostic tests exhausted; these are the final remaining candidate components.",
                    )
                self.logger.error(
                    "Could not generate a plan and no candidates remain. "
                    "Out of ideas: inject additional knowledge or revise observations.")
                return None
            # Will never return None at this point since explicit plan contains something...
            return await self.suggest_action()

    async def record_hypothesis_outcome(
        self,
        hypothesis: DiagnosticFaultHypothesis,
        result: HypothesisVerificationResult,
    ) -> None:
        await super().record_hypothesis_outcome(hypothesis, result)
        if result.outcome == "partial":
            self.logger.info(
                "Hypothesis partially confirmed: some suspected components were faulty "
                "but the system is not yet fully restored. Continuing diagnosis."
            )
        elif result.outcome == "wrong":
            self.logger.warning(
                "Hypothesis was wrong: none of the suspected components were faulty. "
                "All candidates exhausted; session will end."
            )

    def _create_testing_procedure(self) -> Optional[HeuristicTestingProcedure]:
        if len(self.state.current_candidates) == 0:
            self.logger.info(
                "Cannot generate a new HeuristicTestingProcedure: candidate list is empty")
            return None

        test_problem_cost = get_finest_problems_tests_from_components(
            ontology_path=self.configuration.KG_PATH,
            schema_path=self.configuration.ONTOLOGY_PATH,
            subjects=set(URIRef(component_str)
                         for component_str in self.state.current_candidates),
            expand=False)
        self.logger.debug(f"""I queried to get test_problem_cost matrix. I queried with
                    ontology_path={self.configuration.KG_PATH},
                    schema_path={self.configuration.ONTOLOGY_PATH},
                    subjects={set(URIRef(component_str) for component_str in self.state.current_candidates)},
                    expand={False},
                    
                    And got test_problem_cost = {test_problem_cost}
                        """)

        def from_URIRef_to_diagnostic_action(uri: URIRef) -> DiagnosticAction:
            type, target, description = get_diagnostic_action_properties(
                ontology_path=self.configuration.KG_PATH, subject=uri)
            return DiagnosticAction(type=type, target=target, description=description)

        test2problem = {}
        for row in test_problem_cost:
            test = from_URIRef_to_diagnostic_action(row[0])
            if test not in test2problem:
                test2problem.update({test: []})
            test2problem[test].append(problem := row[1])

        self.logger.debug(f"Returning test2problem = {test2problem}")
        return HeuristicTestingProcedure(test2problem=test2problem)


#################################################
# CODE BLOCK FOR QUERYING ONTOLOGY
#################################################
# ZORRO = Namespace("http://www.example.org/zorro/")

def query_ontology_with_subject_object_query(ontology_path: str, schema_path: str, query: str, subject: URIRef, expand: bool = False) -> set[URIRef]:
    if not isinstance(subject, URIRef):
        raise ValueError(
            f"Got in input {subject} of type {type(subject)} instead of URIRef!")

    if expand:
        graph = expand_with_hermit(ontology_path, schema_path)
    else:
        graph = Graph().parse(ontology_path)

    results = graph.query(
        query,
        initBindings={"subject": subject}
    )

    return {row.object for row in results}


def query_ontology_with_subjects_object_query(ontology_path: str, schema_path: str, query: str, subjects: set[URIRef], expand: bool = False) -> set[URIRef]:
    results = set()
    for subject in subjects:
        results.update(query_ontology_with_subject_object_query(
            ontology_path, schema_path, query, subject, expand))
    return results


def get_diagnostic_action_properties(ontology_path: str, subject: URIRef) -> tuple[diagnosticActionTypes, Optional[str], Optional[str]]:
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
        raise ValueError(
            f"Zero or more than one line of attributes returned for action {subject}. It should be just one. They are: {results.serialize(format='csv')}\n\n Inputs are {ontology_path} --- {subject}")
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
# BLOCK FOR LLM(+RETRIEVING)-BASED COMPONENTS EXTRACTION AND CLASSIFICATION FROM TEXT
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
    logger.info(
        f"Calling retrieving tool with query '{component_description}'...")
    if not folder_path:
        raise ValueError(
            "Retrieval folder path is empty, check input arguments...")
    top_chunks = retrieve_top_chunks(query=component_description, client=client, top_k=top_k, folder_path=folder_path, chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap, embed_model=embed_model, tokenizer_model=tokenizer_model, cache_path=cache_path)
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in top_chunks
    )
    logger.info(f"Retrieved context: {to_one_line(context[:100])}...")
    return context


async def get_components_behaving_anomalously_nominally_from_one_symptom(ontology_path: str, schema_path: str, system: URIRef, symptom: str, configuration: Configuration, logger: Logger) -> tuple[list[URIRef], list[URIRef]]:
    if not isinstance(symptom, (str, Observation)):
        raise ValueError(
            f"Input a symptom of type {type(symptom)} instead of str. Raw value: {symptom}")

    all_components = get_subcomponents(ontology_path, schema_path, system)
    all_components = list(all_components)
    all_components.sort(key=str)

    if not all_components:
        raise ValueError(
            f"No component found when querying kg at location {schema_path} with system {system}! Check if there even is such a system in the kg?")

    logger.info(
        f"The set of all possible candidate components is: {[split_uri(x)[1] for x in all_components]}")
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
        name="anomalousNominalExtractor",
        instructions=prompt,
        tools=[function_tool(retrieve_component_context_tool)],
        output_type=AnomalousNominalExtractorOutput,
        model=configuration.NS_ASSISTANT_MODEL)

    logger.debug(
        f"Entering in possibly cached with configuration.USE_CACHE {configuration.USE_CACHE}")
    output: AnomalousNominalExtractorOutput = await possibly_cached_runner_run(agent=anomalousNominalExtractor, input=PROMPTS.AnomalousNominalComponentExtractor_agent_v2_input.value.format(symptom=str(symptom), components="\n    ".join(str(component) for component in all_components)), cached=configuration.USE_CACHE)
    logger.info(f"from the symptom '{symptom}' found {output}")
    return [URIRef(url_str) for url_str in output.components_suggesting_anomaly_presence], [URIRef(url_str) for url_str in output.components_suggesting_nominal_behavior]


#################################################
# BLOCK FOR SYMPTOM -> CANDIDATES COMPONENTS EVIDENCE REASONING
#################################################

async def get_pieces_of_evidence_from_one_symptom(ontology_path: str, schema_path: str, system: URIRef, description: str, configuration: Configuration, logger: Logger) -> dict[str, float]:
    if not isinstance(description, (str, Observation)):
        raise ValueError(
            f"Input a symptom of type {type(description)} instead of str, raw value: {description}")
    pieces_of_evidence = dict()
    # TODO we are forcing simple bba remember to also test other possibility
    components_suggesting_anomaly_presence, components_suggesting_nominal_behavior = await get_components_behaving_anomalously_nominally_from_one_symptom(ontology_path, schema_path, system, description, configuration, logger)
    # I act component by component and keep the from-ko and from-ok evidence separate: taking the union would already be a committment on the reasoning algorithm. Taking the complement is already a strong committment, I don't want to make more of those. TODO: think about this
    for component_ko in components_suggesting_anomaly_presence:
        putative_kos_from_ko = get_putative_failed_components_from_component_behaving_anomalously(
            ontology_path, schema_path, component_ko)
        pieces_of_evidence.update({get_key(putative_kos_from_ko): 0.5})
        logger.info(
            f"from the component_ko '{component_ko}' dependent compoments were added: putative_kos_from_ko = '{[split_uri(x)[1] for x in putative_kos_from_ko]}'")
    for component_ok in components_suggesting_nominal_behavior:
        putative_kos_from_ok = get_putative_failed_components_from_component_behaving_nominally(
            ontology_path, schema_path, component_ok)
        pieces_of_evidence.update({get_key(putative_kos_from_ok): 0.5})
        logger.info(
            f"from the component_ok '{component_ok}' non-dependent component were added: putative_kos_from_ok = '{[split_uri(x)[1] for x in putative_kos_from_ok]}'")
    return pieces_of_evidence


async def get_pieces_of_evidence_from_many_symptoms(ontology_path: str, schema_path: str, system: URIRef, descriptions: list[str], configuration: Configuration, logger: Logger) -> dict[str, float]:
    if not isinstance(descriptions, list):
        raise ValueError(
            f"Input a symptom of type {type(descriptions)} instead of list[str]")
    pieces_of_evidence = dict()
    # Already, we have to make a commitment on how to combine evidence from multiple symptoms. In this case I take the logic of: if already present do nothing. Alternative logic could be: if already present increase strength. TODO: think about this
    for description in descriptions:
        logger.info(
            f"I am going to produce evidence from this observation-description: '{description}'")
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
    putative_nominal_componets = get_putative_failed_components_from_component_behaving_anomalously(
        ontology_path, schema_path, subject, expand)
    return all_components.difference(putative_nominal_componets)


#################################################
# BLOCK FOR CANDIDATE COMPONENTS -> CANDIDATE PROBLEMS REASONING
#################################################

def get_problems_from_component(ontology_path: str, schema_path: str, subject: URIRef, expand: bool = False) -> set[URIRef]:
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

    return query_ontology_with_subjects_object_query(ontology_path, schema_path, query, subjects, expand)


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

    # Faster...
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
    # print(f"Added {added} materialized hasCost assertions.")

    if output_file is None:
        output_file = input_file

    g.serialize(destination=output_file, format="turtle")

    print(f"Saved to: {output_file}")


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


if __name__ == "__main__":
    pass
