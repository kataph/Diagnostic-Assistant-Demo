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
from environment_classes import AssistantState, DiagnosticAction, DiagnosticActionResult, DiagnosticAssistant, DiagnosticFaultHypothesis, DiagnosticPlan, HypothesisVerificationResult, Observation, diagnosticActionTypes, SimplifiedOutcome
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
        Returns None if no actions are available (happens iff length of test2problem is 0).
        """
        if len(self.test2problem) > 0:
            test2gain = [(action, self.get_information_gain(
                action), -action.get_cost()) for action in self.test2problem.keys()]
            # orders by decreasing gain and *increasing* cost. TODO This is O(nlog(n)) and only used for logging. Could be imporved to O(n)
            test2gain.sort(key=lambda x: (x[1], x[2]), reverse=True)
            logger.debug(f"Sorted actions by gain: {test2gain}")
            selected_action = test2gain[0][0]
            next_action = selected_action.model_copy(  # needed because I froze the model in its definition
                update={
                    "reporting_requirements": (
                        "This action is to be executed with the goal of individuating the "
                        f"problem(s) {terminal_uri_parts_gpt(self.test2problem[selected_action])}. "
                        "While you execute it, please keep a watchful eye for some anomalous "
                        "behavior that suggests the presence of the aforementioned problems. "
                        f"At the end of your response include the uppercase verdict token {SimplifiedOutcome.ANOMALOUS.value} "
                        f"if you detect any such signs, or {SimplifiedOutcome.NOMINAL.value} if everything appears normal. "
                        "Include exactly one of these two tokens.\n> "
                    )
                }
            )
            # since I modified the frozen action, I also have to modified the key in the dictionary, otherwise they will not match
            self.test2problem[next_action] = self.test2problem.pop(selected_action)
            return next_action
        else:
            return None
    
    def anomaly_encountered(self, free_text_outcome: str) -> bool:
        return SimplifiedOutcome.ANOMALOUS.value in free_text_outcome
    def no_anomaly_encountered(self, free_text_outcome: str) -> bool:
        return SimplifiedOutcome.NOMINAL.value in free_text_outcome
        
    async def update_test_problem_matrix(self, last_action_outcome: DiagnosticActionResult, logger: Logger) -> None:
        """It will reduce the size of the test2problem dictionary depening on the outcome of the last executed action,
        also returns if an anomaly """
        async def get_simplified_outcome(free_text_outcome: str) -> SimplifiedOutcome:
            # this old design works but requires human input every time and cannot be automatized. Thus, I inserted (mutata mutatis) the below text into the suggested action. 'anomalous' or 'nominal' should then always be already present in the outcome text.
            # for that to be true, the KGOptimal assistant will modify the description of the actions that are supplied to the service agent
            # out = await async_friendly_input(f"The action you've executed was {last_action_outcome.action.get_name()} and was aimed to individuate the problem(s) {terminal_uri_parts_gpt(self.test2problem[last_action_outcome.action])}. Did you find in your testing some anomalous behavior that suggests the presence of the aforementioned problems? Write 'anomalous' if so, 'nominal' otherwise\n> ")
            # while out not in ["anomalous", "nominal"]:
            #     out = await async_friendly_input(f"Please, reply with either 'anomalous' or 'nominal'\n> ")
            # return out
            anomaly_encountered = self.anomaly_encountered(free_text_outcome) 
            no_anomaly_encountered = self.no_anomaly_encountered(free_text_outcome)
            if anomaly_encountered and no_anomaly_encountered:
                raise ValueError(
                    f"The action outcome {free_text_outcome} contains both the {SimplifiedOutcome.ANOMALOUS.value} and the {SimplifiedOutcome.NOMINAL.value} tokens. It should contain only one of these!")
            if not anomaly_encountered and not no_anomaly_encountered:
                raise ValueError(
                    f"The action outcome {free_text_outcome} contains neither the {SimplifiedOutcome.ANOMALOUS.value} nor the {SimplifiedOutcome.NOMINAL.value} tokens. It should contain exactly one of these!")
            return SimplifiedOutcome.ANOMALOUS if anomaly_encountered else SimplifiedOutcome.NOMINAL

        if not last_action_outcome.simplified_outcome:
            simple_outcome = await get_simplified_outcome(last_action_outcome.outcome)
        else:
            simple_outcome = last_action_outcome.simplified_outcome
        match simple_outcome:
            case SimplifiedOutcome.ANOMALOUS:  # Anomalous/Y means that anomalous behavior was found by testing: the action is described as a Y/N question, where Y refers to non-nominal behavior, linked to some failure modes, while Nominal/N refers to nominal behavior, and is linked to the complement set of such failure modes. --> 'Y': keep only the linked failure modes; 'N': remove the linked failure modes
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
            case SimplifiedOutcome.NOMINAL:
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
    found_anomaly_in_current_candidates: bool = False
    current_pieces_of_evidence: list[set] = Field(default_factory=list)
    current_candidates: set = Field(default_factory=set)
    current_explicit_plan: Optional[HeuristicTestingProcedure] = None
    # URI string of the last problem emitted as a hypothesis (for absorbing feedback).
    last_hypothesized_problem: Optional[str] = None
    # Problems confirmed absent via "wrong" hypothesis verification.
    # Filtered out of every future plan so they cannot resurface through
    # top-level system components (e.g. 3CubesSystem :failsVia X :hasCause*).
    excluded_problems: set[str] = Field(default_factory=set)


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
        self.state.found_anomaly_in_current_candidates = False
        self.logger.info(
            f"Starting pieces of evidence: {terminal_uri_parts_gpt(self.state.current_pieces_of_evidence)}")
        self.logger.info(
            f"Starting candidates: {terminal_uri_parts_gpt(self.state.current_candidates)}")
        self.state.current_explicit_plan = self._create_testing_procedure()
        self.logger.info(
            f"Created initial testing procedure: {self.state.current_explicit_plan}")

    async def record_action_outcome(self, last_outcome) -> None:
        self.state.diagnostic_scenario_memory.append(
            last_outcome)  # increases diagnostic memory
        if self.state.current_explicit_plan.anomaly_encountered(last_outcome.outcome): # if anomaly encountered remember it for the current candidate set
            self.state.found_anomaly_in_current_candidates = True
        await self.state.current_explicit_plan.update_test_problem_matrix(last_outcome, self.logger)

    def _delete_current_candidates_from_evidence(self) -> None:
        """Deletes the current candidates from the evidence pieces. Modified variables: self.state.current_pieces_of_evidence"""
        for i, piece in enumerate(self.state.current_pieces_of_evidence):
            self.logger.debug(
                f"piece number {i} was of length {len(piece)}...")
            piece.difference_update(set(self.state.current_candidates))
            self.logger.debug(f"... now it is of length {len(piece)}.")
        self.state.current_pieces_of_evidence = [
            piece for piece in self.state.current_pieces_of_evidence if len(piece) > 0]
        self.logger.info(f"New pieces of evidence: {terminal_uri_parts_gpt(self.state.current_pieces_of_evidence)}")
        
    def _set_current_candidates_from_evidence(self) -> None:
        """Set the current candidates from the current evidence pieces using the minimal_dense_set function 
        and sets the found_anomaly_in_current_candidates to False. Modified variables: self.state.current_candidates; self.state.found_anomaly_in_current_candidates"""
        self.state.current_candidates = minimal_dense_set(
            self.state.current_pieces_of_evidence, self.logger)
        self.state.found_anomaly_in_current_candidates = False
        self.logger.info(f"New candidates: {terminal_uri_parts_gpt(self.state.current_candidates)}")
    
    async def suggest_action(self) -> Optional[DiagnosticAction | DiagnosticFaultHypothesis]:
        """
        Return the next DiagnosticAction from the current plan, or a
        DiagnosticFaultHypothesis when the plan is exhausted and recovery
        cannot find new candidates (i.e. the last standing candidates are
        the best available hypothesis for the root cause).
        Returns None only if there are genuinely no candidates and no
        evidence remaining.
        """
        
        # If a single candidate is remaining and an anomaly was found, return it as an hypothesis. 
        # Note that there may be tests to be carried out that we skip: either they will confirm the presence of a specific
        # problem in the current candidate, or they will not. In the former case we will want to emit a similar 
        # hypothesis, in the latter either an error has occurred or the knowledge graph employed is wrong or incomplete. 
        # In the latter case we would still emit a similar hypothesis as a fallback, so it make sense emitting it now. 
        if len(self.state.current_candidates) == 1 and self.state.found_anomaly_in_current_candidates:
            return DiagnosticFaultHypothesis(
                suspected_components=self.state.current_candidates,
                explanation=(
                    f"There is only one current candidate, and an anomaly was found in it"
                ),
            ) 
        
        if self.state.current_explicit_plan:
            self.logger.debug(f"Current explicit plan is {self.state.current_explicit_plan}")
            remaining_problems = set().union(*self.state.current_explicit_plan.test2problem.values())
        else:
            self.logger.debug(f"Explicit plan is either None, empty, or non-initialized")
            remaining_problems = set()
        
        # If a single problem remains and an anomaly was found, return all current candidates that are related to the problem, as an hypothesis
        if len(remaining_problems) == 1 and self.state.found_anomaly_in_current_candidates:
            sole_problem = next(iter(remaining_problems))
            components = {str(x) for x in get_components_from_problem(
                self.configuration.KG_PATH, URIRef(str(sole_problem))
            )} & self.state.current_candidates
            self.state.last_hypothesized_problem = str(sole_problem)
            self.logger.info(
                f"Single problem remaining: {terminal_uri_parts_gpt([sole_problem])}. "
                f"Emitting hypothesis for components: {terminal_uri_parts_gpt(components)}"
            )
            return DiagnosticFaultHypothesis(
                suspected_components=components,
                explanation=(
                    f"Testing procedure led to finding an anomaly and ended up singling out a unique problem: "
                    f"{terminal_uri_parts_gpt([sole_problem])}."
                ),
            )
            
        # Problems exhausted but some anomaly was found: either error by the service agent or a previously unseen problem. Returning current list of candidates as fallback. 
        if len(remaining_problems) == 0 and self.state.found_anomaly_in_current_candidates:
            self.logger.info(
                f"Problems exhausted but some anomaly was found: either error by the service agent or a previously unseen problem. Returning current list of candidates as fallback: {self.state.current_candidates}"
            )
            return DiagnosticFaultHypothesis(
                suspected_components=self.state.current_candidates,
                explanation=(
                    f"Problems exhausted but some anomaly was found: either error by the service agent or a previously unseen problem. Returning current list of candidates as fallback."
                ),
            )
        
                
        # Case that either 
        # (i) no anomaly found OR
        # (ii) more than 1 remaining problem AND more than 1 current candidate
        
        # If there is a plan and from the plan an action can be obtained
        if self.state.current_explicit_plan is not None and (next_action := self.state.current_explicit_plan.get_next_action(self.logger)):
            self.logger.info(
                f"Got next action from current plan: {next_action.get_name()}")
            return next_action
        # If there is no plan or no action can be extracted from the plan
        else:
            # plan is currently exhausted, must try instantiating a new plan
            self.logger.warning("Current plan bacame exhausted or maybe was not able to create testing procedure during setup... I will remove the current candidates from each piece of evidence, try to generate a new plan and call my suggest_action method again.")
            self._delete_current_candidates_from_evidence()
            self._set_current_candidates_from_evidence()
            self.state.current_explicit_plan = self._create_testing_procedure()
            if not self.state.current_explicit_plan:
                # No new candidates after recovery.
                self.logger.error(
                    "Could not generate a plan and no candidates remain. "
                    "Out of ideas: inject additional knowledge or revise observations.")
                return None
            return await self.suggest_action()

    async def record_hypothesis_outcome(
        self,
        hypothesis: DiagnosticFaultHypothesis,
        result: HypothesisVerificationResult,
    ) -> None:
        await super().record_hypothesis_outcome(hypothesis, result)

        if result.outcome == "correct":
            # Session ends — nothing to update.
            return

        prob_str = self.state.last_hypothesized_problem

        if result.outcome == "partial":
            # Some components of the hypothesized problem are confirmed faulty
            # and are already being progressively repaired by the service agent
            # (tracked via _repaired_comp_ids).  Leave the plan unchanged so
            # suggest_action re-emits the same single-problem hypothesis on the
            # next round; test_repair will pick up the remaining components via
            # already_repaired_ids and return "correct" once all are fixed.
            self.logger.info(
                "Hypothesis partially confirmed: service agent is tracking repairs. "
                "Will re-emit hypothesis on next round for remaining components."
            )

        elif result.outcome == "wrong":
            # The hypothesized problem is not real.
            # We must remove the wrong problem's components from current_candidates
            # AND from current_pieces_of_evidence, then rebuild the plan.
            # Purging only test2problem is insufficient: _create_testing_procedure()
            # queries the ontology fresh from current_candidates, which would
            # reintroduce the wrong problem into the new plan.
            self.logger.warning(
                f"Hypothesis wrong: removing problem {prob_str!r} and its "
                f"components from candidates and evidence, then rebuilding plan."
            )
            if prob_str:
                # Permanently exclude this problem from all future plans.
                # This is the primary guard: top-level system components
                # (e.g. 3CubesSystem :failsVia LightbulbDoesNotTurnOn :hasCause*)
                # reach every leaf problem, so component-only pruning is
                # insufficient — the problem resurfaces via those catch-all paths.
                self.state.excluded_problems.add(prob_str)
                wrong_components = get_components_from_problem(
                    self.configuration.KG_PATH, URIRef(prob_str)
                )
                wrong_strs = {str(c) for c in wrong_components}
                self.state.current_candidates -= wrong_strs
                for piece in self.state.current_pieces_of_evidence:
                    piece.difference_update(wrong_components)
                    piece.difference_update(wrong_strs)
                self.state.current_pieces_of_evidence = [
                    p for p in self.state.current_pieces_of_evidence if p
                ]
                self.logger.info(
                    f"Excluded problem {terminal_uri_parts_gpt([URIRef(prob_str)])} from future plans. "
                    f"Removed candidates: {terminal_uri_parts_gpt(wrong_components)}. "
                    f"Remaining candidates: {terminal_uri_parts_gpt(self.state.current_candidates)}"
                )
            self.state.current_explicit_plan = self._create_testing_procedure()
            self.state.last_hypothesized_problem = None

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
        # Filter out problems confirmed absent via "wrong" hypothesis.
        # Top-level system components (e.g. 3CubesSystem :failsVia X :hasCause*)
        # reach every leaf problem, so without this filter a ruled-out problem
        # resurfaces every time the plan is rebuilt.
        if self.state.excluded_problems:
            before = len(test_problem_cost)
            test_problem_cost = [
                row for row in test_problem_cost
                if str(row[1]) not in self.state.excluded_problems
            ]
            removed = before - len(test_problem_cost)
            if removed:
                self.logger.info(
                    f"Filtered {removed} test-problem row(s) for excluded problems: "
                    f"{terminal_uri_parts_gpt(self.state.excluded_problems)}"
                )
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
    return (split_uri(str(result.type))[1], split_uri(str(result.target))[1], result.description)


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
    formatted_components = "\n    ".join(
        f"{split_uri(str(c))[1]} ({str(c)})" for c in all_components
    )
    output: AnomalousNominalExtractorOutput = await possibly_cached_runner_run(agent=anomalousNominalExtractor, input=PROMPTS.AnomalousNominalComponentExtractor_agent_v2_input.value.format(symptom=str(symptom), components=formatted_components), cached=configuration.USE_CACHE)
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

def get_components_from_problem(ontology_path: str, problem: URIRef) -> set[URIRef]:
    """
    Inverse of get_problems_from_component.
    Given a problem URI, return the components that are associated with it via:
        component :failsVia problem
        component :hasFunction / :defines problem
    """
    # query = """
    # PREFIX : <http://www.example.org/zorro/>
    # SELECT DISTINCT ?object
    # WHERE { ?subject (^:hasCause)*/(^:failsVia|(^:defines/^:hasFunction)) ?object }
    # """
    # # the following filters out components that are less specific while having the same problems 
    query = """
    PREFIX : <http://www.example.org/zorro/> 
    SELECT DISTINCT ?object 
    WHERE { 
    ?subject (^:hasCause)*/(^:failsVia|(^:defines/^:hasFunction)) ?object . 
    FILTER NOT EXISTS {?subject (^:hasCause)*/(^:failsVia|(^:defines/^:hasFunction)) ?subcomponent . ?subcomponent ^:hasSubComponent ?object .}
    """
    graph = Graph().parse(ontology_path)
    results = graph.query(query, initBindings={"subject": problem})
    return {row.object for row in results}


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
