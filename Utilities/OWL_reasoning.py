from pathlib import Path
from rdflib import Graph
from owlready2 import get_ontology, sync_reasoner_hermit, Ontology, onto_path, World



def ttl_to_rdfxml(ttl_path: str, rdfxml_path: str):
    g = Graph()
    g.parse(ttl_path, format="turtle")
    g.serialize(destination=rdfxml_path, format="xml")
    
def expand_with_hermit(ontology_path: str, schema_path: str) -> Ontology:
    # Convert TTL to RDF/XML if needed
    if ontology_path.endswith(".ttl"):
        ttl_to_rdfxml(ontology_path, xml_ontology_path := ontology_path[:-4] + ".xml")
        ontology_path = xml_ontology_path
    if schema_path.endswith(".ttl"):
        ttl_to_rdfxml(schema_path, xml_schema_path := schema_path[:-4] + ".xml")
        schema_path = xml_schema_path
        
    world = World()

    # Load TBox from local file using file:// URI
    tbox_file_uri = Path(schema_path).absolute().as_uri()  # file://... URI
    tbox_onto = world.get_ontology(tbox_file_uri).load()

    # Load ABox from local file using file:// URI
    abox_file_uri = Path(ontology_path).absolute().as_uri()
    onto: Ontology = world.get_ontology(abox_file_uri).load()

    # Run HermiT and materialize inferences
    with onto:
        sync_reasoner_hermit(infer_property_values=True)

    # Save expanded ontology to a temp file
    # tmp = tempfile.NamedTemporaryFile(suffix=".tmp", delete=False)
    # onto.save(file=tmp.name, format="rdfxml")
    # Save expanded ontology to a permanent file
    onto.save(file=ontology_path[:-4]+".tmp", format="rdfxml")
    

    # Load into rdflib
    g = Graph()
    # g.parse(tmp.name, format="xml")
    g.parse(ontology_path[:-4]+".tmp", format="xml")

    return g

if __name__ == '__main__':
    # ontology_path = "/Users/francescocompagno/Desktop/Work_Units/UvA/Experiments/Naive_failure_simulation/Structured_knowledge_sources/zorro-ontology-tbox.ttl"
    # # path_lib = Path(ontology_path).parent
    # # print(str(path_lib));quit()
    # g=expand_with_hermit(ontology_path)
    # quit('quitting...')
    pass