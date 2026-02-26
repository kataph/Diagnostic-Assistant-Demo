from rdflib import URIRef
from rdflib.namespace import split_uri
from collections.abc import Iterable

def format_list(l: list)->str:
    if not l:
        return ""
    return "\n".join([str(x) for x in l])+"\n"
    
def format_conversation_history(conversation_history) -> str:
    out = []
    for msg in conversation_history:
        for item in msg.get("content", []):
            if item["type"] == "input_text":
                out.append(item["text"])
            if item["type"] == "input_image":
                out.append(item["file_id"])
    return "\n".join(out)

def split_uri_str(uri: str) -> str:
    """Return the terminal part of a URI string."""
    if '#' in uri:
        return uri.rsplit('#', 1)[-1]
    else:
        return uri.rstrip('/').rsplit('/', 1)[-1]

def terminal_uri_parts_gpt(obj):
    """
    Recursively replace URIRefs or strings in nested iterables with their terminal URI parts.
    """
    if isinstance(obj, URIRef):
        return split_uri_str(str(obj))
    elif isinstance(obj, str):
        return split_uri_str(obj)
    elif isinstance(obj, dict):
        return {terminal_uri_parts_gpt(k): terminal_uri_parts_gpt(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable):
        return type(obj)(terminal_uri_parts_gpt(x) for x in obj)
    else:
        return obj

    
def to_one_line(s: str) -> str:
    """
    Convert any string into a single line by removing newlines and extra spaces.
    """
    return " ".join(s.split())
