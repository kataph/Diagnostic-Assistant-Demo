from rdflib import URIRef
from rdflib.namespace import split_uri
from collections.abc import Iterable


def format_list(l: list) -> str:
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
                out.append(item.get("file_id") or "[inline image]")
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


def to_capital_case(s: str) -> str:
    if len(s) == 0:
        return s
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:]


def to_PascalCase(s: str) -> str:
    l = [y for x in s.split(' ') for y in x.split('_')]
    return "".join([to_capital_case(x) for x in l])


def test_to_pascal_case():
    in_out = [
        ("3_cubes", "3Cubes"),
        ("10_cubes", "10Cubes"),
        ("A b cd", "ABCd"),
        ("snake_case", "SnakeCase"),
    ]
    for inp, out in in_out:
        assert out == to_PascalCase(inp)
    print("All good from test_to_pascal_case")


if __name__ == "__main__":
    test_to_pascal_case()
