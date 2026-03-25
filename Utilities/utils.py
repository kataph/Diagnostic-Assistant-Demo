import ast


def get_key(s):
    """
    This method gives a unique string identifier to a set.

    Args:
        s (set): Set.

    Returns:
        k (string): Unique string identifier for s.
    """
    elements = list(s)
    elements.sort()  # Sort the elements to maintain order
    return "{" + ", ".join("'" + element + "'" for element in elements) + "}"


def get_set(k):
    """
    This method retrieves a set from a unique string identifier.

    Args:
        k (string): Unique string identifier.

    Returns:
        s (set): Corresponding set.
    """
    elements = ast.literal_eval(k)
    return set(elements)
