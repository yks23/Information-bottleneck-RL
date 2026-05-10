import ast


def countdown_binary_reward(numbers, target, expression):
    """Conservative verifier: returns 1 only when parsing and exact match succeed."""
    allowed = set(numbers)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return 0

    used = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            used.append(int(node.value))
        elif isinstance(node, ast.Name):
            return 0

    for u in used:
        if u not in allowed:
            return 0

    try:
        value = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return 0
    return 1 if value == target else 0
