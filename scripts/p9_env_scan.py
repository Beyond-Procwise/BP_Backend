"""Every os.getenv in src/, with its default and where it is read."""
import ast, pathlib, collections

found = collections.defaultdict(list)
for path in sorted(pathlib.Path("src").rglob("*.py")):
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        continue
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = None
        if isinstance(fn, ast.Attribute) and fn.attr == "getenv":
            name = "getenv"
        elif isinstance(fn, ast.Attribute) and fn.attr == "get" and \
             isinstance(fn.value, ast.Attribute) and fn.value.attr == "environ":
            name = "environ.get"
        if not name or not node.args:
            continue
        key = node.args[0]
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            continue
        default = ""
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            default = repr(node.args[1].value)
        found[key.value].append((str(path), node.lineno, default))

print(f"distinct environment variables read in src/: {len(found)}\n")
for key in sorted(found):
    sites = found[key]
    p, ln, d = sites[0]
    extra = f"  (+{len(sites)-1} more)" if len(sites) > 1 else ""
    print(f"{key:38s} default={d:18s} {p.replace('src/','')}:{ln}{extra}")
