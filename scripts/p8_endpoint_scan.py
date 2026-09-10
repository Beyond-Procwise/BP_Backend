"""Every write/send endpoint, and whether it resolves the caller."""
import ast, pathlib, sys

WRITE = {"post", "put", "patch", "delete"}
rows = []

for path in sorted(pathlib.Path("src/api/routers").glob("*.py")):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        methods, routes = [], []
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            fn = dec.func
            if isinstance(fn, ast.Attribute) and fn.attr.lower() in WRITE:
                methods.append(fn.attr.upper())
                routes.append(dec.args[0].value if dec.args and isinstance(dec.args[0], ast.Constant) else "?")
        if not methods:
            continue
        args = node.args
        defaults = list(args.defaults) + list(args.kw_defaults)
        has_principal = False
        for d in defaults:
            if isinstance(d, ast.Call) and getattr(d.func, "id", getattr(d.func, "attr", "")) == "Depends":
                if d.args and getattr(d.args[0], "id", "") == "require_user":
                    has_principal = True
        # does the body call the gate?
        src = ast.get_source_segment(path.read_text(), node) or ""
        gated = "gate(" in src
        rows.append((path.name, methods[0], routes[0], node.name, has_principal, gated))

unauth = [r for r in rows if not r[4]]
print(f"write/send endpoints in src/api/routers: {len(rows)}")
print(f"  with Depends(require_user): {len(rows) - len(unauth)}")
print(f"  WITHOUT:                    {len(unauth)}")
print()
cur = None
for name, method, route, fn, has, gated in sorted(unauth):
    if name != cur:
        cur = name
        print(f"--- {name}")
    print(f"    {method:6s} {route:52s} {fn}")
