"""Runtime-grounded API repair diagnostics for generated code failures.

This module provides programmatic error classification, AST-based code
location, and live runtime introspection to build repair prompts that
are grounded in the actual Python environment rather than static
whitelist/denylist rules.

Supported failure modes:
  - AttributeError (object missing attribute, module missing attribute)
  - TypeError (unexpected keyword argument, wrong argument count)
  - NameError (undefined name)
  - ImportError / ModuleNotFoundError
  - SyntaxError (invalid Python syntax)
  - KeyError / IndexError (failed indexing)
  - ValueError (generic value/shape/unpacking mismatch)

Usage::

    from vesta.core.api_repair_diagnostics import build_api_discovery_report

    report = build_api_discovery_report(
        generated_code=previous_code,
        error_message=last_error,
        runtime_namespace={"pm": pm, "np": np, "plt": plt, "stats": stats},
    )
    # report is a human-readable string suitable for injection into a repair prompt
"""

import ast
import inspect
import re
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = __import__("logging").getLogger("api_repair_diagnostics")

# ---------------------------------------------------------------------------
# 1. Error classification
# ---------------------------------------------------------------------------

ERROR_PATTERNS: Dict[str, re.Pattern] = {
    "attribute_error_object": re.compile(
        r"AttributeError:\s*'([^']+)'\s+object\s+has\s+no\s+attribute\s+'([^']+)'"
    ),
    "attribute_error_module": re.compile(
        r"AttributeError:\s*module\s+'([^']+)'\s+has\s+no\s+attribute\s+'([^']+)'"
    ),
    "type_error_unexpected_kwarg": re.compile(
        r"TypeError:\s*(.*?)\(\)\s+got\s+an\s+unexpected\s+keyword\s+argument\s+'([^']+)'"
    ),
    "type_error_takes_positional": re.compile(
        r"TypeError:\s*(.*?)\(\)\s+takes\s+"
    ),
    "name_error": re.compile(
        r"NameError:\s*name\s+'([^']+)'\s+is\s+not\s+defined"
    ),
    "import_error": re.compile(
        r"(?:ImportError|ModuleNotFoundError):\s*No\s+module\s+named\s+'([^']+)'"
    ),
    "syntax_error": re.compile(r"SyntaxError:\s*(.*)"),
    "key_error": re.compile(r"KeyError:\s*(.*)"),
    "index_error": re.compile(r"IndexError:\s*(.*)"),
    "value_error": re.compile(r"ValueError:\s*(.*)"),
}


def _classify_error(error_message: str) -> Tuple[str, Optional[re.Match]]:
    """Classify an error message and return (category, regex_match)."""
    for category, pattern in ERROR_PATTERNS.items():
        match = pattern.search(error_message)
        if match is not None:
            return category, match
    return "unknown", None


# ---------------------------------------------------------------------------
# 2. Code location (AST-based)
# ---------------------------------------------------------------------------


def _find_attribute_access(
    code: str, attr_name: str
) -> List[Tuple[int, str]]:
    """Find all lines in ``code`` where ``.attr_name`` is accessed."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    results: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == attr_name:
            line_no: int = getattr(node, "lineno", 0)
            line_text: str = code.splitlines()[line_no - 1] if line_no > 0 else ""
            results.append((line_no, line_text.strip()))
    return results


def _find_name_usage(code: str, name: str) -> List[Tuple[int, str]]:
    """Find all lines in ``code`` where ``name`` is used as a Name node."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    results: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            line_no: int = getattr(node, "lineno", 0)
            line_text: str = code.splitlines()[line_no - 1] if line_no > 0 else ""
            results.append((line_no, line_text.strip()))
    return results


def _find_call_with_keyword(code: str, keyword: str) -> List[Tuple[int, str]]:
    """Find all lines in ``code`` where ``keyword=...`` is passed to a call."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    results: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == keyword:
                    line_no: int = getattr(node, "lineno", 0)
                    line_text: str = (
                        code.splitlines()[line_no - 1] if line_no > 0 else ""
                    )
                    results.append((line_no, line_text.strip()))
    return results


# ---------------------------------------------------------------------------
# 3. Runtime introspection helpers
# ---------------------------------------------------------------------------


def _resolve_dotted_name(
    dotted_name: str, namespace: Dict[str, Any]
) -> Optional[Any]:
    """Resolve a dotted name like ``pm.gp.cov.Periodic`` from a namespace dict."""
    parts = dotted_name.split(".")
    if len(parts) == 0:
        return None
    root = parts[0]
    if root not in namespace:
        return None
    obj = namespace[root]
    for part in parts[1:]:
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
    return obj


def _find_object_by_name(
    name: str,
    namespace: Dict[str, Any],
    max_depth: int = 4,
) -> Tuple[Optional[Any], Optional[str]]:
    """Breadth-first search for an object whose ``__name__`` or path suffix matches ``name``.

    Returns ``(obj, dotted_path)`` or ``(None, None)``.
    """
    queue: Deque[Tuple[Any, str]] = deque()
    visited: set[int] = set()

    for key, val in namespace.items():
        if key.startswith("_"):
            continue
        queue.append((val, key))
        visited.add(id(val))

    depth = 0
    while len(queue) > 0 and depth < max_depth:
        next_queue: Deque[Tuple[Any, str]] = deque()
        while len(queue) > 0:
            obj, path = queue.popleft()
            # Check __name__ attribute (works for functions, classes in most cases)
            if getattr(obj, "__name__", None) == name:
                return obj, path
            # Check path suffix — if the last component of the dotted path matches,
            # we've found the object by its exposed name regardless of __name__ quirks.
            if path.rsplit(".", 1)[-1] == name:
                return obj, path
            try:
                for attr_name in dir(obj):
                    if attr_name.startswith("_"):
                        continue
                    try:
                        attr_val = getattr(obj, attr_name)
                        attr_id = id(attr_val)
                        if attr_id in visited:
                            continue
                        visited.add(attr_id)
                        next_queue.append((attr_val, f"{path}.{attr_name}"))
                    except Exception:
                        continue
            except Exception:
                continue
        queue = next_queue
        depth += 1

    return None, None


def _introspect_object(obj: Any, max_members: int = 30) -> str:
    """Return a concise introspection report for a live Python object."""
    lines: List[str] = []
    try:
        members = inspect.getmembers(obj)
    except Exception:
        members = []

    public_members: List[str] = [
        name for name, _ in members if not name.startswith("_")
    ]
    if len(public_members) > max_members:
        public_members = public_members[:max_members]
        public_members.append("...")

    if len(public_members) > 0:
        lines.append(f"  Public members: {', '.join(public_members)}")
    else:
        lines.append("  No public members introspected.")

    # Try to get __init__ signature
    try:
        if inspect.isclass(obj):
            sig = inspect.signature(obj.__init__)
            lines.append(f"  __init__ signature: {sig}")
    except Exception:
        pass

    # Try to get callable signature
    try:
        if callable(obj) and not inspect.isclass(obj):
            sig = inspect.signature(obj)
            lines.append(f"  Signature: {sig}")
    except Exception:
        pass

    return "\n".join(lines)


def _search_related_in_container(
    container_obj: Any, query: str, max_results: int = 5
) -> List[Tuple[str, str]]:
    """Search a container object (module/class) for members whose names contain ``query``."""
    if container_obj is None:
        return []

    try:
        names = dir(container_obj)
    except Exception:
        return []

    query_lower = query.lower()
    matches: List[Tuple[str, str]] = []
    for name in names:
        if query_lower in name.lower() and not name.startswith("_"):
            try:
                member = getattr(container_obj, name)
                sig_str = ""
                try:
                    if callable(member):
                        sig_str = str(inspect.signature(member))
                except Exception:
                    pass
                matches.append((name, sig_str))
            except Exception:
                pass
        if len(matches) >= max_results:
            break

    return matches


def _find_parent_module(obj: Any) -> Optional[Any]:
    """Try to find the module or class that contains ``obj``."""
    if hasattr(obj, "__module__"):
        module_name = obj.__module__
        try:
            import importlib

            return importlib.import_module(module_name)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# 4. Error-specific diagnostic builders
# ---------------------------------------------------------------------------


def _diagnose_attribute_error_object(
    *,
    error_message: str,
    generated_code: str,
    runtime_namespace: Dict[str, Any],
    match: re.Match,
) -> str:
    """Diagnose ``AttributeError: 'X' object has no attribute 'Y'``."""
    object_type_name: str = match.group(1)
    missing_attr: str = match.group(2)

    lines: List[str] = [
        f"Detected: AttributeError on '{object_type_name}' object — missing attribute '{missing_attr}'.",
    ]

    # Find the failing line(s) in generated code
    code_locations = _find_attribute_access(generated_code, missing_attr)
    if len(code_locations) > 0:
        lines.append("Failing code location(s):")
        for line_no, line_text in code_locations:
            lines.append(f"  line {line_no}: {line_text}")
    else:
        lines.append(
            "Could not locate the exact `." + missing_attr + "` access in generated code."
        )

    # Try to resolve the actual runtime object by name via BFS
    resolved_obj: Optional[Any]
    resolved_path: Optional[str]
    resolved_obj, resolved_path = _find_object_by_name(
        name=object_type_name,
        namespace=runtime_namespace,
    )

    if resolved_obj is not None and resolved_path is not None:
        lines.append(f"\nRuntime introspection of {resolved_path}:")
        lines.append(_introspect_object(resolved_obj))

        # Check if missing attribute exists
        has_attr = False
        try:
            has_attr = hasattr(resolved_obj, missing_attr)
        except Exception:
            pass
        if not has_attr:
            lines.append(
                f"\nCONFIRMED: '{missing_attr}' is NOT a public member of {resolved_path}."
            )

        # Search for related members in the same container
        container: Optional[Any] = None
        if "." in resolved_path:
            parent_path: str = resolved_path.rsplit(".", 1)[0]
            container = _resolve_dotted_name(parent_path, runtime_namespace)
        if container is None:
            container = _find_parent_module(resolved_obj)

        if container is not None:
            related = _search_related_in_container(container, missing_attr)
            if len(related) > 0:
                lines.append(
                    f"\nRelated runtime APIs in the same module containing '{missing_attr}':"
                )
                for name, sig in related:
                    lines.append(f"  {name}{sig}")
    else:
        lines.append(
            f"\nCould not resolve a runtime object matching type name '{object_type_name}'."
        )
        lines.append(
            "The generated code may be using an object from a module not available in the sandbox."
        )

    return "\n".join(lines)


def _diagnose_attribute_error_module(
    *,
    error_message: str,
    generated_code: str,
    runtime_namespace: Dict[str, Any],
    match: re.Match,
) -> str:
    """Diagnose ``AttributeError: module 'X' has no attribute 'Y'``."""
    module_name: str = match.group(1)
    missing_attr: str = match.group(2)

    lines: List[str] = [
        f"Detected: AttributeError on module '{module_name}' — missing attribute '{missing_attr}'.",
    ]

    # Find usage in code
    code_locations = _find_attribute_access(generated_code, missing_attr)
    if len(code_locations) > 0:
        lines.append("Failing code location(s):")
        for line_no, line_text in code_locations:
            lines.append(f"  line {line_no}: {line_text}")

    # Resolve the module in namespace — try dotted name, then BFS
    module_obj = _resolve_dotted_name(module_name, runtime_namespace)
    module_path = module_name
    if module_obj is None:
        # BFS fallback: search for any object matching the last component
        short_name = module_name.rsplit(".", 1)[-1]
        module_obj, module_path = _find_object_by_name(
            name=short_name,
            namespace=runtime_namespace,
        )

    if module_obj is not None and module_path is not None:
        lines.append(f"\nRuntime introspection of {module_path}:")
        try:
            public_names = [n for n in dir(module_obj) if not n.startswith("_")]
            if len(public_names) > 30:
                public_names = public_names[:30] + ["..."]
            lines.append(f"  Public members: {', '.join(public_names)}")
        except Exception as e:
            lines.append(f"  Could not introspect module: {e}")

        lines.append(
            f"\nCONFIRMED: '{missing_attr}' is NOT a public member of {module_path}."
        )

        # Search for related
        related = _search_related_in_container(module_obj, missing_attr)
        if len(related) > 0:
            lines.append(
                f"\nRelated runtime APIs in {module_path} containing '{missing_attr}':"
            )
            for name, sig in related:
                lines.append(f"  {name}{sig}")
    else:
        lines.append(
            f"\nModule '{module_name}' is not importable in the current sandbox."
        )
        lines.append(
            "The generated code may be importing a module that is not installed."
        )

    return "\n".join(lines)


def _diagnose_type_error_unexpected_kwarg(
    *,
    error_message: str,
    generated_code: str,
    runtime_namespace: Dict[str, Any],
    match: re.Match,
) -> str:
    """Diagnose ``TypeError: X() got an unexpected keyword argument 'Y'``."""
    callable_desc: str = match.group(1).strip()
    bad_kwarg: str = match.group(2)

    lines: List[str] = [
        f"Detected: TypeError — unexpected keyword argument '{bad_kwarg}' "
        f"passed to {callable_desc}().",
    ]

    # Find the call in generated code
    code_locations = _find_call_with_keyword(generated_code, bad_kwarg)
    if len(code_locations) > 0:
        lines.append("Failing code location(s):")
        for line_no, line_text in code_locations:
            lines.append(f"  line {line_no}: {line_text}")
    else:
        lines.append(
            f"Could not locate the exact call with `{bad_kwarg}=...` in generated code."
        )

    # Try to resolve the callable and get its actual signature
    # The callable_desc may be dotted (e.g. "Axes.stem") or simple (e.g. "stem")
    resolved_callable: Optional[Any] = None
    resolved_path: Optional[str] = None

    # Strategy 1: dotted name resolution (e.g., "np.mean")
    resolved_callable = _resolve_dotted_name(callable_desc, runtime_namespace)
    if resolved_callable is not None:
        resolved_path = callable_desc
    else:
        # Strategy 2: search namespace roots for direct attribute match
        for key, val in runtime_namespace.items():
            if hasattr(val, callable_desc):
                try:
                    resolved_callable = getattr(val, callable_desc)
                    resolved_path = f"{key}.{callable_desc}"
                    break
                except Exception:
                    continue

    if resolved_callable is None and "." in callable_desc:
        # Strategy 3: for "Axes.stem", find the class by BFS then grab the method
        class_name, method_name = callable_desc.rsplit(".", 1)
        resolved_class: Optional[Any]
        class_path: Optional[str]
        resolved_class, class_path = _find_object_by_name(
            name=class_name,
            namespace=runtime_namespace,
        )
        if resolved_class is not None and class_path is not None:
            try:
                resolved_callable = getattr(resolved_class, method_name)
                resolved_path = f"{class_path}.{method_name}"
            except AttributeError:
                pass

    if resolved_callable is None and "." not in callable_desc:
        # Strategy 4: BFS for the callable name directly
        resolved_callable, resolved_path = _find_object_by_name(
            name=callable_desc,
            namespace=runtime_namespace,
        )

    if resolved_callable is not None:
        lines.append(f"\nRuntime introspection of {resolved_path}:")
        try:
            sig = inspect.signature(resolved_callable)
            lines.append(f"  Actual signature: {sig}")
        except Exception as e:
            lines.append(f"  Could not get signature: {e}")

        lines.append(
            f"\nCONFIRMED: parameter '{bad_kwarg}' is NOT accepted by {resolved_path} "
            f"in this installed version."
        )
    else:
        lines.append(
            f"\nCould not resolve callable '{callable_desc}' in the runtime namespace."
        )

    return "\n".join(lines)


def _diagnose_name_error(
    *,
    error_message: str,
    generated_code: str,
    runtime_namespace: Dict[str, Any],
    match: re.Match,
) -> str:
    """Diagnose ``NameError: name 'X' is not defined``."""
    missing_name: str = match.group(1)

    lines: List[str] = [
        f"Detected: NameError — name '{missing_name}' is not defined at runtime.",
    ]

    # Find usage in code
    code_locations = _find_name_usage(generated_code, missing_name)
    if len(code_locations) > 0:
        lines.append("Failing code location(s):")
        for line_no, line_text in code_locations:
            lines.append(f"  line {line_no}: {line_text}")
    else:
        lines.append(
            f"Could not locate usage of '{missing_name}' in generated code."
        )

    # Check if name exists in namespace
    if missing_name in runtime_namespace:
        lines.append(
            f"\nNOTE: '{missing_name}' IS available in the sandbox namespace, "
            f"but the generated code may be using it before it is defined or in a different scope."
        )
    else:
        lines.append(
            f"\n'{missing_name}' is NOT available in the sandbox namespace."
        )
        # Suggest similar names
        all_names: List[str] = []
        for val in runtime_namespace.values():
            try:
                all_names.extend(
                    n for n in dir(val) if not n.startswith("_")
                )
            except Exception:
                pass
        all_names.extend(runtime_namespace.keys())

        similar = [
            n for n in set(all_names) if missing_name.lower() in n.lower() or n.lower() in missing_name.lower()
        ][:5]
        if len(similar) > 0:
            lines.append(f"Similar names in namespace: {', '.join(similar)}")

    return "\n".join(lines)


def _diagnose_import_error(
    *,
    error_message: str,
    generated_code: str,
    runtime_namespace: Dict[str, Any],
    match: re.Match,
) -> str:
    """Diagnose ``ImportError: No module named 'X'``."""
    module_name: str = match.group(1)

    lines: List[str] = [
        f"Detected: ImportError — module '{module_name}' is not installed.",
    ]

    lines.append(
        "The sandbox does not allow arbitrary imports. Use only libraries already in scope."
    )
    lines.append(
        f"Available libraries in namespace: {', '.join(sorted(runtime_namespace.keys()))}"
    )

    return "\n".join(lines)


def _diagnose_generic_python_error(
    *,
    category: str,
    error_message: str,
    generated_code: str,
    runtime_namespace: Dict[str, Any],
    match: Optional[re.Match],
) -> str:
    """Diagnose generic Python failures without library-specific advice."""
    error_lines: List[str] = error_message.splitlines()
    first_error_line: str = error_lines[0] if len(error_lines) > 0 else error_message
    lines: List[str] = [
        f"Detected: {category}.",
        "This is a generic Python runtime or syntax failure, not a library-specific API diagnosis.",
        "Use the full traceback below and the generated code to identify the exact failing line and value flow.",
        f"Error summary: {first_error_line}",
    ]
    if match is not None and len(match.groups()) > 0:
        lines.append(f"Matched detail: {match.group(1)}")
    lines.append(f"Available runtime names: {', '.join(sorted(runtime_namespace.keys()))}")
    if len(generated_code) > 0:
        lines.append("Generated code is available in the repair prompt immediately below this diagnostics block.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Public API
# ---------------------------------------------------------------------------


def build_api_discovery_report(
    *,
    generated_code: str,
    error_message: str,
    runtime_namespace: Dict[str, Any],
) -> str:
    """Build a runtime-grounded API discovery report for a code-generation failure.

    This function is the entry point for the repair diagnostics system.
    It classifies the error, locates the failing code, introspects the
    live Python runtime, and returns a human-readable report suitable
    for injection into a repair prompt.

    Args:
        generated_code: The generated Python code that failed.
        error_message: The exception message (often from ``format_exception_msg``).
        runtime_namespace: The dict of objects available in the execution
            sandbox (e.g. ``{"pm": pm, "np": np, "plt": plt}``).

    Returns:
        A formatted string report. Empty string if the error type is not
        recognized (falls back to generic repair).
    """
    category, match = _classify_error(error_message)

    if category == "attribute_error_object" and match is not None:
        return _diagnose_attribute_error_object(
            error_message=error_message,
            generated_code=generated_code,
            runtime_namespace=runtime_namespace,
            match=match,
        )
    elif category == "attribute_error_module" and match is not None:
        return _diagnose_attribute_error_module(
            error_message=error_message,
            generated_code=generated_code,
            runtime_namespace=runtime_namespace,
            match=match,
        )
    elif category == "type_error_unexpected_kwarg" and match is not None:
        return _diagnose_type_error_unexpected_kwarg(
            error_message=error_message,
            generated_code=generated_code,
            runtime_namespace=runtime_namespace,
            match=match,
        )
    elif category == "name_error" and match is not None:
        return _diagnose_name_error(
            error_message=error_message,
            generated_code=generated_code,
            runtime_namespace=runtime_namespace,
            match=match,
        )
    elif category == "import_error" and match is not None:
        return _diagnose_import_error(
            error_message=error_message,
            generated_code=generated_code,
            runtime_namespace=runtime_namespace,
            match=match,
        )
    elif category in ("syntax_error", "key_error", "index_error", "value_error"):
        return _diagnose_generic_python_error(
            category=category,
            error_message=error_message,
            generated_code=generated_code,
            runtime_namespace=runtime_namespace,
            match=match,
        )
    else:
        return _diagnose_generic_python_error(
            category="unknown_error",
            error_message=error_message,
            generated_code=generated_code,
            runtime_namespace=runtime_namespace,
            match=match,
        )


def build_grounded_repair_prompt(
    *,
    base_prompt: str,
    previous_code: str,
    error_message: str,
    runtime_namespace: Dict[str, Any],
) -> str:
    """Build a full repair prompt with runtime-grounded API discovery injected.

    This is a convenience wrapper around :func:`build_api_discovery_report`
    that formats the report into a complete repair prompt string.

    Args:
        base_prompt: The original code-generation prompt (preserves context).
        previous_code: The generated code that failed.
        error_message: The exception message from execution.
        runtime_namespace: The sandbox namespace for runtime introspection.

    Returns:
        A complete repair prompt string ready to send to the LLM.
    """
    discovery_report: str = build_api_discovery_report(
        generated_code=previous_code,
        error_message=error_message,
        runtime_namespace=runtime_namespace,
    )

    return (
        f"{base_prompt}\n\n"
        f"The previous generated code failed during execution.\n\n"
        f"{'═' * 60}\n"
        f"RUNTIME API DISCOVERY (from the live Python environment)\n"
        f"{'═' * 60}\n"
        f"{discovery_report}\n"
        f"{'═' * 60}\n\n"
        f"Previous code:\n```python\n{previous_code}\n```\n\n"
        f"Execution error:\n{error_message}\n\n"
        f"Fix the code while preserving the original modeling intent. "
        f"Use ONLY APIs verified by the runtime discovery above."
    )
