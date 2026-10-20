
## A) Scope & Audience

This guide is for **solo developers and small-to-medium teams** writing modern Python 3 (3.10+), whether producing quick scripts, internal tools, or distributable packages. It synthesizes PEPs, practitioner books, and tool conventions from the attached reference into a single, internally consistent standard. Where the reference presents multiple valid approaches, this guide resolves them into one default with documented exceptions.[^1]

**Compatibility assumption:** Python ≥ 3.10. All examples use modern syntax (structural pattern matching, `X | Y` union types, etc.) where applicable.

***

## B) Executive Summary

### Non-Negotiables

These 15 rules are the highest-signal items distilled from every section of the reference. Violating any of these should require an explicit, documented reason.

1. **4-space indentation, no tabs.**[^1]
2. **Autoformat all code with Black (or Ruff formatter).** Never debate formatting in review.[^1]
3. **Sort imports automatically** (Ruff/isort: stdlib → third-party → local, blank line between groups).[^1]
4. **Type-hint all public function signatures.** Treat type hints as design documentation.[^1]
5. **Write docstrings for every public module, class, and function** (PEP 257: triple double-quotes, one-line summary, blank line, body).[^1]
6. **Use `enumerate`, `zip`, `reversed`, comprehensions** — never manual index manipulation.[^1]
7. **Single Responsibility:** each function/class has one reason to change.[^1]
8. **Test with pytest.** Use fixtures to separate setup from assertions. Parametrize for multiple scenarios.[^1]
9. **Run linting, type-checking, and security scanning in CI.** Fail the build on violations.[^1]
10. **Implement `__repr__` on every non-trivial class.**[^1]
11. **Prefer composition over inheritance.**[^1]
12. **Use context managers (`with`) for all resource management.**[^1]
13. **Explicit is better than implicit** — no magic globals, no side-effect-heavy imports.[^1]
14. **Install pre-commit hooks** enforcing formatter + linter + type checker on every commit.[^1]
15. **Make small, safe changes iteratively rather than large rewrites** (Fowler's refactoring principle).[^1]

### House Defaults

If you adopt only one standard: **install Ruff (with its formatter in Black-compatible mode) + mypy, configure pre-commit, and follow PEP 8.** This single stack covers formatting, import sorting, linting, and type checking with near-zero configuration.[^1]

***

## C) Philosophy ("Pythonic")

The reference anchors Pythonic philosophy in PEP 20 (The Zen of Python) and then operationalizes it through books like *Effective Python*, *Fluent Python*, and *Robust Python*. Here is what this guide optimizes for, in priority order:[^1]

1. **Readability** — "Code is read 10x more than written. Optimize for the reader, not the writer."[^1]
2. **Explicitness** — "Explicit is better than implicit." Favor named tuples, type hints, and well-named variables over clever one-liners.[^1]
3. **Consistency** — Follow PEP 8 ruthlessly; delegate formatting to tools. Consistent style reduces cognitive load across teams.[^1]
4. **Safe change** — Comprehensive tests + type hints + small refactorings mean you can evolve code confidently.[^1]
5. **Maintainability** — SOLID principles, clear module boundaries, and dependency inversion keep codebases healthy as they grow.[^1]

> **Source note:** Attachment section "Distilled Principles of Elegant Python" — principles 1-3 (Style & Readability) and 10-12 (Type Safety & Quality).

***

## D) The Style Guide


***

### D.1) Layout & Formatting

#### Rule D.1.1 — Use 4 spaces per indentation level

**Rationale:** PEP 8's universal standard; every major tool assumes it.[^1]

```python
# ✅ Good
def calculate_total(items: list[float]) -> float:
    subtotal = sum(items)
    tax = subtotal * 0.08
    return subtotal + tax
```

```python
# ❌ Bad
def calculate_total(items: list[float]) -> float:
   subtotal = sum(items)  # 3-space indent
   tax = subtotal * 0.08
   return subtotal + tax
```

**Exceptions:** Google's internal guide uses 2 spaces — this is acknowledged in the reference but **not recommended** as the default.[^1]

> **Source note:** Section A, PEP 8 — "Use 4 spaces for indentation."

#### Rule D.1.2 — Maximum line length: 88 characters (Black default)

**Rationale:** PEP 8 says 79; Black defaults to 88 as a practical compromise that reduces forced line breaks without sacrificing readability. Since we delegate formatting to Black/Ruff, 88 is the resolved default.[^1]

```python
# ✅ Good — fits within 88 chars naturally
result = compute_shipping(weight=package.weight, destination=address.zip_code)
```

```python
# ❌ Bad — unnecessary manual wrapping at 79
result = compute_shipping(
    weight=package.weight,
    destination=address.zip_code,
)  # forced break when the line is only 82 chars
```

**Team choice:** Teams may choose 79 (strict PEP 8) or 120 (wide monitors). Document it in `pyproject.toml`.

> **Source note:** Section A, PEP 8 — "maximum line length 79 characters" resolved against Section E, Black — "minimal configuration; widely adopted."

#### Rule D.1.3 — Let Black/Ruff handle all whitespace, trailing commas, and quote style

**Rationale:** "Standardize formatting to eliminate style debates."[^1]

**Exceptions:** None. Do not override Black's decisions except via its supported configuration keys.

> **Source note:** Section E, Black — "any color you like."

#### Rule D.1.4 — Specific Docstring Formatting (Google style)

**Rationale:** Standardize opinionated docstring formatting beyond PEP 257 by adopting Google-style section headers. Google style is widely used in the Python ecosystem, well-supported by tooling (Sphinx via `sphinx.ext.napoleon`, mkdocstrings, IDE renderers), and reads cleanly as plain prose.

**Section headers must be capitalized words followed by a colon** (`Args:`, `Returns:`, `Raises:`, `Attributes:`, `Yields:`, `Examples:`). Do not use the `:args:` / `:returns:` ReST/Sphinx field syntax. Body items under each header are indented one level (4 spaces) and follow the pattern `name: description`.

```python
# ✅ Good:
#   - One-line summary at the top, blank line, then optional further description.
#   - Google-style section headers with a trailing colon.
#   - Documents each pertinent argument.
#   - Omits a Returns: section when the return value is obvious from the type annotation.

def my_cool_function(my_cool_name: str) -> None:
    """This is my really cool function.

    Args:
        my_cool_name: String of my really cool name.
    """
    print(f"Wow! Super Cool Function {my_cool_name}")
```

```python
# ✅ Good:
#   - Adds further description for a more complicated function.
#   - Documents a non-obvious return value beyond its type.

def my_cooler_complex_function(my_cool_name: str) -> dict[str, str]:
    """This is my really cool function.

    It builds and prints a small dictionary keyed off the caller's name and
    returns the dictionary for further use.

    Args:
        my_cool_name: String of my really cool name.

    Returns:
        A two-entry dictionary mapping ``"key_1"`` and ``"key_2"`` to formatted
        string values derived from ``my_cool_name``.
    """
    super_cool_dict = {
        "key_1": f"val_1_{my_cool_name}",
        "key_2": f"val_2_{my_cool_name}",
    }
    print(f"Wow! Super Cool Function {my_cool_name}")
    print(f"Awesome Dictionary Too: {super_cool_dict}")
    return super_cool_dict
```

```python
# ❌ Bad:
#   - Uses ReST/Sphinx :args: / :returns: field syntax instead of Google headers.
#   - Documents a Returns: that is already obvious from the function signature.

def my_cool_function(my_cool_name: str) -> None:
    """This is my really cool function.

    :args:
        my_cool_name: String of my really cool name.

    :returns:
        Nothing.
    """
    print("Wow! Super Cool Function")
```

**Structure of a Completely Filled Out Acceptable Docstring:**

```python
"""[One-liner Function Description]

[Further description of function]

Args:
    [arg_item_name]: [arg_item_desc]

Returns:
    [Return Value Description]

Raises:
    [error_type]: [error_meaning_desc]

Attributes:
    [Description of Pertinent Attribute 1]
    [Description of Pertinent Attribute 2]

Examples:
    >>> call_function(args)
    [Return Value]
"""
```

**Exceptions:** Omit the `Returns:` section when the function's return value is fully described by its type annotation and the one-line summary (e.g. simple getters, predicates, and parsers). Always document non-trivial `Args:` and any `Raises:` on public functions.

> **Source note:** Google Python Style Guide — "Docstrings" section. Compatible with `sphinx.ext.napoleon` for documentation generation.

***

### D.2) Naming Conventions

| Element | Convention | Example |
| :-- | :-- | :-- |
| Modules / packages | `snake_case`, short, no hyphens | `data_loader`, `utils` |
| Classes | `PascalCase` | `ShippingCalculator` |
| Functions / methods | `snake_case` | `get_user_by_id()` |
| Variables | `snake_case` | `total_price` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_RETRIES = 3` |
| Private | Leading underscore | `_internal_cache` |
| "Truly private" (name-mangled) | Double leading underscore | `__secret` (rarely needed) |

[^1]

#### Rule D.2.1 — Descriptive names beat short names

**Rationale:** "Readability counts."[^1]

```python
# ✅ Good
remaining_attempts = MAX_RETRIES - failed_count
```

```python
# ❌ Bad
ra = MR - fc
```

**Exceptions:** Short names are fine for trivial loop variables (`i`, `x`) and well-known mathematical conventions.

> **Source note:** Section A, PEP 8 (naming conventions) + Distilled Principle \#3 — "Descriptive names beat short names."

***

### D.3) Imports & Module Organization

#### Rule D.3.1 — Three groups, blank-line separated, alphabetically sorted within each group

```python
# ✅ Good
import os
import sys
from pathlib import Path

import httpx
from pydantic import BaseModel

from myproject.core import config
from myproject.models import User
```

**Rationale:** isort/Ruff enforces stdlib → third-party → local grouping automatically.[^1]

```python
# ❌ Bad
from myproject.models import User
import sys
import httpx
from pathlib import Path
```


#### Rule D.3.2 — No wildcard imports

```python
# ❌ Bad
from os.path import *
```

**Rationale:** Wildcard imports pollute the namespace and violate "explicit is better than implicit."[^1]

**Exceptions:** `__init__.py` files that intentionally re-export a public API may use `__all__` + wildcard imports internally, but prefer explicit re-exports.

> **Source note:** Section E, isort — "Group imports: stdlib, third-party, local, with blank lines between."

***

### D.4) Functions & Control Flow

#### Rule D.4.1 — Prefer early returns (guard clauses) over deep nesting

```python
# ✅ Good
def process_order(order: Order) -> Receipt:
    if not order.items:
        raise EmptyOrderError("Order has no items")
    if order.is_cancelled:
        raise CancelledOrderError(order.id)
    return _build_receipt(order)
```

```python
# ❌ Bad
def process_order(order: Order) -> Receipt:
    if order.items:
        if not order.is_cancelled:
            return _build_receipt(order)
        else:
            raise CancelledOrderError(order.id)
    else:
        raise EmptyOrderError("Order has no items")
```

**Rationale:** Flat is better than nested (PEP 20). Guard clauses make the "happy path" obvious.[^1]

#### Rule D.4.2 — Use built-in iteration idioms

```python
# ✅ Good
for i, item in enumerate(items):
    print(f"{i}: {item}")

names = [user.name for user in users if user.active]
```

```python
# ❌ Bad
for i in range(len(items)):
    print(f"{i}: {items[i]}")

names = list(map(lambda u: u.name, filter(lambda u: u.active, users)))
```

**Rationale:** "Prefer `enumerate` over manual indexing; use comprehensions over `map`/`filter`."[^1]

#### Rule D.4.3 — Avoid cleverness; favor clarity

A comprehension that requires more than ~2 conditions or nested loops should become a regular loop or a helper function. Readability always wins.

> **Source note:** Section B, *Effective Python* — "Prefer `enumerate` over manual indexing; use comprehensions over `map`/`filter`"; *Writing Idiomatic Python* — "Replace index manipulation with `enumerate`, `zip`, and `reversed`."

***

### D.5) Data Modeling

#### Rule D.5.1 — Default to `dataclasses` (or `NamedTuple`) for plain data

```python
# ✅ Good
from dataclasses import dataclass

@dataclass(frozen=True)
class Coordinate:
    lat: float
    lon: float
```

```python
# ❌ Bad — manual boilerplate
class Coordinate:
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon
    def __eq__(self, other): ...
    def __repr__(self): ...
```

**Rationale:** Dataclasses generate `__init__`, `__repr__`, `__eq__` automatically. Use `frozen=True` for immutability by default.[^1]

#### Rule D.5.2 — Implement `__repr__` for debugging; `__str__` for user display

```python
# ✅ Good
class Money:
    def __repr__(self) -> str:
        return f"Money(amount={self.amount!r}, currency={self.currency!r})"

    def __str__(self) -> str:
        return f"{self.currency} {self.amount:.2f}"
```

**Rationale:** "Implement `__repr__` for debugging, `__str__` for user display." Making custom objects behave like native Python types is a core Pythonic pattern.[^1]

> **Source note:** Section A, Python Data Model — "`__repr__` for debugging, `__str__` for user display"; Section B, *Fluent Python* — "Master special methods to make custom objects behave like built-ins."

***

### D.6) Errors & Logging

#### Rule D.6.1 — Raise exceptions for exceptional conditions; return values for expected outcomes

```python
# ✅ Good — expected outcome
def find_user(user_id: int) -> User | None:
    return db.query(User).filter_by(id=user_id).first()

# ✅ Good — exceptional condition
def withdraw(account: Account, amount: Decimal) -> None:
    if amount > account.balance:
        raise InsufficientFundsError(account.id, amount, account.balance)
    account.balance -= amount
```

```python
# ❌ Bad — returning error codes
def withdraw(account, amount):
    if amount > account.balance:
        return -1  # What does -1 mean?
    account.balance -= amount
    return 0
```


#### Rule D.6.2 — Define domain-specific exception types

```python
class OrderError(Exception):
    """Base for order-domain errors."""

class EmptyOrderError(OrderError): ...
class CancelledOrderError(OrderError): ...
```

**Rationale:** Typed exceptions enable callers to catch specific failure modes and keep business logic independent of infrastructure (Clean Architecture principle).[^1]

> **Source note:** Section C, *Clean Architecture in Python* — "Keep business logic independent of frameworks and databases"; Distilled Principle \#1 — "Explicit beats implicit."

***

### D.7) Types & Type Hints

#### Rule D.7.1 — Type-hint all public APIs; internal code is encouraged but optional

```python
# ✅ Good
def calculate_discount(price: float, rate: float) -> float:
    return price * rate
```

```python
# ❌ Bad — public function with no hints
def calculate_discount(price, rate):
    return price * rate
```

**Rationale:** "Use type hints not just for checking, but as design documentation."[^1]

#### Rule D.7.2 — Gradual adoption strategy

1. **Start with public function signatures** and module boundaries.
2. **Expand inward** to internal functions as code is touched.
3. **Run mypy in CI** from day one — even with incomplete coverage, it catches regressions.
4. **Target `--strict` mode** on new modules; apply retroactively as bandwidth allows.

**Team choice:** Some teams prefer pyright over mypy for speed. Both are valid; pick one and enforce it consistently.

> **Source note:** Section E, mypy — "Type check gradually: start with entry points, expand inward"; Section B, *Robust Python* — "Use type hints not just for checking, but as design documentation."

***

### D.8) Testing

#### Rule D.8.1 — Use pytest as the test framework

**Rationale:** Pytest's assertion introspection, fixtures, and parametrize eliminate boilerplate. The reference identifies *Python Testing with pytest* as the definitive guide.[^1]

#### Rule D.8.2 — Separate setup from assertions with fixtures

```python
# ✅ Good
import pytest

@pytest.fixture
def sample_order() -> Order:
    return Order(items=[Item("Widget", 9.99)], customer_id=42)

def test_order_total(sample_order: Order) -> None:
    assert sample_order.total == pytest.approx(9.99)
```

```python
# ❌ Bad — setup mixed into test
def test_order_total():
    order = Order(items=[Item("Widget", 9.99)], customer_id=42)
    assert order.total == pytest.approx(9.99)
    # Okay for trivial cases, but doesn't scale
```


#### Rule D.8.3 — Parametrize for multiple scenarios

```python
@pytest.mark.parametrize("input_val,expected", [
    (0, "zero"),
    (1, "one"),
    (-1, "negative"),
])
def test_classify_number(input_val: int, expected: str) -> None:
    assert classify(input_val) == expected
```

**Rationale:** "Parametrize tests to run multiple scenarios with minimal code."[^1]

#### Rule D.8.4 — Tests must be deterministic

No dependence on wall-clock time, network, filesystem state, or execution order. Use fixtures and mocking to isolate tests.

> **Source note:** Section D, *Python Testing with pytest* — "Use fixtures to separate test setup from test logic"; pytest docs — "Parametrize tests to run multiple scenarios with minimal code."

***

### D.9) Project Structure & Packaging

#### Rule D.9.1 — Use the `src` layout

```
my-project/
├── pyproject.toml
├── README.md
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── core.py
│       └── models.py
└── tests/
    ├── conftest.py
    ├── test_core.py
    └── test_models.py
```

**Rationale:** "Structure projects with clear separation of concerns from the start." The `src` layout prevents accidental imports from the project root.[^1]

#### Rule D.9.2 — Use `pyproject.toml` as the single source of project metadata

Consolidate tool configuration (Black/Ruff, mypy, pytest) in `pyproject.toml` wherever supported.

#### Rule D.9.3 — Document team-specific conventions in `CONTRIBUTING.md`

**Rationale:** The reference recommends this for teams adopting shared tooling.[^1]

> **Source note:** Section B, *The Hitchhiker's Guide to Python* — "Structure projects with clear separation of concerns from the start"; Practical Application Strategy — "Document team-specific conventions in `CONTRIBUTING.md`."

***

### D.10) Security & Quality Checks

#### Rule D.10.1 — Run bandit in CI for security scanning

**Rationale:** "Identifies common security issues (SQL injection, hardcoded passwords, etc.)."[^1]

#### Rule D.10.2 — Automate everything; fail the build on violations

The reference is explicit: "Fail CI builds on quality violations to maintain standards."[^1]


| Check | Tool | Blocks merge? |
| :-- | :-- | :-- |
| Formatting | Ruff (formatter) | Yes |
| Linting | Ruff (linter) | Yes |
| Type checking | mypy | Yes |
| Security | bandit | Yes |
| Tests | pytest | Yes |

> **Source note:** Distilled Principle \#12 — "Configure pre-commit hooks with Black, Ruff/flake8, mypy, and bandit. Fail CI builds on quality violations."

***

## E) Tooling & Automation

### Recommended Stack

The reference maps a clear tool ecosystem. Here is the resolved, non-redundant stack:[^1]


| Role | Tool | Why this one |
| :-- | :-- | :-- |
| **Formatter** | **Ruff** (formatter mode) | Black-compatible, 10-100x faster, Rust-based [^1] |
| **Linter** | **Ruff** (linter mode) | Replaces Flake8 + isort + dozens of plugins in one binary [^1] |
| **Type checker** | **mypy** | Reference implementation of PEP 484; widest ecosystem [^1] |
| **Security scanner** | **bandit** | PyCQA-maintained; catches hardcoded secrets, SQL injection, etc. [^1] |
| **Git hooks** | **pre-commit** | "Install hooks once; enforce quality automatically" [^1] |

**Team choice:** pylint is a valid addition for teams wanting deeper static analysis (inference-based checking), but it is more verbose. Use it as a complement, not a replacement.[^1]

### Starter `pyproject.toml` Configuration

```toml
[tool.ruff]
target-version = "py310"
line-length = 88

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "S",    # flake8-bandit
    "C4",   # flake8-comprehensions
    "RUF",  # ruff-specific rules
]

[tool.ruff.lint.isort]
known-first-party = ["my_project"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"

[tool.bandit]
exclude_dirs = ["tests"]
```


### Starter `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.13.0
    hooks:
      - id: mypy
        additional_dependencies: []

  - repo: https://github.com/PyCQA/bandit
    rev: 1.8.0
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
```

> **Source note:** Section E — Black, Ruff, mypy, bandit, pre-commit entries; Practical Application Strategy — "Adopt Black + Ruff to eliminate style discussions. Establish pre-commit hooks."

***

## F) Practical Checklists

### "Before You Open a PR" Checklist

- [ ] All pre-commit hooks pass locally (`pre-commit run --all-files`)
- [ ] `ruff check .` and `ruff format --check .` return clean
- [ ] `mypy .` returns clean (or only pre-existing errors on a tracked baseline)
- [ ] `pytest` passes with no new failures
- [ ] New public functions/classes have type hints and docstrings
- [ ] No hardcoded secrets, tokens, or passwords (bandit clean)
- [ ] Commit messages are concise and describe *why*, not just *what*
- [ ] Branch is rebased on latest `main`


### PR Review Checklist

1. **Formatting:** Does it pass Ruff/Black? (Should be automatic — if not, reject.)[^1]
2. **Naming:** Are names descriptive and consistent with project conventions?[^1]
3. **Type hints:** Are public signatures fully typed? Are new types sensible?[^1]
4. **Docstrings:** Do public APIs have PEP 257-compliant docstrings?[^1]
5. **Single Responsibility:** Does each new function/class do one thing?[^1]
6. **Guard clauses:** Are errors handled early? Is the happy path clear?[^1]
7. **Pythonic idioms:** Uses `enumerate`/`zip`/comprehensions instead of manual loops?[^1]
8. **Data modeling:** Uses dataclasses/NamedTuple where appropriate? Has `__repr__`?[^1]
9. **Tests:** Are new behaviors tested? Are tests deterministic and parametrized?[^1]
10. **Fixtures:** Is test setup separated from assertions?[^1]
11. **No wildcard imports:** All imports explicit?[^1]
12. **Context managers:** Resources opened with `with`?[^1]
13. **Composition over inheritance:** No unnecessary class hierarchies?[^1]
14. **Security:** No hardcoded credentials? No unsafe `eval`/`exec`?[^1]
15. **Error handling:** Domain-specific exceptions, not bare `except`?[^1]
16. **Small changes:** Is this PR focused? Could it be split?[^1]
17. **No dead code:** No commented-out blocks or unused imports?[^1]
18. **Dependency hygiene:** Are new dependencies justified and pinned?

### Common Anti-Patterns & Preferred Alternatives

| Anti-Pattern | Why It's Bad | Do This Instead |
| :-- | :-- | :-- |
| `for i in range(len(lst))` | Manual indexing is error-prone and unreadable | `for i, val in enumerate(lst)` [^1] |
| `map(lambda …)` / `filter(lambda …)` | Less readable than comprehensions in Python | List/dict/set comprehension [^1] |
| Bare `except:` or `except Exception:` | Swallows errors silently, hides bugs | Catch specific exception types [^1] |
| Deep inheritance hierarchies | Fragile, hard to reason about | Composition + protocols [^1] |
| Mutable default arguments (`def f(x=[])`) | Shared state across calls | `def f(x: list | None = None)` then `x = x or []` |
| God class / god function | Violates Single Responsibility | Split into focused units [^1] |
| `from module import *` | Namespace pollution, implicit dependencies | Explicit imports [^1] |
| No `__repr__` on custom classes | Debugging is painful | Always implement `__repr__` [^1] |
| Manual resource management (`f = open(…)`) | Leaks on exception | `with open(…) as f:` [^1] |
| Returning error codes (`return -1`) | Ambiguous, not self-documenting | Raise typed exceptions [^1] |
| Large, monolithic PRs | Hard to review, risky to merge | Small, incremental refactorings [^1] |
| Skipping type hints on public APIs | Callers must read implementation to understand contract | Always type-hint public signatures [^1] |
