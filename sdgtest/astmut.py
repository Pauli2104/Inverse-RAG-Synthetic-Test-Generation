"""Automatic first-order mutants of the system under test (independent of the test generator).

Classic mutation operators (as in MuJava/MutPy/mutmut) are applied to one AST node at a time in
``sdgtest/sut.py``. Code that only implements the hand-designed fault switches (``self.m``) and the
MUTANTS dictionary is excluded, so every generated mutant changes the reference behaviour itself.

Operators
  AOR  arithmetic operator replacement           (+ <-> -, * <-> /)
  ROR  relational operator replacement           (< <-> <=, > <-> >=, == <-> !=, < <-> >)
  LCR  logical connector replacement             (and <-> or)
  UOD  unary 'not' deletion
  CRP  constant replacement                      (int n -> n+1 and n -> 0; True <-> False; str s -> '')
  SDL  statement deletion                        (statement -> pass)
  RVR  return value replacement                  (return e -> return None)
  CND  condition negation                        (if c -> if not c)
"""
from __future__ import annotations

import ast
import copy
import random
import signal
import types
from dataclasses import dataclass

import sdgtest.sut as sut_mod

SRC_PATH = sut_mod.__file__


@dataclass
class AstMutant:
    mid: str
    op: str
    line: int
    desc: str
    source: str


def _is_switch(node) -> bool:
    """True when a node references the hand-designed fault switch (self.m) or the MUTANTS dict."""
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr == "m" and isinstance(n.value, ast.Name) and n.value.id == "self":
            return True
        if isinstance(n, ast.Name) and n.id == "MUTANTS":
            return True
    return False


def _excluded_lines(tree):
    """Line ranges of statements that implement or define the hand-designed switches."""
    bad = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.Assign, ast.Return, ast.Expr, ast.IfExp)) and _is_switch(node):
            if isinstance(node, ast.If):
                # only the test expression of an If is excluded; bodies may still be mutated
                bad.update(range(node.test.lineno, node.test.end_lineno + 1))
            else:
                bad.update(range(node.lineno, node.end_lineno + 1))
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MUTANTS" for t in node.targets):
            bad.update(range(node.lineno, node.end_lineno + 1))
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str):
            bad.update(range(node.lineno, node.end_lineno + 1))  # docstrings
    return bad


SWAP_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
SWAP_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}


def _sites(tree, bad):
    """Enumerate (op, path-index, description) mutation sites."""
    sites = []
    nodes = list(ast.walk(tree))
    for i, n in enumerate(nodes):
        ln = getattr(n, "lineno", None)
        if ln is None or ln in bad:
            continue
        if isinstance(n, ast.BinOp) and type(n.op) in SWAP_BIN:
            if not (isinstance(n.op, ast.Add) and (isinstance(n.left, ast.Constant) and isinstance(n.left.value, str)
                                                   or isinstance(n.right, ast.Constant) and isinstance(n.right.value, str))):
                sites.append(("AOR", i, f"{type(n.op).__name__}->{SWAP_BIN[type(n.op)].__name__}"))
        elif isinstance(n, ast.Compare) and type(n.ops[0]) in SWAP_CMP:
            sites.append(("ROR", i, f"{type(n.ops[0]).__name__}->{SWAP_CMP[type(n.ops[0])].__name__}"))
        elif isinstance(n, ast.BoolOp):
            sites.append(("LCR", i, "and<->or"))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            sites.append(("UOD", i, "remove not"))
        elif isinstance(n, ast.Constant):
            if isinstance(n.value, bool):
                sites.append(("CRP", i, f"{n.value}->{not n.value}"))
            elif isinstance(n.value, int) and not isinstance(n.value, bool):
                sites.append(("CRP", i, f"{n.value}->{n.value + 1}"))
                sites.append(("CRP0", i, f"{n.value}->0"))
            elif isinstance(n.value, str) and n.value:
                sites.append(("CRP", i, "string->''"))
        elif isinstance(n, ast.If):
            sites.append(("CND", i, "negate condition"))
        elif isinstance(n, ast.Return) and n.value is not None:
            sites.append(("RVR", i, "return None"))
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.Expr, ast.Return, ast.If, ast.For)) and \
                not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant)):
            sites.append(("SDL", i, f"delete {type(n).__name__}"))
    return sites


def _apply(tree, op, idx):
    t = copy.deepcopy(tree)
    n = list(ast.walk(t))[idx]
    if op == "AOR":
        n.op = SWAP_BIN[type(n.op)]()
    elif op == "ROR":
        n.ops[0] = SWAP_CMP[type(n.ops[0])]()
    elif op == "LCR":
        n.op = ast.Or() if isinstance(n.op, ast.And) else ast.And()
    elif op == "UOD":
        repl = n.operand
        for parent in ast.walk(t):
            for field, val in ast.iter_fields(parent):
                if val is n:
                    setattr(parent, field, repl)
                elif isinstance(val, list):
                    for k, v in enumerate(val):
                        if v is n:
                            val[k] = repl
    elif op == "CRP":
        if isinstance(n.value, bool):
            n.value = not n.value
        elif isinstance(n.value, int):
            n.value = n.value + 1
        else:
            n.value = ""
    elif op == "CRP0":
        n.value = 0
    elif op == "CND":
        n.test = ast.UnaryOp(op=ast.Not(), operand=n.test)
    elif op == "RVR":
        n.value = ast.Constant(value=None)
    elif op == "SDL":
        for parent in ast.walk(t):
            for field, val in ast.iter_fields(parent):
                if isinstance(val, list):
                    for k, v in enumerate(val):
                        if v is n:
                            val[k] = ast.Pass()
    ast.fix_missing_locations(t)
    return ast.unparse(t)


def generate(sample: int | None = None, seed: int = 0) -> list[AstMutant]:
    src = open(SRC_PATH, encoding="utf-8").read()
    tree = ast.parse(src)
    bad = _excluded_lines(tree)
    sites = _sites(tree, bad)
    rng = random.Random(seed)
    if sample and sample < len(sites):
        # stratified by operator so that every operator is represented
        by = {}
        for s in sites:
            by.setdefault(s[0], []).append(s)
        quota = {k: max(1, round(sample * len(v) / len(sites))) for k, v in by.items()}
        sites = [s for k, v in by.items() for s in rng.sample(v, min(len(v), quota[k]))]
    nodes = list(ast.walk(tree))
    out, seen = [], set()
    for op, idx, desc in sites:
        try:
            msrc = _apply(tree, op, idx)
        except Exception:
            continue
        if msrc in seen or msrc == ast.unparse(tree):
            continue
        seen.add(msrc)
        out.append(AstMutant(f"A{len(out):04d}_{op}", op, getattr(nodes[idx], "lineno", -1), desc, msrc))
    return out


def load(mut: AstMutant):
    """Compile a mutant into a fresh module; returns the module or raises."""
    mod = types.ModuleType(f"sdgtest.sut_{mut.mid}")
    mod.__package__ = "sdgtest"
    mod.__file__ = SRC_PATH
    import sys
    sys.modules[mod.__name__] = mod  # dataclasses need the module registered
    try:
        exec(compile(mut.source, mod.__file__, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(mod.__name__, None)
    return mod


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def call_with_timeout(fn, seconds=2):
    if not hasattr(signal, "SIGALRM"):  # Windows: no alarm signal; run without a time limit
        return fn()
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
