"""Measure agreement between the reference harness and native framework verdicts.

Runs promptfoo and Inspect AI on the exported suite for the reference agent and every
mutant, compares each pass/fail verdict with the harness verdict for the same
(test, agent) pair, and prints the sentence for the footnote in Section 4.

Usage (from the project root, on a machine with internet access):
    pip install inspect-ai            # plus Node.js for npx promptfoo
    python export_suite.py            # writes export/ (if not done yet)
    python frameworks/measure_agreement.py            # run frameworks + compare
    python frameworks/measure_agreement.py --skip-run # only compare existing outputs
Options: --frameworks promptfoo inspect   --mutants M05_guard_off M06_guard_no_decode ...
Outputs: frameworks/agreement/<framework>_<agent>.json and frameworks/agreement/summary.json
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from sdgtest.harness import check  # noqa: E402
from sdgtest.kb import build_kb  # noqa: E402
from sdgtest.sut import MUTANTS, CampusAgent, safe_call  # noqa: E402

OUT = os.path.join(ROOT, "frameworks", "agreement")
AGENTS = ["reference"] + list(MUTANTS)


def load_suite():
    with open(os.path.join(ROOT, "export", "suite.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def harness_verdicts(suite, agents):
    kb = build_kb(seed=7)
    out = {}
    for a in agents:
        agent = CampusAgent(kb, None if a == "reference" else a)
        out[a] = {t["id"]: all(check(x, safe_call(agent, t["query"], t["attachments"])) for x in t["assertions"])
                  for t in suite}
    return out


# ---------------------------------------------------------------- promptfoo
def _run_logged(cmd, name, env=None):
    """Run a framework command, save its full output to agreement/<name>.log and fail loudly."""
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = os.path.join(OUT, f"{name}.log")
    with open(log, "w", encoding="utf-8") as fh:
        fh.write("COMMAND: " + " ".join(map(str, cmd)) + "\n\nSTDOUT:\n" + (r.stdout or "") + "\n\nSTDERR:\n" + (r.stderr or ""))
    if r.returncode != 0:
        tail = "\n".join(((r.stderr or "") + (r.stdout or "")).strip().splitlines()[-15:])
        raise RuntimeError(f"{name} exited with code {r.returncode}. Last lines:\n{tail}\n(full output: {log})")
    return r


def run_promptfoo(agent):
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        raise RuntimeError("npx not found: install Node.js (nodejs.org, LTS) and open a new terminal to run promptfoo")
    path = os.path.join(OUT, f"promptfoo_{agent}.json")
    env = dict(os.environ, SDG_MUTANT="" if agent == "reference" else agent, PROMPTFOO_PYTHON=sys.executable)
    _run_logged([npx, "-y", "promptfoo@latest", "eval", "-c", os.path.join(ROOT, "frameworks", "promptfooconfig.yaml"),
                 "-o", path, "--no-cache", "--no-progress-bar"], f"promptfoo_{agent}", env)
    return path


def _find_rows(obj):
    """Locate promptfoo result rows (dicts with 'success' and a test case) at any depth/version."""
    if isinstance(obj, list):
        if obj and all(isinstance(r, dict) and "success" in r for r in obj):
            return obj
        for x in obj:
            r = _find_rows(x)
            if r:
                return r
    elif isinstance(obj, dict):
        for k in ("results", "table", "body"):
            if k in obj:
                r = _find_rows(obj[k])
                if r:
                    return r
        for v in obj.values():
            r = _find_rows(v)
            if r:
                return r
    return None


def parse_promptfoo(path):
    with open(path, encoding="utf-8") as fh:
        rows = _find_rows(json.load(fh)) or []
    verdicts = {}
    for r in rows:
        desc = (r.get("testCase") or {}).get("description") or r.get("description") or ""
        tid = desc.split(" ")[0]
        if tid:
            verdicts[tid] = bool(r["success"])
    return verdicts


# ---------------------------------------------------------------- Inspect AI
def _inspect_cmd():
    """The 'inspect' CLI that belongs to this Python environment (never a system-wide one)."""
    scripts = os.path.dirname(sys.executable)
    for name in ("inspect.exe", "inspect"):
        cand = os.path.join(scripts, name)
        if os.path.exists(cand):
            return [cand]
    return [sys.executable, "-m", "inspect_ai"]


def _load_inspect_task():
    import importlib.util
    spec = importlib.util.spec_from_file_location("sdg_inspect_task", os.path.join(ROOT, "frameworks", "inspect_task.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_inspect(agent):
    """Run the Inspect AI task in-process through inspect_ai.eval(); fall back to the CLI if that fails.
    Inspect resolves task files and log folders with pathlib glob patterns, which must be RELATIVE on
    Python 3.12+ ("Non-relative patterns are unsupported"), so we work from the project folder and pass
    only relative paths."""
    os.chdir(ROOT)
    rel_log = os.path.join("frameworks", "agreement", f"inspect_{agent}")
    os.makedirs(rel_log, exist_ok=True)
    log_dir = os.path.join(ROOT, rel_log)
    mutant = "" if agent == "reference" else agent
    try:
        from inspect_ai import eval as inspect_eval
        task = _load_inspect_task().sdg_suite(mutant=mutant)
        try:
            logs = inspect_eval(task, model="mockllm/model", log_dir=rel_log, log_format="json", display="none")
        except TypeError:  # older Inspect versions without display/log_format arguments
            logs = inspect_eval(task, model="mockllm/model", log_dir=rel_log)
        log = logs[0]
        if getattr(log, "status", "success") != "success":
            raise RuntimeError(f"Inspect AI run for {agent} ended with status '{log.status}': {getattr(log, 'error', '')}")
        _INSPECT_MEMO[log_dir] = log
        return log_dir
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback
        with open(os.path.join(OUT, f"inspect_{agent}_inprocess_error.log"), "w", encoding="utf-8") as fh:
            traceback.print_exc(file=fh)
        print(f"    in-process Inspect AI failed ({type(e).__name__}: {e}); trying the command line", flush=True)
    cmd = _inspect_cmd() + ["eval", "frameworks/inspect_task.py",
                            "--model", "mockllm/model", "--log-dir", rel_log.replace(os.sep, "/"), "--log-format", "json"]
    if mutant:
        cmd += ["-T", f"mutant={mutant}"]
    _run_logged(cmd, f"inspect_{agent}")
    if not glob.glob(os.path.join(log_dir, "*.json")) + glob.glob(os.path.join(log_dir, "*.eval")):
        raise RuntimeError(f"Inspect AI ran but wrote no log to {log_dir}")
    return log_dir


_INSPECT_MEMO = {}


def preflight(frameworks):
    """Check that each framework is usable from THIS Python before spending time on it."""
    ok = []
    print(f"Python used: {sys.executable} ({sys.version.split()[0]})")
    for fw in frameworks:
        if fw == "inspect":
            try:
                import inspect_ai  # noqa: F401
                ok.append(fw)
                print(f"  Inspect AI {getattr(inspect_ai, '__version__', '?')}: found")
            except ImportError:
                print("  Inspect AI: NOT installed in this Python. Fix:  " + sys.executable + " -m pip install inspect-ai")
        elif fw == "promptfoo":
            if shutil.which("npx") or shutil.which("npx.cmd"):
                ok.append(fw)
                print("  Node.js/npx: found (promptfoo will be downloaded by npx)")
            else:
                print("  promptfoo: Node.js/npx NOT found. Fix: install Node.js LTS from nodejs.org, then open a new terminal")
    return ok


def parse_inspect(log_dir):
    log = _INSPECT_MEMO.get(log_dir)
    if log is None or not getattr(log, "samples", None):
        from inspect_ai.log import read_eval_log
        logs = sorted(glob.glob(os.path.join(log_dir, "*.json")) + glob.glob(os.path.join(log_dir, "*.eval")), key=os.path.getmtime)
        if not logs:
            return {}
        log = read_eval_log(logs[-1])
    verdicts = {}
    for s in log.samples or []:
        score = next(iter(s.scores.values())) if s.scores else None
        if score is not None:
            verdicts[str(s.id)] = str(score.value) in ("C", "1", "1.0", "True", "correct")
    return verdicts


RUN = {"promptfoo": run_promptfoo, "inspect": run_inspect}
PARSE = {"promptfoo": parse_promptfoo, "inspect": parse_inspect}
LOC = {"promptfoo": lambda a: os.path.join(OUT, f"promptfoo_{a}.json"), "inspect": lambda a: os.path.join(OUT, f"inspect_{a}")}
NAME = {"promptfoo": "promptfoo", "inspect": "Inspect AI"}


def mutation_score(verdicts, ids):
    killed = sum(1 for m in MUTANTS if m in verdicts and any(verdicts["reference"].get(i) and verdicts[m].get(i) is False for i in ids))
    return killed / len(MUTANTS)


def _venv_python():
    for rel in ((".venv", "Scripts", "python.exe"), (".venv", "bin", "python")):
        p = os.path.join(ROOT, *rel)
        if os.path.exists(p):
            return p
    return None


def main():
    try:
        import inspect_ai  # noqa: F401
    except ImportError:
        vp = _venv_python()
        if vp and os.path.abspath(vp) != os.path.abspath(sys.executable) and not os.environ.get("SDG_RELAUNCHED"):
            print(f"Inspect AI is not installed in {sys.executable}; re-running with the project environment {vp}", flush=True)
            sys.exit(subprocess.call([vp, os.path.abspath(__file__), *sys.argv[1:]], env=dict(os.environ, SDG_RELAUNCHED="1")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--frameworks", nargs="+", default=["promptfoo", "inspect"], choices=list(RUN))
    ap.add_argument("--mutants", nargs="+", default=list(MUTANTS))
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    agents = ["reference"] + args.mutants
    suite = load_suite()
    ids = [t["id"] for t in suite]
    ref = harness_verdicts(suite, agents)
    summary = {"n_tests": len(ids), "agents": agents, "harness_mutation_score": mutation_score(ref, ids), "frameworks": {}}
    usable = args.frameworks if args.skip_run else preflight(args.frameworks)
    if not usable:
        print("\nNo framework can run from this Python. Use the project environment, e.g.\n"
              "  .venv\\Scripts\\python.exe frameworks\\measure_agreement.py\nafter scripts\\run_frameworks.bat has created it.")
        return
    for fw in usable:
        got = {}
        try:
            for a in agents:
                print(f"  {fw}: running {a} ...", flush=True)
                loc = LOC[fw](a) if args.skip_run else RUN[fw](a)
                got[a] = PARSE[fw](loc) if os.path.exists(loc) else {}
                if not got[a]:
                    raise RuntimeError(f"no verdicts parsed for agent '{a}' (output: {loc})")
        except Exception as e:  # e.g. Node.js/npx or inspect-ai missing: skip this framework, keep the others
            print(f"WARNING: {fw} skipped ({type(e).__name__}: {e})", flush=True)
            summary["frameworks"][fw] = {"skipped": f"{type(e).__name__}: {e}"}
            continue
        pairs = [(a, i) for a in agents for i in ids]
        covered = [(a, i) for a, i in pairs if i in got[a]]
        agree = [(a, i) for a, i in covered if got[a][i] == ref[a][i]]
        disagree = [{"agent": a, "test": i, "harness": ref[a][i], fw: got[a][i]} for a, i in covered if got[a][i] != ref[a][i]]
        summary["frameworks"][fw] = {
            "pairs_expected": len(pairs), "pairs_with_verdict": len(covered), "agreeing": len(agree),
            "agreement_rate": len(agree) / len(covered) if covered else None,
            "mutation_score": mutation_score(got, ids) if got.get("reference") else None,
            "disagreements_first_20": disagree[:20]}
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "disagreements_first_20"} for k, v in summary["frameworks"].items()}, indent=1))
    summary["frameworks"] = {k: v for k, v in summary["frameworks"].items() if "skipped" not in v} | \
        {k: v for k, v in summary["frameworks"].items() if "skipped" in v}
    parts = []
    for fw, r in summary["frameworks"].items():
        if r.get("agreement_rate") is None:
            continue
        parts.append(f"{100 * r['agreement_rate']:.2f}\\% of (test, agent) pairs under {NAME[fw]} "
                     f"({r['agreeing']:,}/{r['pairs_with_verdict']:,})")
    if parts:
        print("\nFootnote text for Section 4 (replace the \\todoauth{...} footnote):\n")
        print(f"\\footnote{{Re-running the exported suite ({len(ids)} tests) against the reference agent and all "
              f"{len(agents) - 1} mutants under the native frameworks reproduced the harness verdicts for "
              f"{' and '.join(parts)}; mutation scores were "
              + ", ".join(f"{summary['frameworks'][fw]['mutation_score']:.2f} ({NAME[fw]})" for fw in summary['frameworks'] if summary['frameworks'][fw].get('mutation_score') is not None)
              + f" versus {summary['harness_mutation_score']:.2f} (harness).}}")
    else:
        print("\nNo framework verdicts were obtained. Reasons:")
        for fw, r in summary["frameworks"].items():
            print(f"  - {NAME.get(fw, fw)}: {r.get('skipped', 'ran but produced no verdicts')}")
        print("Send this output (and the files in frameworks/agreement/) so the scripts can be fixed.")


if __name__ == "__main__":
    main()
