"""Minimal, dependency-free chat client (Anthropic, OpenAI, Ollama, mock) with an on-disk cache.

Environment variables:
  ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY   provider keys
  OLLAMA_HOST                             default http://localhost:11434
Every call is cached in llmstudy/cache/<hash>.json, so interrupted runs resume without paying twice.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "cache")

# USD per million tokens (input, output) - edit to current prices; used only for the cost guard.
PRICES = {"claude-haiku-4-5": (1.0, 5.0), "claude-3-5-haiku-latest": (0.8, 4.0), "claude-3-haiku-20240307": (0.25, 1.25), "claude-sonnet-4-5": (3.0, 15.0), "gpt-4o-mini": (0.15, 0.6),
          "gpt-4.1-mini": (0.4, 1.6), "gpt-4.1": (2.0, 8.0), "gpt-4.1-nano": (0.1, 0.4),
          "deepseek-flash": (0.30, 1.20), "deepseek-v4-pro": (1.32, 3.96)}  # DeepSeek: peak-hour list prices


# Local Ollama calls must never go through a system/corporate proxy
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _open(req, timeout, local=False):
    return (_DIRECT.open if local else urllib.request.urlopen)(req, timeout=timeout)


class BudgetExceeded(RuntimeError):
    pass


def _budget(default=40.0):
    """SDG_MAX_USD, read leniently (a pasted 'set' line or a '$' sign must not crash the run)."""
    import re
    raw = os.environ.get("SDG_MAX_USD", "")
    m = re.match(r"\s*\$?\s*(\d+(?:\.\d+)?)", raw)
    if raw and not m:
        print(f"WARNING: SDG_MAX_USD='{raw}' is not a number; using US$ {default:.0f}")
    return float(m.group(1)) if m else default


class LLM:
    spent_usd = 0.0
    max_usd = _budget()
    calls = 0

    def __init__(self, provider: str, model: str, temperature: float = 0.0, max_tokens: int = 600, mock_fn=None):
        self.provider, self.model, self.temperature, self.max_tokens = provider, model, temperature, max_tokens
        self.mock_fn = mock_fn
        os.makedirs(CACHE, exist_ok=True)

    def chat(self, system: str, messages: list[dict], sample: int = 0, json_mode: bool = False) -> str:
        """messages: [{'role': 'user'|'assistant', 'content': str}]; sample distinguishes repeated stochastic runs.
        json_mode asks local (Ollama) models for syntactically valid JSON; other providers follow the prompt."""
        self._json = json_mode
        ident = [self.provider, self.model, self.temperature, system, messages, sample]
        if json_mode and self.provider == "ollama":
            ident.append("json")
        key = hashlib.sha256(json.dumps(ident, ensure_ascii=False).encode()).hexdigest()
        path = os.path.join(CACHE, key[:2], key + ".json")
        if os.path.exists(path):
            return json.load(open(path, encoding="utf-8"))["text"]
        if self.provider == "mock":
            text, usage = self.mock_fn(system, messages), (0, 0)
        else:
            if LLM.spent_usd > LLM.max_usd:
                raise BudgetExceeded(f"cost guard reached (US$ {LLM.spent_usd:.2f} > {LLM.max_usd}); raise SDG_MAX_USD to continue")
            text, usage = self._call(system, messages)
            pin, pout = (0.0, 0.0) if self.provider == "ollama" else PRICES.get(self.model, (3.0, 15.0))
            LLM.spent_usd += usage[0] / 1e6 * pin + usage[1] / 1e6 * pout
        LLM.calls += 1
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"text": text, "usage": usage, "model": self.model}, open(path, "w", encoding="utf-8"))
        return text

    # ------------------------------------------------------------------ providers
    def _post(self, url, body, headers, retries=6):
        data = json.dumps(body).encode()
        for k in range(retries):
            try:
                req = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
                with _open(req, 1800 if self.provider == "ollama" else 120, local=self.provider == "ollama") as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 529) and k < retries - 1:
                    time.sleep(2 ** k)
                    continue
                raise RuntimeError(f"{self.provider} HTTP {e.code}: {e.read().decode()[:300]}") from e
            except urllib.error.URLError:
                if k < retries - 1:
                    time.sleep(2 ** k)
                    continue
                raise

    def _call(self, system, messages):
        if self.provider == "anthropic":
            r = self._post("https://api.anthropic.com/v1/messages",
                           {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                            "system": system, "messages": messages},
                           {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"})
            text = "".join(b.get("text", "") for b in r.get("content", []))
            u = r.get("usage", {})
            return text, (u.get("input_tokens", 0), u.get("output_tokens", 0))
        if self.provider in ("openai", "deepseek"):
            url = ("https://api.deepseek.com/chat/completions" if self.provider == "deepseek"
                   else "https://api.openai.com/v1/chat/completions")
            key = os.environ["DEEPSEEK_API_KEY" if self.provider == "deepseek" else "OPENAI_API_KEY"]
            r = self._post(url,
                           {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                            "messages": [{"role": "system", "content": system}] + messages},
                           {"authorization": f"Bearer {key}"})
            u = r.get("usage", {})
            return r["choices"][0]["message"]["content"], (u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        if self.provider == "ollama":
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            body = {"model": self.model, "stream": False,
                                                "options": {"temperature": self.temperature, "num_predict": self.max_tokens,
                                                            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "8192"))},
                                                "messages": [{"role": "system", "content": system}] + messages}
            if getattr(self, "_json", False):
                body["format"] = "json"
            r = self._post(f"{host}/api/chat", body, {})
            return r["message"]["content"], (r.get("prompt_eval_count", 0), r.get("eval_count", 0))
        raise ValueError(self.provider)


def parse_json(text: str):
    """Extract the first JSON object from a model reply."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):]
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------- key / model check
def list_models(provider):
    """Model ids the key can use; raises RuntimeError with a readable message on a bad key."""
    if provider == "anthropic":
        url, hdr = "https://api.anthropic.com/v1/models?limit=1000", {"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                                                                      "anthropic-version": "2023-06-01"}
    elif provider == "openai":
        url, hdr = "https://api.openai.com/v1/models", {"authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}
    elif provider == "deepseek":
        url, hdr = "https://api.deepseek.com/models", {"authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY', '')}"}
    elif provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        try:
            with _open(f"{host}/api/tags", 30, local=True) as r:
                return [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Ollama is not running at {host} ({type(e).__name__}). Install it from ollama.com "
                               "and start it (it runs in the background after installation).") from e
    else:
        return []
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError(f"the {provider} API rejected the key (HTTP {e.code}). Check that it is a valid "
                               f"{ {'anthropic': 'Anthropic (sk-ant-...)', 'deepseek': 'DeepSeek (sk-...)'}.get(provider, 'OpenAI (sk-...)') } key.") from e
        raise RuntimeError(f"{provider} model list failed: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach the {provider} API ({e.reason}). Check the internet connection "
                           "or a firewall/proxy on this network.") from e
    return [m["id"] for m in data.get("data", [])]


def pick_small_model(provider, agent_model, available):
    """A smaller, older model of the same provider for fault L08 (never the agent model itself)."""
    prefs = {"anthropic": ["claude-3-5-haiku-latest", "claude-3-5-haiku-20241022", "claude-3-haiku-20240307"],
             "openai": ["gpt-4.1-nano", "gpt-3.5-turbo", "gpt-4o-mini"],
             "deepseek": ["deepseek-flash"]}.get(provider, [])
    same = lambda m: m == agent_model or m.startswith(agent_model + "-")  # alias vs dated id
    for m in prefs:
        if (m in available or any(a.startswith(m) for a in available)) and not same(m):
            return m
    cands = [m for m in available if not same(m) and ("haiku" in m or "nano" in m)]
    return sorted(cands)[0] if cands else None


if __name__ == "__main__":
    import sys
    prov = sys.argv[1] if len(sys.argv) > 1 else "anthropic"
    wanted = sys.argv[2:]
    try:
        avail = list_models(prov)
    except Exception as e:  # noqa: BLE001
        print("ERROR:", e)
        sys.exit(1)
    print(f"{prov} {'server' if prov == 'ollama' else 'key'} OK; {len(avail)} models available: {', '.join(sorted(avail))}")
    missing = [m for m in wanted if m not in avail and not any(a.startswith(m) for a in avail)
               and not (prov == "ollama" and m + ":latest" in avail)]
    for m in wanted:
        print(f"  {m}: {'available' if m not in missing else 'NOT available'}")
    if missing:
        print("ERROR: some models are not available for this key.")
        sys.exit(1)
    # one tiny paid request (a few tokens) to catch billing problems before the long run
    try:
        LLM(prov, wanted[0] if wanted else "", max_tokens=5)._call("Reply with OK.", [{"role": "user", "content": "ping"}])
        print("test request OK" + ("" if prov == "ollama" else " (account has credit)"))
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "credit balance" in msg or "insufficient_quota" in msg or "billing" in msg.lower():
            print("ERROR: the account has no credit left. Add credit (Anthropic Console > Plans & Billing, "
                  "or OpenAI > Billing), then run again.")
        else:
            print("ERROR: test request failed:", msg[:300])
        sys.exit(1)
