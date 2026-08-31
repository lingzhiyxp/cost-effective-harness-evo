#!/usr/bin/env bash
# Shell-side half of the LLM provider switch. Source it, then call
# `use_llm_provider <name>` to point LLM_API_KEY / LLM_BASE_URL at one of the
# provider-scoped pairs held side by side in .env (UMD_LLM_API_KEY,
# SCALE_LLM_API_KEY, ...).
#
# Two halves are needed because the two arms reach the LLM differently:
#   step-evolve arm -> evolve.py, which calls load_dotenv(override=True) and would
#                      overwrite anything exported here; it does its own resolution
#                      in _resolve_llm_provider(), keyed off LLM_PROVIDER.
#   baseline arm    -> `uv run harbor run`, which never loads .env at all and reads
#                      LLM_API_KEY / LLM_BASE_URL straight from the environment.
# Exporting LLM_PROVIDER *and* the resolved pair covers both without either half
# having to know which arm is running.
#
# Usage:
#   . scripts/llm_provider.sh
#   use_llm_provider scale
#   uv run harbor run ...
#
# Providers are whatever .env defines a <NAME>_LLM_API_KEY for. Naming one that
# isn't defined is an error rather than a silent fallback to the default account.

use_llm_provider() {
    local name="${1:-}"
    if [ -z "$name" ]; then
        echo "use_llm_provider: 需要 provider 名（.env 中定义了 <NAME>_LLM_API_KEY 的那些）" >&2
        return 2
    fi
    local prefix; prefix="$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')"

    local key_var="${prefix}_LLM_API_KEY"
    local url_var="${prefix}_LLM_BASE_URL"
    local model_var="${prefix}_LLM_MODEL"

    if [ -z "${!key_var:-}" ]; then
        echo "use_llm_provider: .env 中没有 $key_var（先 set -a && . ./.env && set +a）" >&2
        return 1
    fi

    export LLM_PROVIDER="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')"
    export LLM_API_KEY="${!key_var}"
    [ -n "${!url_var:-}" ]   && export LLM_BASE_URL="${!url_var}"
    [ -n "${!model_var:-}" ] && export LLM_MODEL="${!model_var}"

    # Agent Debugger keeps its own pair; fall back to the main one so a provider
    # only has to define ADB_* when it actually differs.
    local adb_key_var="${prefix}_ADB_LLM_API_KEY"
    local adb_url_var="${prefix}_ADB_LLM_BASE_URL"
    export ADB_LLM_API_KEY="${!adb_key_var:-${!key_var}}"
    [ -n "${!adb_url_var:-${!url_var:-}}" ] && export ADB_LLM_BASE_URL="${!adb_url_var:-${!url_var}}"

    echo "[llm] provider=$LLM_PROVIDER  base_url=$LLM_BASE_URL  key=${LLM_API_KEY:0:6}***"
}

# 冒烟检查：用当前生效的 LLM_API_KEY / LLM_BASE_URL 打一次 /responses，
# 确认密钥、URL 形式（Scale 的 base_url 少了 /v1 会 403）与模型名都对。
check_llm_provider() {
    local model="${1:-${LLM_MODEL:-gpt-5.4}}"
    uv run python - "$model" <<'PYEOF'
import json, os, sys, urllib.request, urllib.error
model = sys.argv[1]
base = (os.environ.get("LLM_BASE_URL") or "").rstrip("/")
key = os.environ.get("LLM_API_KEY") or ""
if not base or not key:
    print("  [FAIL] LLM_BASE_URL 或 LLM_API_KEY 未设置"); sys.exit(1)
req = urllib.request.Request(
    f"{base}/responses", method="POST",
    data=json.dumps({"model": model, "input": "ok", "reasoning": {"effort": "low"}}).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=90) as r:
        usage = json.load(r).get("usage") or {}
except urllib.error.HTTPError as e:
    print(f"  [FAIL] {base}/responses -> HTTP {e.code}: {e.read().decode()[:200]}"); sys.exit(1)
except Exception as e:
    print(f"  [FAIL] {base}/responses -> {type(e).__name__}: {e}"); sys.exit(1)
# UsageTracer / usage_report 依赖的字段，缺任何一个都会让成本审计失效
missing = [f for f in ("input_tokens", "output_tokens", "total_tokens") if f not in usage]
if "cached_tokens" not in (usage.get("input_tokens_details") or {}):
    missing.append("input_tokens_details.cached_tokens")
if "reasoning_tokens" not in (usage.get("output_tokens_details") or {}):
    missing.append("output_tokens_details.reasoning_tokens")
if missing:
    print(f"  [FAIL] /responses 可用但 usage 缺字段: {missing}（成本审计会失效）"); sys.exit(1)
print(f"  [ok] {base}/responses 可用，model={model}，usage 字段齐全")
PYEOF
}
