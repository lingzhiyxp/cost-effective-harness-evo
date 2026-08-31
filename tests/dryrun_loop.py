"""Drive the wired loop's HCE path over two iterations with fabricated results."""
import json, pathlib, shutil, subprocess, sys, tempfile
sys.argv = ["x"]
sys.path.insert(0, "/fs/nexus-projects/MeMas/projects/Harness-Evo/cost-effective-harness-evo")
import evolve
from hce import runtime as R
from hce.profiles import TaskProfileDB
from hce import gate as G

TASKS = ["chess-best-move", "cobol-modernization", "configure-git-webserver",
         "extract-elf", "gcode-to-text", "git-multibranch"]

def fake_job(root, tasks, scores):
    root.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        d = root / f"{t}__ab12cd"; (d / "verifier").mkdir(parents=True); (d / "agent").mkdir()
        (d / "verifier" / "reward.txt").write_text("1.0" if scores[t] >= 1 else "0.0")
        (d / "result.json").write_text(json.dumps({"exception_info": None}))
        (d / "agent" / "nexau_in_memory_tracer.cleaned.json").write_text(json.dumps({
            "generation_count": 12 + len(t),
            "messages": [{"role": "assistant", "tool_calls": [
                {"name": "Tool: run_shell_command",
                 "output": {"result": {"content": "x" * (4000 + 400 * len(t)), "exit_code": 0}}}]}]}))
        (d / "agent" / "usage.jsonl").write_text("\n".join(json.dumps(
            {"role": "code_agent", "model": "gpt-5.4-mini",
             "usage": {"input_tokens": 9000 + 500 * i, "output_tokens": 400}})
            for i in range(3)))
    return root

with tempfile.TemporaryDirectory() as tmp:
    exp = pathlib.Path(tmp) / "exp"; (exp / "runs").mkdir(parents=True)
    ws = exp / "workspace"; ws.mkdir()
    (ws / "systemprompt.md").write_text("seed\n"); (ws / "tool.py").write_text("v = 1\n")
    g = lambda *a: subprocess.run(["git", *a], cwd=ws, capture_output=True, text=True).stdout.strip()
    g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
    g("add", "-A"); g("commit", "-qm", "v0")

    cfg = evolve.load_config("configs/hce/smoke-tb2-6.yaml")
    db = TaskProfileDB.load(exp / "task_profiles.json")

    # ---- iteration 1: no contract, uniform sample, becomes the profile ------
    g("tag", "iteration_1_before")
    fp1 = evolve.harness_fingerprint(ws)
    plan1 = R.plan_iteration(config=cfg, exp_dir=exp, project_dir=evolve.PROJECT_DIR,
                             iteration=1, db=db, all_tasks=TASKS)
    print(f"iter1 mode={plan1.record['mode']} selected={len(plan1.task_names)}/6 "
          f"rollouts={plan1.record['budget_rollouts']}")
    truth1 = {t: (1.0 if i % 2 == 0 else 0.0) for i, t in enumerate(TASKS)}
    job1 = fake_job(exp / "runs" / "iteration_001" / "b", plan1.task_names, truth1)
    m1 = R.measure(job1, max_iterations=300, max_context_tokens=200000)
    est1, dec1 = R.score_iteration(plan=plan1, measured=m1, s_ref=None, config=cfg)
    print(f"iter1 estimate={est1.s_hat:.4f} se={est1.se:.4f} gate={dec1.verdict}")
    R.commit_measurements(db=db, measured=m1, iteration=1, fingerprint=fp1, accepted=True)
    s_ref = est1.s_hat

    # ---- iteration 2: the agent ships a contract, gate rejects, revert ------
    (exp / "change_manifest.json").write_text(json.dumps({"iteration": 1, "changes": [{
        "id": "chg-1", "files": ["tool.py"], "predicted_fixes": [],
        "evaluation_contract": {"scope": "conditional", "expected_effect": "improve",
            "activation_predicate": {"max_command_output_tokens": ">1000"}}}]}))
    (ws / "tool.py").write_text("v = 2\n"); (ws / "added.py").write_text("new\n")
    g("add", "-A"); g("commit", "-qm", "chg-1"); g("tag", "iteration_1")
    g("tag", "iteration_2_before")

    plan2 = R.plan_iteration(config=cfg, exp_dir=exp, project_dir=evolve.PROJECT_DIR,
                             iteration=2, db=db, all_tasks=TASKS)
    r = plan2.record
    print(f"iter2 mode={r['mode']} activated={r['rationale'].get('n_activated')} "
          f"selected={len(plan2.task_names)} strata=" +
          " ".join(f"{n}({len(s['pool'])}->{len(s['taken'])},pi={s['pi']})"
                   for n, s in r["strata"].items()))
    truth2 = {t: 0.0 for t in TASKS}                      # the change breaks everything
    job2 = fake_job(exp / "runs" / "iteration_002" / "b", plan2.task_names, truth2)
    m2 = R.measure(job2, max_iterations=300, max_context_tokens=200000)
    est2, dec2 = R.score_iteration(plan=plan2, measured=m2, s_ref=s_ref, config=cfg)
    print(f"iter2 estimate={est2.s_hat:.4f} (ref {s_ref:.4f}) gate={dec2.verdict} [{dec2.rule}]")
    assert not dec2.accepted, "a change that breaks every task must be rejected"

    sha = G.revert_to(ws, "iteration_1_before", reason=dec2.rule, iteration=1)
    assert g("rev-parse", "HEAD^{tree}") == g("rev-parse", "iteration_1_before^{tree}")
    assert not (ws / "added.py").exists() and (ws / "tool.py").read_text() == "v = 1\n"
    print(f"revert ok: tree back to iteration_1_before, tags intact, HEAD={sha[:8]}")

    before = db.p_hist(TASKS)
    R.commit_measurements(db=db, measured=m2, iteration=2, fingerprint="fp2", accepted=False)
    after = db.p_hist(TASKS)
    # Iteration 2 measured these at 0.0 and was rejected, so the baseline must
    # be untouched -- including for tasks iteration 1 never measured, which stay
    # at the default rather than adopting a reverted harness's score.
    assert after == before, {t: (before[t], after[t]) for t in TASKS if before[t] != after[t]}
    measured_in_1 = [t for t in TASKS if t in plan1.task_names]
    assert all(after[t] == truth1[t] for t in measured_in_1)
    print(f"rejected measurements did not move p_hist "
          f"({len(measured_in_1)} tasks carry an iteration-1 baseline, "
          f"{6 - len(measured_in_1)} still at default)")

    # Fingerprint after the revert equals iteration 1's -> its results are reusable.
    assert evolve.harness_fingerprint(ws) == fp1
    print("post-revert fingerprint matches iteration 1: next eval of this tree is free")
    print("\nDRY RUN OK")
