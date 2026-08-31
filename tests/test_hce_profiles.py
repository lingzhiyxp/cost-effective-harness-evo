from hce.profiles import TaskProfileDB


def _db(tmp_path):
    return TaskProfileDB(tmp_path / "task_profiles.json", dataset="d", token_estimator="chars_div_4")


def test_round_trip(tmp_path):
    db = _db(tmp_path)
    db.set_static("a", difficulty="medium", category="debugging")
    db.record_outcome("a", iteration=1, fingerprint="f1", accepted=True, score=0.667,
                      n_pass=2, n_fail=1)
    db.record_mechanism("a", iteration=1, agg={"n_steps": 20}, per_rollout=[{"n_steps": 20}],
                        mean_usd=0.15)
    db.save()
    back = TaskProfileDB.load(tmp_path / "task_profiles.json")
    assert back.tasks == db.tasks and back.dataset == "d"
    assert back.p_hist(["a"])["a"] == 0.667
    assert back.mechanisms(["a"])["a"] == {"n_steps": 20}


def test_rejected_measurement_does_not_move_baseline(tmp_path):
    db = _db(tmp_path)
    db.record_outcome("a", iteration=1, fingerprint="f1", accepted=True, score=1.0)
    assert db.p_hist(["a"])["a"] == 1.0
    # A rejected iteration is evidence about the change, not a new baseline.
    db.record_outcome("a", iteration=2, fingerprint="f2", accepted=False, score=0.0)
    assert db.p_hist(["a"])["a"] == 1.0
    assert db.tasks["a"]["outcome"]["p_hist_source_iteration"] == 1
    assert len(db.tasks["a"]["outcome"]["history"]) == 2
    # ...and an accepted one afterwards does move it.
    db.record_outcome("a", iteration=3, fingerprint="f3", accepted=True, score=0.5)
    assert db.p_hist(["a"])["a"] == 0.5
    assert db.tasks["a"]["outcome"]["p_hist_source_iteration"] == 3


def test_infra_only_iteration_is_not_a_zero(tmp_path):
    db = _db(tmp_path)
    db.record_outcome("a", iteration=1, fingerprint="f1", accepted=True, score=0.5)
    db.record_outcome("a", iteration=2, fingerprint="f2", accepted=True, score=None, n_infra=3)
    assert db.p_hist(["a"])["a"] == 0.5          # not dragged to 0
    assert db.tasks["a"]["outcome"]["n_accepted_observations"] == 1


def test_unknown_task_gets_default(tmp_path):
    db = _db(tmp_path)
    assert db.p_hist(["missing"])["missing"] == 0.0
    assert db.variance(["missing"])["missing"] == 0.0
    assert db.difficulty(["missing"])["missing"] == "unknown"
