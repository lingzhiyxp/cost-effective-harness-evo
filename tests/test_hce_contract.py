import pytest
from hce.contract import ContractError, SCOPE_GLOBAL, SCOPE_CONDITIONAL, parse


def _manifest(**contract_overrides):
    contract = {"scope": "conditional", "mechanism": "m",
                "activation_predicate": {"max_command_output_tokens": ">2000"},
                "expected_effect": "improve"}
    contract.update(contract_overrides)
    return {"iteration": 3, "changes": [{
        "id": "chg-1", "files": ["tools/shell_tools/run_shell_command.py"],
        "predicted_fixes": ["a"], "risk_tasks": ["b"],
        "evaluation_contract": contract}]}


def test_parses_and_preserves_ahe_fields():
    c = parse(_manifest(), iteration=3)
    assert c.changes[0].change_id == "chg-1"
    assert c.predicted_fixes == ["a"] and c.risk_tasks == ["b"]
    assert not c.is_global


def test_missing_contract_is_an_error_with_guidance():
    m = _manifest(); m["changes"][0].pop("evaluation_contract")
    with pytest.raises(ContractError, match="missing `evaluation_contract`"):
        parse(m, iteration=3)


def test_bad_predicate_message_is_returnable_to_the_agent():
    with pytest.raises(ContractError) as e:
        parse(_manifest(activation_predicate={"nope": ">1"}), iteration=3)
    assert "chg-1" in str(e.value) and "legal keys" in str(e.value)


def test_touching_a_global_file_forces_global_scope():
    m = _manifest()
    m["changes"][0]["files"] = ["systemprompt.md"]
    c = parse(m, iteration=3)
    assert c.is_global and c.changes[0].scope == SCOPE_GLOBAL
    assert c.changes[0].forced_global_reason == "systemprompt.md"


def test_declared_global_needs_no_predicate():
    m = _manifest(scope="global", activation_predicate={})
    c = parse(m, iteration=3)
    assert c.is_global and c.changes[0].activation_predicate == {}


def test_activation_is_a_union_over_changes():
    m = _manifest()
    m["changes"].append({"id": "chg-2", "files": ["x.py"], "evaluation_contract": {
        "scope": "conditional", "activation_predicate": {"has_python_traceback": True},
        "expected_effect": "improve"}})
    c = parse(m, iteration=3)
    mechs = {"t1": {"max_command_output_tokens": 3000, "has_python_traceback": False},
             "t2": {"max_command_output_tokens": 100, "has_python_traceback": True},
             "t3": {"max_command_output_tokens": 100, "has_python_traceback": False}}
    tasks, detail = c.activation(mechs)
    assert tasks == ["t1", "t2"]                       # union, not intersection
    assert detail["mode"] == SCOPE_CONDITIONAL and len(detail["per_change"]) == 2


def test_global_activation_is_everything():
    m = _manifest(scope="global", activation_predicate={})
    tasks, detail = parse(m, iteration=3).activation({"a": {}, "b": {}})
    assert tasks == ["a", "b"] and detail["mode"] == SCOPE_GLOBAL
