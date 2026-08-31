import pytest
from hce.mechanism import FEATURE_SCHEMA
from hce.predicate import PredicateError, validate, matches, activation_set

S = FEATURE_SCHEMA


def test_accepts_every_grammar_form():
    for p in ({"max_command_output_tokens": ">2000"},
              {"n_steps": ">=10"}, {"n_steps": "<77"}, {"n_steps": "<=77"},
              {"n_nonzero_exit": "==0"}, {"n_nonzero_exit": "!=0"},
              {"n_steps": 19}, {"has_python_traceback": True},
              {"has_python_traceback": False}, {"n_steps": "*"},
              {"max_command_output_tokens": ">2000", "has_python_traceback": True}):
        validate(p, S)


def test_unknown_key_is_rejected_and_lists_legal_keys():
    with pytest.raises(PredicateError) as e:
        validate({"invented_feature": ">1"}, S)
    assert "unknown feature" in str(e.value)
    assert "max_command_output_tokens" in str(e.value)   # the legal set is shown


def test_type_mismatches_are_rejected():
    with pytest.raises(PredicateError, match="use true or false"):
        validate({"has_python_traceback": ">8000"}, S)
    with pytest.raises(PredicateError, match="cannot test it"):
        validate({"n_steps": True}, S)


def test_empty_predicate_is_rejected():
    with pytest.raises(PredicateError, match="empty"):
        validate({}, S)


def test_unparseable_comparison_is_rejected():
    with pytest.raises(PredicateError, match="cannot parse"):
        validate({"n_steps": ">> 5"}, S)


def test_keys_are_anded():
    mech = {"max_command_output_tokens": 3000, "has_python_traceback": False}
    assert matches({"max_command_output_tokens": ">2000"}, mech)[0]
    assert not matches({"max_command_output_tokens": ">2000",
                        "has_python_traceback": True}, mech)[0]


def test_missing_feature_matches_and_is_reported():
    matched, unknown = matches({"peak_context_tokens": ">1000"}, {"n_steps": 5})
    assert matched and unknown == ["peak_context_tokens"]


def test_activation_set_is_sorted_and_reports_unknowns():
    mechs = {"b": {"n_steps": 50}, "a": {"n_steps": 5}, "c": {}}
    sel, unknown = activation_set({"n_steps": ">10"}, mechs)
    assert sel == ["b", "c"]              # c has no data, so it is kept
    assert unknown == {"c": ["n_steps"]}


def test_proposal_example_threshold_matches_nothing_here():
    # Guards the finding from the profiling corpus: the proposal's worked
    # example is written against a dataset whose observations are far larger.
    mechs = {f"t{i}": {"max_command_output_tokens": v}
             for i, v in enumerate([167, 2002, 5026])}
    assert activation_set({"max_command_output_tokens": ">8000"}, mechs)[0] == []
    assert len(activation_set({"max_command_output_tokens": ">2000"}, mechs)[0]) == 2
