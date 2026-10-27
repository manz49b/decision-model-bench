import pytest

from bench.engines import normalise


def test_normalise_choice_and_noul():
    answers = {"intent": {"choice": "refund", "confidence": 0.9}, "unhappy": {"noul": 0.8}}
    assert normalise(answers) == {
        "intent": {"label": "refund", "score": 0.9},
        "unhappy": {"label": None, "score": 0.8},
    }


def test_bootstrap_matches_point_metrics():
    import pandas as pd

    from bench import metrics

    df = pd.DataFrame({"case_id": range(200), "gold": ["a", "b"] * 100, "label": ["a"] * 200})
    out = metrics.choice_summary(df)
    assert out["accuracy"] == 0.5 and out["acc_lo"] < 0.5 < out["acc_hi"]

    df = pd.DataFrame({"case_id": range(4), "gold": [True, True, False, False], "score": [0.9, 0.2, 0.8, 0.1]})
    assert abs(metrics.noul_summary(df)["f1"] - 0.5) < 1e-9  # tp 1, fp 1, fn 1


def test_append_keeps_columns_aligned(tmp_path, monkeypatch):
    import pandas as pd

    from bench import run

    monkeypatch.setattr(run, "RESULTS", tmp_path / "runs.csv")
    run.append(pd.DataFrame([{"a": 1, "b": 2}]))
    run.append(pd.DataFrame([{"b": 4, "a": 3}]))  # same columns, different order
    run.append(pd.DataFrame([{"a": 5, "new": 6, "b": 7}]))  # a new column
    df = pd.read_csv(tmp_path / "runs.csv")
    assert df["a"].tolist() == [1, 3, 5] and df["b"].tolist() == [2, 4, 7] and df["new"].iloc[2] == 6


def test_holm_matches_hand_computation():
    from bench.story_table import holm

    # sorted p: 0.01, 0.02, 0.04 -> 3*0.01=0.03, max(0.03, 2*0.02=0.04)=0.04, max(0.04, 1*0.04)=0.04
    assert holm([0.04, 0.01, 0.02]) == [0.04, 0.03, 0.04]


def test_holm_matches_hand_computed():
    from bench.story_table import holm

    # sorted 0.01, 0.03, 0.04 -> ×3, ×2, ×1 = 0.03, 0.06, 0.04 -> monotone 0.03, 0.06, 0.06
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_mcnemar_exact_on_discordant_pairs():
    import pandas as pd

    from bench import metrics

    # 10 cases only a gets right, none only b: two-sided exact p = 2 × 0.5^10
    a = pd.DataFrame({"case_id": range(20), "repeat": 0, "correct": [True] * 20})
    b = a.assign(correct=[False] * 10 + [True] * 10)
    t = metrics.mcnemar(a, b)
    assert (t["only_a_right"], t["only_b_right"]) == (10, 0)
    assert t["p_value"] == pytest.approx(2 * 0.5**10)
    assert metrics.mcnemar(a, a)["p_value"] == 1.0
