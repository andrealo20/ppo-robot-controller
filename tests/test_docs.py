"""The documentation must not drift away from the measurements.

The README and the design notes both quote the multi-seed table and the
summary statistics derived from it. Those numbers come from
``results/seed_sweep_summary.csv``, which ``run_seed_sweep.sh`` writes after
training and evaluating all five seeds. That sweep costs hours of compute and
cannot run in CI, so nothing else would catch a table hand-edited into
disagreement with the data behind it. This test re-reads the CSV, recomputes
the headline statistics from it, and holds both documents to the result.
"""

import csv
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "seed_sweep_summary.csv"
DOCUMENTS = ["README.md", "docs/design.md"]

# One row of the per-seed table, in the exact shape both documents write it:
# | 0 | 46/50 (92%) | -30.01 +/- 37.45 |
TABLE_ROW = re.compile(
    r"^\|\s*(\d+)\s*"
    r"\|\s*(\d+)/(\d+)\s*\((\d+(?:\.\d+)?)%\)\s*"
    r"\|\s*(-?\d+\.\d+)\s*\+/-\s*(\d+\.\d+)\s*\|$"
)


@pytest.fixture(scope="module")
def sweep():
    """Per-seed rows of the sweep summary, keyed by seed."""
    if not SUMMARY.exists():
        pytest.skip("run run_seed_sweep.sh first")
    with SUMMARY.open(encoding="utf-8", newline="") as fh:
        rows = {row["seed"]: row for row in csv.DictReader(fh)}
    assert rows, "seed sweep summary has no rows"
    return rows


@pytest.fixture(scope="module")
def stats(sweep):
    """Headline statistics, recomputed from the per-seed success rates.

    Both standard deviations are here on purpose: the README quotes the
    sample standard deviation (n-1 denominator, the spread of the five runs
    treated as a sample of the training procedure), while design.md also
    quotes the population one (n denominator).
    """
    rates = [float(row["success_rate"]) for row in sweep.values()]
    n = len(rates)
    mean = sum(rates) / n
    sum_sq = sum((rate - mean) ** 2 for rate in rates)
    return {
        "seeds": n,
        "mean": mean,
        "sample_sd": math.sqrt(sum_sq / (n - 1)),
        "population_sd": math.sqrt(sum_sq / n),
        "min_rate": min(rates),
        "resolved": sum(int(row["resolved"]) for row in sweep.values()),
        "total": sum(int(row["total"]) for row in sweep.values()),
    }


def _flat_text(document):
    """Document text with runs of whitespace collapsed, so that a claim can be
    matched whether or not the paragraph happens to wrap in the middle of it.
    """
    return re.sub(r"\s+", " ", (ROOT / document).read_text(encoding="utf-8"))


def _table_rows(document):
    rows = {}
    for line in (ROOT / document).read_text(encoding="utf-8").splitlines():
        match = TABLE_ROW.match(line.strip())
        if match:
            rows[match.group(1)] = match.groups()[1:]
    return rows


def _search(pattern, document):
    match = re.search(pattern, _flat_text(document))
    assert match, f"{document} has no text matching {pattern!r}"
    return match


@pytest.mark.parametrize("document", DOCUMENTS)
def test_per_seed_table_matches_the_summary_csv(sweep, document):
    table = _table_rows(document)
    assert set(table) == set(sweep), (
        f"{document} tabulates seeds {sorted(table)}, the summary has {sorted(sweep)}"
    )
    for seed, row in sweep.items():
        resolved, total, rate, mean_reward, std_reward = table[seed]
        assert int(resolved) == int(row["resolved"]), f"{document} seed {seed}"
        assert int(total) == int(row["total"]), f"{document} seed {seed}"
        assert float(rate) == float(row["success_rate"]), f"{document} seed {seed}"
        assert float(mean_reward) == float(row["mean_reward"]), (
            f"{document} seed {seed}"
        )
        assert float(std_reward) == float(row["std_reward"]), f"{document} seed {seed}"


@pytest.mark.parametrize("document", DOCUMENTS)
def test_headline_mean_and_sample_sd_match_the_summary_csv(stats, document):
    match = _search(r"Mean (\d+\.\d+)% \+/- (\d+\.\d+)% held-out success", document)
    assert float(match.group(1)) == round(stats["mean"], 1), document
    assert float(match.group(2)) == round(stats["sample_sd"], 1), document


@pytest.mark.parametrize("document", DOCUMENTS)
def test_pooled_episode_count_matches_the_summary_csv(stats, document):
    match = _search(r"pooled:? (\d+)/(\d+) episodes", document)
    assert int(match.group(1)) == stats["resolved"], document
    assert int(match.group(2)) == stats["total"], document


@pytest.mark.parametrize("document", DOCUMENTS)
def test_seed_count_matches_the_summary_csv(stats, document):
    counts = re.findall(r"(\d+) seeds", _flat_text(document))
    assert counts, f"{document} never says how many seeds the sweep covered"
    assert all(int(count) == stats["seeds"] for count in counts), (
        f"{document} claims {counts} seeds, the summary has {stats['seeds']}"
    )


def test_design_notes_population_sd_matches_the_summary_csv(stats):
    match = _search(r"population std (\d+\.\d+)%", "docs/design.md")
    assert float(match.group(1)) == round(stats["population_sd"], 1)


def test_readme_worst_seed_claim_is_still_true(stats):
    """The README says every seed cleared a floor; keep that floor honest."""
    match = _search(r"every individual seed at (\d+)% or higher", "README.md")
    assert float(match.group(1)) <= stats["min_rate"]
    assert float(match.group(1)) == round(stats["min_rate"])


def test_readme_status_badge_matches_the_summary_csv(stats):
    """The badge quotes the headline number too, in URL-escaped form."""
    match = _search(r"status-(\d+\.\d+)%25%20held--out%20success", "README.md")
    assert float(match.group(1)) == round(stats["mean"], 1)
