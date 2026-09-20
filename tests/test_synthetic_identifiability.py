from experiments.synthetic_identifiability import DiagnosticConfig, run_diagnostic


def test_constructive_coverage_diagnostic_is_exact_at_full_coverage():
    config = DiagnosticConfig(num_nodes=8, num_blocks=4, seeds=2, samples=32)
    rows, payload = run_diagnostic(config)

    by_coverage = {item["coverage"]: item for item in payload["summary"]}
    assert by_coverage[1.0]["overall_edge_f1_mean"] == 1.0
    assert by_coverage[1.0]["within_covered_edge_f1_mean"] == 1.0
    assert by_coverage[1.0]["covered_intervention_alignment_error_mean"] < 1e-12
    assert max(row["observational_fit_rmse"] for row in rows) < 1e-12
    assert by_coverage[0.0]["overall_edge_f1_mean"] < by_coverage[1.0]["overall_edge_f1_mean"]


def test_partial_coverage_preserves_covered_subgraph():
    config = DiagnosticConfig(num_nodes=12, num_blocks=4, seeds=1, samples=16)
    _, payload = run_diagnostic(config)
    partial = [item for item in payload["summary"] if item["coverage"] in {0.25, 0.5}]
    assert all(item["within_covered_edge_f1_mean"] == 1.0 for item in partial)
