"""Focused regression tests for the causal-field core semantics."""

import inspect

import torch

from metacausal_field import (
    CausalFieldConfig,
    DirectionalInfluenceFunction,
    GaussianInterpolation,
    MetaCausalLoss,
    MultimodalCausalField,
    compute_causal_effects,
)


def _identity_influence_module() -> DirectionalInfluenceFunction:
    module = DirectionalInfluenceFunction(
        feature_dim=4,
        num_heads=1,
        dropout=0.0,
    )
    with torch.no_grad():
        module.query_proj.weight.zero_()
        module.query_proj.bias.zero_()
        module.key_proj.weight.zero_()
        module.key_proj.bias.zero_()
        module.value_proj.weight.copy_(torch.eye(4))
        module.value_proj.bias.zero_()
        module.out_proj.weight.copy_(torch.eye(4))
        module.out_proj.bias.zero_()
        for parameter in module.sparsity_gate.parameters():
            parameter.zero_()
        # Positive x displacement raises source->target logit.
        module.directional_bias_proj.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
    return module


def test_directional_bias_is_pair_dependent_and_uses_source_target_order():
    module = _identity_influence_module()
    field = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])

    # The historical two-tensor API remains available.
    influence, output = module(field, spatial_positions=positions)
    components = module(field, spatial_positions=positions, return_components=True)

    directional_bias = components['directional_bias'][0, 0]
    assert directional_bias[0, 1] > 0  # displacement 0 -> 1 is +x
    assert directional_bias[1, 0] < 0  # displacement 1 -> 0 is -x
    assert influence[0, 0, 1] > influence[0, 1, 0]
    assert not any(name == 'direction_bias' for name, _ in module.named_parameters())

    legacy_state = module.state_dict()
    legacy_state['direction_bias'] = torch.zeros(1, 1, 1, 1)
    _identity_influence_module().load_state_dict(legacy_state, strict=True)

    # G[i,j] means source i -> target j, hence the target update is G^T V.
    expected = torch.bmm(influence.transpose(1, 2), field)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(influence, components['influence_matrix'])


def test_causal_effect_utility_reads_outgoing_source_row():
    graph = torch.tensor(
        [
            [0.0, 0.75, 0.25],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    effects = compute_causal_effects(torch.zeros(3, 1), graph, intervention_position=0)
    torch.testing.assert_close(effects, torch.tensor([0.0, 0.75, 0.25]))


def test_sparsity_loss_is_nonconstant_for_row_stochastic_graphs():
    config = CausalFieldConfig(feature_dim=4, num_heads=1)
    loss_fn = MetaCausalLoss(config)
    field = torch.zeros(1, 2, 2, 4)
    uniform = torch.full((1, 4, 4), 0.25)
    concentrated = torch.eye(4).unsqueeze(0)

    uniform_loss, _ = loss_fn.structure_regularization(uniform, field)
    concentrated_loss, _ = loss_fn.structure_regularization(concentrated, field)

    # Both matrices have the same L1 mean; entropy correctly distinguishes them.
    assert torch.isclose(uniform.abs().mean(), concentrated.abs().mean())
    assert uniform_loss > concentrated_loss


def test_pre_normalization_gate_regularizer_has_parameter_gradients():
    torch.manual_seed(7)
    module = DirectionalInfluenceFunction(feature_dim=4, num_heads=1, dropout=0.0)
    field = torch.randn(2, 4, 4)
    components = module(field, return_components=True)
    loss_fn = MetaCausalLoss(CausalFieldConfig(feature_dim=4, num_heads=1))

    sparsity, _ = loss_fn.structure_regularization(
        components['influence_matrix'],
        field,
        sparsity_source=components['sparsity_gate'],
    )
    sparsity.backward()

    gate_gradient = module.sparsity_gate[2].weight.grad
    assert gate_gradient is not None
    assert torch.isfinite(gate_gradient).all()
    assert gate_gradient.abs().sum() > 0


def test_optional_factor_supervision_reaches_factor_localizer():
    torch.manual_seed(11)
    config = CausalFieldConfig(
        spatial_size=(2, 2),
        feature_dim=8,
        num_heads=2,
        num_propagation_steps=1,
        dropout=0.0,
    )
    model = MultimodalCausalField(
        config,
        visual_encoder=None,
        num_factors=3,
        enable_language=False,
    )
    outputs = model(torch.randn(2, 4, 8))
    targets = {
        'factor_spatial_targets': torch.tensor([[0, 1, 2], [3, 2, 1]]),
        # Rows and columns use source -> target order.
        'factor_graph_targets': torch.tensor(
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        ),
    }

    losses = MetaCausalLoss(config)(outputs, targets)
    assert 'factor_localization' in losses
    assert 'factor_graph' in losses
    losses['total'].backward()

    gradient = model.factor_localizer.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_generation_starts_from_bos_without_reference_answer_prefixes():
    torch.manual_seed(13)
    config = CausalFieldConfig(
        spatial_size=(2, 2),
        feature_dim=8,
        num_heads=2,
        num_propagation_steps=1,
        dropout=0.0,
        vocab_size=12,
        max_text_length=8,
    )
    model = MultimodalCausalField(
        config,
        visual_encoder=None,
        vocab_size=config.vocab_size,
        enable_language=True,
    ).eval()
    with torch.no_grad():
        model.text_decoder.lm_head.weight.zero_()
        model.text_decoder.lm_head.bias.zero_()
        model.text_decoder.lm_head.bias[config.eos_token_id] = 10.0

    assert "decoder_input_ids" not in inspect.signature(model.generate_text).parameters
    assert "decoder_input_ids" not in inspect.signature(model.generate_counterfactual_text).parameters

    visual = torch.randn(2, 4, 8)
    prompts = torch.tensor([[1, 4, 2, 0], [1, 5, 2, 0]])
    factual = model.generate_text(visual, input_ids=prompts, max_new_tokens=5)
    assert factual["generated_ids"].shape == (2, 1)
    assert factual["generated_ids"].eq(config.eos_token_id).all()
    assert "lm_logits" not in factual

    counterfactual = model.generate_counterfactual_text(
        visual,
        intervention_type="modify",
        intervention_params={
            "position": torch.tensor([[0.25, 0.25], [0.75, 0.75]]),
            "direction": torch.ones(2, config.feature_dim),
            "radius": 0.2,
        },
        input_ids=prompts,
        num_rollout_steps=1,
        max_new_tokens=5,
    )
    assert counterfactual["generated_ids_counterfactual"].shape == (2, 1)
    assert counterfactual["generated_ids_counterfactual"].eq(config.eos_token_id).all()
    assert "lm_logits_counterfactual" not in counterfactual


def test_gaussian_sigma_uses_grid_units_on_normalized_coordinates():
    size = 14
    y_grid = torch.linspace(0, 1, size)
    x_grid = torch.linspace(0, 1, size)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
    positions = torch.stack([xx, yy], dim=-1).reshape(1, size * size, 2)
    features = torch.zeros(1, size * size, 1)
    features[0, 0, 0] = 1.0

    local = GaussianInterpolation(sigma=1.0, sigma_in_grid_units=True)
    field = local(
        features,
        positions,
        query_positions=positions,
        query_grid_shape=(size, size),
    )

    assert field[0, 0, 0, 0] > 0.1
    assert field[0, -1, -1, 0] < 1e-20


def test_all_missing_factor_labels_do_not_create_nan_cross_entropy():
    config = CausalFieldConfig(feature_dim=4, num_heads=1, num_factor_classes=3)
    criterion = MetaCausalLoss(config)
    outputs = {
        "factor_logits": torch.randn(2, 3, 3, requires_grad=True),
        "field": torch.zeros(2, 1, 1, 4),
        "influence_matrix": torch.ones(2, 1, 1),
    }
    losses = criterion(outputs, {"factor_targets": torch.full((2, 3), -100)})

    assert "factor" not in losses
    assert torch.isfinite(losses["total"])
