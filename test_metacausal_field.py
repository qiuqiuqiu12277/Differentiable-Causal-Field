"""
简单测试脚本，验证 MetaCausalField 的基本功能
"""

import torch
import sys
from metacausal_field import (
    CausalFieldConfig,
    MultimodalCausalField,
    MetaCausalLoss
)

def test_basic_import():
    """测试基本导入"""
    print("✅ 测试 1: 基本导入 - 通过")
    return True

def test_config():
    """测试配置类"""
    config = CausalFieldConfig(
        feature_dim=256,
        num_heads=4,
        num_propagation_steps=2
    )
    
    assert config.feature_dim == 256
    assert config.num_heads == 4
    assert config.num_propagation_steps == 2
    
    print("✅ 测试 2: 配置类 - 通过")
    return True

def test_model_initialization():
    """测试模型初始化"""
    try:
        config = CausalFieldConfig(
            feature_dim=128,  # 使用较小的维度快速测试
            num_heads=4,
            num_propagation_steps=2
        )
        
        # 不使用 visual_encoder，设为 None
        model = MultimodalCausalField(config, visual_encoder=None)
        assert model.propagation.influence_fn.num_heads == config.num_heads
        
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print("✅ 测试 3: 模型初始化 - 通过")
        return True
    except Exception as e:
        print(f"❌ 测试 3: 模型初始化 - 失败: {e}")
        return False

def test_forward_pass():
    """测试前向传播"""
    try:
        config = CausalFieldConfig(
            feature_dim=64,  # 使用最小的维度快速测试
            num_heads=2,
            num_propagation_steps=2
        )
        
        model = MultimodalCausalField(config, visual_encoder=None)
        model.eval()
        
        # 创建模拟输入
        batch_size = 2
        num_patches = 49  # 7x7
        feature_dim = 64
        
        visual_features = torch.randn(batch_size, num_patches, feature_dim)
        
        # 前向传播
        with torch.no_grad():
            outputs = model(visual_features)
        
        # 检查输出
        assert 'field' in outputs
        assert 'influence_matrix' in outputs
        assert 'influence_matrices' in outputs
        assert 'field_trajectory' in outputs
        assert 'readout' in outputs
        assert 'initial_field' in outputs
        
        # 检查形状
        B, H, W, C = outputs['field'].shape
        assert B == batch_size
        assert (H, W) == config.spatial_size
        assert C == feature_dim
        
        B2, N, N2 = outputs['influence_matrix'].shape
        assert B2 == batch_size
        assert N == N2 == config.spatial_size[0] * config.spatial_size[1]
        assert (outputs['influence_matrix'] >= 0).all()  # 影响矩阵应该非负
        assert len(outputs['field_trajectory']) == config.num_propagation_steps + 1
        assert len(outputs['influence_matrices']) == config.num_propagation_steps
        
        B3, C2 = outputs['readout'].shape
        assert B3 == batch_size
        assert C2 == feature_dim
        
        print("✅ 测试 4: 前向传播 - 通过")
        print(f"  - 场形状: {outputs['field'].shape}")
        print(f"  - 影响矩阵形状: {outputs['influence_matrix'].shape}")
        print(f"  - Readout形状: {outputs['readout'].shape}")
        return True
    except Exception as e:
        print(f"❌ 测试 4: 前向传播 - 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """测试损失计算"""
    try:
        config = CausalFieldConfig(
            feature_dim=32,
            num_heads=2,
            num_propagation_steps=2
        )
        
        # 创建模拟输出（包含必要的字段）
        batch_size = 2
        num_patches = 16  # 4x4
        vocab_size = 100
        feature_dim = 32
        
        # 确保所有tensor都有梯度
        outputs = {
            'field': torch.randn(batch_size, 4, 4, feature_dim, requires_grad=True),
            'influence_matrix': torch.rand(batch_size, num_patches, num_patches, requires_grad=True),
            'readout': torch.randn(batch_size, feature_dim, requires_grad=True),
            'initial_field': torch.randn(batch_size, 4, 4, feature_dim, requires_grad=True),
            'field_trajectory': [
                torch.randn(batch_size, 4, 4, feature_dim, requires_grad=True)
                for _ in range(3)
            ],
            'influence_matrices': [
                torch.rand(batch_size, num_patches, num_patches, requires_grad=True)
                for _ in range(2)
            ],
            'lm_logits': torch.randn(batch_size, 10, vocab_size, requires_grad=True)  # 添加语言模型logits
        }
        
        targets = {'lm_targets': torch.randint(0, vocab_size, (batch_size, 10))}
        
        # 创建损失函数
        loss_fn = MetaCausalLoss(config)
        
        # 计算损失（forward 只接受 outputs 和 targets 两个参数）
        loss = loss_fn(outputs, targets)
        
        # 检查损失
        assert isinstance(loss, dict)
        assert 'total' in loss
        # 检查损失值是否合理
        assert loss['total'] >= 0, "总损失应该是非负的"
        
        print("✅ 测试 5: 损失计算 - 通过")
        print(f"  - 总损失: {loss['total'].item():.4f}")
        print(f"  - LM损失: {loss.get('lm', 0):.4f}")
        print(f"  - 一致性损失: {loss.get('consistency', 0):.4f}")
        print(f"  - 反事实损失: {loss.get('counterfactual', 0):.4f}")
        print(f"  - 稀疏性损失: {loss.get('sparsity', 0):.4f}")
        print(f"  - 平滑性损失: {loss.get('smoothness', 0):.4f}")
        return True
    except Exception as e:
        print(f"❌ 测试 5: 损失计算 - 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_counterfactual():
    """测试反事实推理"""
    try:
        config = CausalFieldConfig(
            feature_dim=32,
            num_heads=2,
            num_propagation_steps=2
        )
        
        model = MultimodalCausalField(config, visual_encoder=None)
        model.eval()
        
        # 创建模拟输入
        batch_size = 1
        num_patches = 16
        feature_dim = 32
        
        visual_features = torch.randn(batch_size, num_patches, feature_dim)
        
        # 测试移除干预
        with torch.no_grad():
            cf_outputs = model.counterfactual_forward(
                visual_features,
                intervention_type='remove',
                intervention_params={
                    'position': torch.tensor([[0.5, 0.5]]),
                    'radius': 0.2
                },
                num_rollout_steps=2
            )
            inferred_cf_outputs = model.counterfactual_forward(
                visual_features,
                intervention_type='remove',
                intervention_params={'radius': 0.2},
                num_rollout_steps=1
            )
        
        # 检查输出
        assert 'field_factual' in cf_outputs
        assert 'field_counterfactual' in cf_outputs
        assert 'intervention_mask' in cf_outputs
        assert 'factual_trajectory' in cf_outputs
        assert 'counterfactual_trajectory' in cf_outputs
        assert 'delta_trajectory' in cf_outputs
        assert 'cf_pred_trajectory' in cf_outputs
        
        # 检查形状
        assert cf_outputs['field_factual'].shape == cf_outputs['field_counterfactual'].shape
        
        # 检查干预掩码
        assert (cf_outputs['intervention_mask'] >= 0).all()
        assert (cf_outputs['intervention_mask'] <= 1).all()
        assert len(cf_outputs['factual_trajectory']) == config.num_propagation_steps + 1
        assert len(cf_outputs['counterfactual_trajectory']) == 3
        assert len(cf_outputs['cf_pred_trajectory']) == 2
        assert inferred_cf_outputs['intervention_mask'].shape == cf_outputs['intervention_mask'].shape
        
        print("✅ 测试 6: 反事实推理 - 通过")
        print(f"  - 事实场形状: {cf_outputs['field_factual'].shape}")
        print(f"  - 反事实场形状: {cf_outputs['field_counterfactual'].shape}")
        print(f"  - 干预掩码范围: [{cf_outputs['intervention_mask'].min():.2f}, {cf_outputs['intervention_mask'].max():.2f}]")
        return True
    except Exception as e:
        print(f"❌ 测试 6: 反事实推理 - 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factor_level_graph():
    """测试 patch influence 能聚合到命名因子空间"""
    try:
        config = CausalFieldConfig(
            feature_dim=32,
            num_heads=2,
            num_propagation_steps=1,
        )
        model = MultimodalCausalField(config, visual_encoder=None, num_factors=3)
        model.eval()
        visual_features = torch.randn(2, 16, 32)
        with torch.no_grad():
            outputs = model(visual_features)

        assert 'factor_spatial_attention' in outputs
        assert 'factor_influence_matrix' in outputs
        assert outputs['factor_spatial_attention'].shape == (2, 3, config.spatial_size[0] * config.spatial_size[1])
        assert outputs['factor_influence_matrix'].shape == (2, 3, 3)
        diag = outputs['factor_influence_matrix'].diagonal(dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)

        print("✅ 测试 7: 因子级图聚合 - 通过")
        return True
    except Exception as e:
        print(f"❌ 测试 7: 因子级图聚合 - 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("=" * 60)
    print("MetaCausalField 功能测试")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_import,
        test_config,
        test_model_initialization,
        test_forward_pass,
        test_loss_computation,
        test_counterfactual,
        test_factor_level_graph,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
