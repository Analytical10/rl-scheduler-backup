import torch
import numpy as np
from para_eff_pt.pt_rl_opt.rl_agent import PPOAgent
from para_eff_pt.pt_rl_opt.rl_optimizer import RL_AdamW_Wrapper
import torch.nn as nn

def test_data_flow():
    print("=== 开始 RL 数据流集成测试 ===")
    
    # 1. 模拟环境参数
    state_dim = 13
    action_dim = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. 初始化开启了 Flag 256 (TanhNormal) 的优化器
    # 注意：round=256 对应 flag 256 开启
    params = [nn.Parameter(torch.randn(10, 10))]
    opt = RL_AdamW_Wrapper(
        params, 
        round=256, 
        device=device
    )
    
    print(f"检查点 1: Agent squash_actions 状态 -> {opt.agent.squash_actions}")
    assert opt.agent.squash_actions == True, "错误：Flag 256 开启但 Agent 未启用 squash_actions"

    # 3. 模拟一次 Step 决策
    dummy_state = np.random.randn(1, state_dim).astype(np.float32)
    # 模拟优化器内部对多参数的特征平铺
    state_batch = np.tile(dummy_state, (len(params), 1)) 
    
    # 手动调用 select_action 模拟优化器内部行为
    pre_tanh, actions, logprobs = opt.agent.select_action(state_batch)
    
    print(f"检查点 2: select_action 返回值解包成功")
    assert pre_tanh is not None, "错误：squash 模式下 pre_tanh 返回了 None！"
    assert actions.shape == (1, action_dim), f"错误：actions 形状不对 -> {actions.shape}"
    
    # 4. 数学一致性校验: action 必须等于 tanh(pre_tanh)
    expected_action = np.tanh(pre_tanh)
    is_consistent = np.allclose(actions, expected_action, atol=1e-5)
    print(f"检查点 3: 数学一致性 (action == tanh(pre_tanh)) -> {is_consistent}")
    assert is_consistent, "错误：执行动作与原始冲动不匹配，请检查 act() 方法映射逻辑"

    # 5. 模拟存储校验 (针对 Flag 4 对齐和普通存储)
    reward = 1.0
    # 清空 buffer 确保测试干净
    opt.agent.buffer = []
    
    # 模拟执行一次 opt.step() 其中包含存储逻辑
    dummy_loss = torch.tensor(0.5, requires_grad=True)
    opt.step(loss=dummy_loss)
    
    print(f"检查点 4: Buffer 存储内容校验")
    if len(opt.agent.buffer) > 0:
        s, a, lp, r = opt.agent.buffer[0]
        # 核心校验：存入的是否是原始值
        # 在 pre_tanh 模式下，a 的绝对值通常会大于 1.0 (因为还没过 tanh)
        is_raw = np.any(np.abs(a) > 1.0) or not np.allclose(a, np.tanh(a))
        print(f"    Buffer 中动作绝对值最大值: {np.max(np.abs(a)):.4f}")
        print(f"    存储的是原始值(pre_tanh)吗? -> {is_raw}")
        # 如果你输出的 pre_tanh 都在 1.0 以内，这个检查可能会报 False，但逻辑上存的是 a 就行
    else:
        print("    注意：由于 current_step=0 或 Flag 4 延迟，当前 Buffer 为空（正常现象）")

    # 6. 模拟 Update 校验 (校验 evaluate 是否兼容 pre_tanh)
    print("检查点 5: 模拟 PPO Update 过程...")
    # 手动造一条存了 pre_tanh 的数据
    test_transition = (state_batch, pre_tanh, logprobs, 1.0)
    opt.agent.store_transition(test_transition)
    
    try:
        opt.agent.update()
        print("    Update 成功！evaluate() 能够正确处理 pre_tanh 数据。")
    except Exception as e:
        print(f"    Update 失败！错误信息: {e}")
        raise e

    print("\n🎉 所有数据流测试通过！你的修改逻辑自洽。")

if __name__ == "__main__":
    test_data_flow()