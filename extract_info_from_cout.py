#!/usr/bin/env python3
"""
从cout.txt提取训练数据并生成info.json
"""

import json
import re
import os
import sys


def extract_data_from_cout(cout_path):
    """从cout.txt提取关键指标"""
    
    data = {
        # 训练指标
        "battle_won_mean": [],
        "battle_won_mean_T": [],
        "belief_logvar_mean": [],
        "belief_logvar_mean_T": [],
        "belief_loss": [],
        "belief_loss_T": [],
        "belief_raw_nll": [],
        "belief_raw_nll_T": [],
        "belief_unseen_frac": [],
        "belief_unseen_frac_T": [],
        "belief_weight": [],
        "belief_weight_T": [],
        "ep_length_mean": [],
        "ep_length_mean_T": [],
        "epsilon": [],
        "epsilon_T": [],
        "grad_norm": [],
        "grad_norm_T": [],
        "loss": [],
        "loss_T": [],
        "q_loss": [],
        "q_loss_T": [],
        "q_taken_mean": [],
        "q_taken_mean_T": [],
        "return_mean": [],
        "return_mean_T": [],
        "return_std": [],
        "return_std_T": [],
        "target_mean": [],
        "target_mean_T": [],
        "td_error_abs": [],
        "td_error_abs_T": [],
        # 测试指标
        "test_battle_won_mean": [],
        "test_battle_won_mean_T": [],
        "test_ep_length_mean": [],
        "test_ep_length_mean_T": [],
        "test_return_mean": [],
        "test_return_mean_T": [],
        "test_return_std": [],
        "test_return_std_T": [],
        # episode
        "episode": [],
        "episode_T": [],
    }
    
    with open(cout_path, 'r') as f:
        lines = f.readlines()
    
    current_t_env = None
    
    for line in lines:
        # 查找t_env - 可能在Recent Stats行
        t_env_match = re.search(r't_env:\s+(\d+)', line)
        if t_env_match:
            current_t_env = int(t_env_match.group(1))
        
        if current_t_env is None:
            continue
        
        t_env = current_t_env
        
        # 提取各种指标
        metrics = [
            ("battle_won_mean", r'battle_won_mean:\s+([\d.]+)'),
            ("belief_logvar_mean", r'belief_logvar_mean:\s+([-\d.]+)'),
            ("belief_loss", r'belief_loss:\s+([\d.]+)'),
            ("belief_raw_nll", r'belief_raw_nll:\s+([\d.]+)'),
            ("belief_unseen_frac", r'belief_unseen_frac:\s+([\d.]+)'),
            ("belief_weight", r'belief_weight:\s+([\d.]+)'),
            ("ep_length_mean", r'ep_length_mean:\s+([\d.]+)'),
            ("epsilon", r'epsilon:\s+([\d.]+)'),
            ("grad_norm", r'grad_norm:\s+([\d.]+)'),
            ("loss", r'loss:\s+([\d.]+)'),
            ("q_loss", r'q_loss:\s+([\d.]+)'),
            ("q_taken_mean", r'q_taken_mean:\s+([\d.]+)'),
            ("return_mean", r'return_mean:\s+([\d.]+)'),
            ("return_std", r'return_std:\s+([\d.]+)'),
            ("target_mean", r'target_mean:\s+([\d.]+)'),
            ("td_error_abs", r'td_error_abs:\s+([\d.]+)'),
            ("test_battle_won_mean", r'test_battle_won_mean:\s+([\d.]+)'),
            ("test_ep_length_mean", r'test_ep_length_mean:\s+([\d.]+)'),
            ("test_return_mean", r'test_return_mean:\s+([\d.]+)'),
            ("test_return_std", r'test_return_std:\s+([\d.]+)'),
        ]
        
        for metric_name, pattern in metrics:
            match = re.search(pattern, line)
            if match:
                value = float(match.group(1))
                # 检查是否已经添加过这个时间步的数据
                if len(data[metric_name]) == 0 or data[metric_name + "_T"][-1] != t_env:
                    data[metric_name].append(value)
                    data[metric_name + "_T"].append(t_env)
        
        # 提取episode
        episode_match = re.search(r'Episode:\s+(\d+)', line)
        if episode_match:
            episode = int(episode_match.group(1))
            if len(data["episode"]) == 0 or data["episode_T"][-1] != t_env:
                data["episode"].append(episode)
                data["episode_T"].append(t_env)
    
    return data


def main():
    # 目标实验路径
    exp_path = "/home/liwenlei/pymarl-rlc-main/results/sacred/3s5z_vs_3s6z/qmix_history_token_belief/3"
    
    cout_path = os.path.join(exp_path, "cout.txt")
    info_path = os.path.join(exp_path, "info.json")
    
    if not os.path.exists(cout_path):
        print(f"错误: {cout_path} 不存在")
        return
    
    print(f"正在从 {cout_path} 提取数据...")
    data = extract_data_from_cout(cout_path)
    
    # 打印提取的数据统计
    print("\n提取的数据统计:")
    for key in sorted(data.keys()):
        if not key.endswith("_T"):
            print(f"  {key}: {len(data[key])} 个数据点")
    
    # 保存为info.json
    with open(info_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n数据已保存到: {info_path}")
    print(f"文件大小: {os.path.getsize(info_path)} bytes")


if __name__ == "__main__":
    main()
