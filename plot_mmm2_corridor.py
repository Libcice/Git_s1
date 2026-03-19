import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def find_latest_experiment(base_path):
    """找到最新的实验目录"""
    exp_dirs = glob.glob(os.path.join(base_path, "[0-9]*"))
    if not exp_dirs:
        return None
    # 按目录名数字排序，取最大的
    exp_dirs.sort(key=lambda x: int(os.path.basename(x)))
    return exp_dirs[-1]


def load_info_json(exp_path):
    """加载info.json文件"""
    info_path = os.path.join(exp_path, "info.json")
    if not os.path.exists(info_path):
        return None
    with open(info_path, 'r') as f:
        return json.load(f)


def smooth_curve(data, window=10):
    """平滑曲线"""
    if len(data) < window:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    return smoothed


def plot_win_rate():
    """绘制MMM2和Corridor的胜率曲线"""
    
    # 配置
    maps = {
        "MMM2": "/home/liwenlei/pymarl-rlc-main/results/sacred/MMM2/qmix_history_token_belief",
        "Corridor": "/home/liwenlei/pymarl-rlc-main/results/sacred/corridor/qmix_history_token_belief"
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"MMM2": "#2E86AB", "Corridor": "#A23B72"}
    
    for idx, (map_name, base_path) in enumerate(maps.items()):
        ax = axes[idx]
        
        # 找到最新实验
        latest_exp = find_latest_experiment(base_path)
        if not latest_exp:
            print(f"未找到 {map_name} 的实验结果")
            continue
        
        print(f"{map_name}: 使用实验 {os.path.basename(latest_exp)}")
        
        # 加载数据
        info = load_info_json(latest_exp)
        if not info:
            print(f"无法加载 {map_name} 的info.json")
            continue
        
        # 提取胜率数据
        if "test_battle_won_mean" in info and "test_battle_won_mean_T" in info:
            win_rates = info["test_battle_won_mean"]
            timesteps = info["test_battle_won_mean_T"]
            
            if len(win_rates) == 0:
                print(f"{map_name}: 没有胜率数据")
                continue
            
            # 平滑曲线
            smoothed_win_rates = smooth_curve(win_rates, window=5)
            
            # 绘制
            ax.plot(timesteps, win_rates, alpha=0.3, color=colors[map_name], label="Raw")
            ax.plot(timesteps, smoothed_win_rates, linewidth=2, color=colors[map_name], label="Smoothed")
            
            # 设置标签
            ax.set_xlabel("Timesteps", fontsize=12)
            ax.set_ylabel("Test Battle Won Mean", fontsize=12)
            ax.set_title(f"{map_name} - QMIX History Token Belief", fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.05, 1.05])
            
            # 添加最终胜率标注
            final_win_rate = win_rates[-1]
            ax.axhline(y=final_win_rate, color='red', linestyle='--', alpha=0.5)
            ax.text(timesteps[-1], final_win_rate, f'  Final: {final_win_rate:.2%}', 
                   fontsize=10, va='bottom', ha='right', color='red')
            
            print(f"{map_name}: 最终胜率 = {final_win_rate:.2%}, 数据点 = {len(win_rates)}")
        else:
            print(f"{map_name}: info.json中缺少test_battle_won_mean数据")
    
    plt.tight_layout()
    
    # 保存图片
    output_path = "/home/liwenlei/pymarl-rlc-main/mmm2_corridor_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图片已保存: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    plot_win_rate()
