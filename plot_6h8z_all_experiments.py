import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def find_all_experiments(base_path):
    """找到所有实验目录"""
    exp_dirs = glob.glob(os.path.join(base_path, "[0-9]*"))
    # 按目录名数字排序
    exp_dirs.sort(key=lambda x: int(os.path.basename(x)))
    return exp_dirs


def load_info_json(exp_path):
    """加载info.json文件"""
    info_path = os.path.join(exp_path, "info.json")
    if not os.path.exists(info_path):
        return None
    try:
        with open(info_path, 'r') as f:
            return json.load(f)
    except:
        return None


def load_config(exp_path):
    """加载config.json获取实验配置"""
    config_path = os.path.join(exp_path, "config.json")
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return {}


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


def plot_all_experiments():
    """绘制6h_vs_8z所有实验的胜率对比"""
    
    base_path = "/home/liwenlei/pymarl-rlc-main/results/sacred/6h_vs_8z/qmix_history_token_belief"
    
    # 找到所有实验
    exp_dirs = find_all_experiments(base_path)
    if not exp_dirs:
        print(f"未找到实验结果: {base_path}")
        return
    
    print(f"找到 {len(exp_dirs)} 个实验")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_dirs)))
    
    legend_entries = []
    final_results = []
    
    for idx, exp_dir in enumerate(exp_dirs):
        exp_id = os.path.basename(exp_dir)
        info = load_info_json(exp_dir)
        config = load_config(exp_dir)
        
        if not info:
            print(f"实验 {exp_id}: 无法加载info.json")
            continue
        
        # 提取配置信息
        history_steps = config.get("history_steps", "?")
        
        # 提取胜率数据
        if "test_battle_won_mean" in info and "test_battle_won_mean_T" in info:
            win_rates = info["test_battle_won_mean"]
            timesteps = info["test_battle_won_mean_T"]
            
            if len(win_rates) == 0:
                print(f"实验 {exp_id}: 没有胜率数据")
                continue
            
            # 平滑曲线
            smoothed_win_rates = smooth_curve(win_rates, window=5)
            
            # 绘制平滑曲线
            line, = ax.plot(timesteps, smoothed_win_rates, 
                           linewidth=2, 
                           color=colors[idx],
                           label=f"Exp {exp_id} (steps={history_steps})")
            
            # 绘制原始数据（透明度较低）
            ax.plot(timesteps, win_rates, 
                   alpha=0.2, 
                   color=colors[idx],
                   linewidth=1)
            
            final_win_rate = win_rates[-1]
            max_win_rate = max(win_rates)
            final_results.append({
                "exp_id": exp_id,
                "history_steps": history_steps,
                "final": final_win_rate,
                "max": max_win_rate,
                "points": len(win_rates)
            })
            
            print(f"实验 {exp_id} (steps={history_steps}): 最终胜率={final_win_rate:.2%}, 最大胜率={max_win_rate:.2%}, 数据点={len(win_rates)}")
    
    # 设置图表属性
    ax.set_xlabel("Timesteps", fontsize=14)
    ax.set_ylabel("Test Battle Won Mean", fontsize=14)
    ax.set_title("6h_vs_8z - QMIX History Token Belief\nAll Experiments Comparison", 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    # 图例放在左上角
    legend = ax.legend(loc='upper left', fontsize=9, ncol=1, 
              bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
    
    # 添加结果表格在图例下方（左上角）
    if final_results:
        table_text = "Final Results:\n"
        table_text += "-" * 35 + "\n"
        table_text += f"{'Exp':<6}{'Steps':<8}{'Final':<10}{'Max':<10}\n"
        table_text += "-" * 35 + "\n"
        for r in final_results:
            table_text += f"{r['exp_id']:<6}{r['history_steps']:<8}{r['final']:<10.2%}{r['max']:<10.2%}\n"
        
        # 获取图例的高度，在图例下方放置表格
        ax.text(0.02, 0.55, table_text, 
               transform=ax.transAxes,
               fontsize=8,
               verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片
    output_path = "/home/liwenlei/pymarl-rlc-main/6h8z_all_experiments_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图片已保存: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    plot_all_experiments()
