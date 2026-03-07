#!/usr/bin/env python3
"""
绘制 6h_vs_8z 地图下各算法的 test_battle_won_mean 对比图
排除 mae 算法，并添加曲线平滑功能
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.ndimage import uniform_filter1d


def smooth_curve(values, window_size=20):
    """使用移动平均平滑曲线"""
    if len(values) < window_size:
        return values
    # 使用 uniform_filter1d 进行平滑
    smoothed = uniform_filter1d(values, size=window_size, mode='nearest')
    return smoothed


def get_latest_run(algorithm_dir):
    """获取算法目录下最新的运行（最大的run id）"""
    run_dirs = glob.glob(os.path.join(algorithm_dir, "*"))
    if not run_dirs:
        return None
    
    # 提取数字id并找到最大的
    run_ids = []
    for d in run_dirs:
        try:
            run_id = int(os.path.basename(d))
            run_ids.append((run_id, d))
        except ValueError:
            continue
    
    if not run_ids:
        return None
    
    # 返回最新（最大id）的运行目录
    return max(run_ids, key=lambda x: x[0])[1]


def load_algorithm_data(base_dir, algorithm_name):
    """加载指定算法的最新运行数据"""
    algo_dir = os.path.join(base_dir, algorithm_name)
    if not os.path.exists(algo_dir):
        print(f"警告: 算法目录不存在 {algo_dir}")
        return None, None
    
    latest_run = get_latest_run(algo_dir)
    if not latest_run:
        print(f"警告: 算法 {algorithm_name} 没有找到运行记录")
        return None, None
    
    info_file = os.path.join(latest_run, "info.json")
    if not os.path.exists(info_file):
        print(f"警告: info.json 不存在 {info_file}")
        return None, None
    
    try:
        with open(info_file, 'r') as f:
            data = json.load(f)
        
        # 获取 test_battle_won_mean 和对应的时间步
        test_won = data.get('test_battle_won_mean', [])
        test_won_T = data.get('test_battle_won_mean_T', [])
        
        if not test_won or not test_won_T:
            print(f"警告: 算法 {algorithm_name} 没有 test_battle_won_mean 数据")
            return None, None
        
        return np.array(test_won_T), np.array(test_won)
    except Exception as e:
        print(f"错误: 加载 {algorithm_name} 数据失败: {e}")
        return None, None


def plot_algorithm_comparison(base_dir, output_path=None, smooth_window=20):
    """绘制所有算法的对比图"""
    
    # 获取所有算法目录
    algo_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) 
                 if os.path.isdir(d)]
    
    algorithms = []
    for d in algo_dirs:
        algo_name = os.path.basename(d)
        # 排除非算法目录（如 .git, __pycache__ 等）
        # 排除包含 'mae' 的算法
        if (not algo_name.startswith('.') and 
            not algo_name.startswith('_') and
            'mae' not in algo_name.lower()):
            algorithms.append(algo_name)
    
    if not algorithms:
        print(f"错误: 在 {base_dir} 中没有找到算法目录")
        return
    
    print(f"发现 {len(algorithms)} 个算法（已排除 mae）: {algorithms}")
    
    # 设置图形
    plt.figure(figsize=(14, 8))
    
    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    # 加载并绘制每个算法的数据
    for i, algo_name in enumerate(sorted(algorithms)):
        timesteps, values = load_algorithm_data(base_dir, algo_name)
        
        if timesteps is not None and values is not None:
            # 平滑曲线
            smoothed_values = smooth_curve(values, window_size=smooth_window)
            
            # 绘制平滑曲线
            plt.plot(timesteps, smoothed_values, 
                    label=algo_name, 
                    color=colors[i],
                    linewidth=2.5,
                    alpha=0.9)
            
            # 可选：绘制原始数据的半透明散点
            # plt.scatter(timesteps, values, color=colors[i], alpha=0.1, s=5)
            
            # 打印统计信息
            print(f"{algo_name}: {len(values)} 个数据点, "
                  f"最终胜率: {values[-1]:.3f}, "
                  f"最高胜率: {max(values):.3f}")
    
    # 设置图形属性
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Test Battle Won Mean', fontsize=12)
    plt.title('Algorithm Comparison on 6h_vs_8z\nTest Battle Won Mean vs Time Steps (Smoothed)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 设置y轴范围在0-1之间
    plt.ylim(-0.05, 1.05)
    
    # 自动调整x轴
    plt.tight_layout()
    
    # 保存图形
    if output_path is None:
        output_path = os.path.join(base_dir, 'algorithm_comparison.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存到: {output_path}")
    print(f"平滑窗口大小: {smooth_window}")
    
    # 显示图形
    plt.show()


if __name__ == "__main__":
    # 基础目录
    base_dir = "/home/liwenlei/pymarl-rlc-main/results/sacred/6h_vs_8z"
    
    # 检查目录是否存在
    if not os.path.exists(base_dir):
        print(f"错误: 目录不存在 {base_dir}")
        exit(1)
    
    # 绘制对比图（平滑窗口大小为20）
    plot_algorithm_comparison(base_dir, smooth_window=20)
