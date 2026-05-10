
import json
import numpy as np
import matplotlib.pyplot as plt

# 读取两份 JSON
with open('training_data_20260118_195832.json', 'r') as f:
    ours_data = json.load(f)
with open('training_data_20260216_204115.json', 'r') as f:
    baseline_data = json.load(f)

# 提取 episode_rewards
ours_rewards = np.array(ours_data['episode_rewards'])
baseline_rewards = np.array(baseline_data['episode_rewards'])
n_episodes = len(ours_rewards)
episodes = np.arange(n_episodes)

# 计算平滑曲线
window = 20
smooth_ours = np.convolve(ours_rewards, np.ones(window)/window, mode='valid')
smooth_base = np.convolve(baseline_rewards, np.ones(window)/window, mode='valid')
smooth_eps = episodes[:len(smooth_ours)]

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['axes.edgecolor'] = 'none'
plt.rcParams['grid.color'] = '#e0e0e0'

colors = {'ours': '#2E86AB', 'baseline': '#A23B72'}

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# 原始值：细线 + 半透明
ax.plot(episodes, ours_rewards, color=colors['ours'], alpha=0.3,
        linewidth=0.8, label='_nolegend_')
ax.plot(episodes, baseline_rewards, color=colors['baseline'], alpha=0.3,
        linewidth=0.8, label='_nolegend_')

# 平滑值：粗线实线
ax.plot(smooth_eps, smooth_ours, color=colors['ours'],
        linewidth=2.5, label='Ours (with MPC)')
ax.plot(smooth_eps, smooth_base, color=colors['baseline'],
        linewidth=2.5, label='Baseline (w/o MPC)')

# 标题和标签
ax.set_title('Training Reward Curves', fontsize=16, fontweight='bold')
ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Episode Return', fontsize=14)
ax.set_ylim(-150, 150)
ax.legend(loc='best', framealpha=0.9, fontsize=14)
ax.grid(True, linestyle='-', alpha=0.4)

plt.tight_layout()
plt.savefig('/mnt/agents/output/training_reward_single.png', bbox_inches='tight', dpi=300)
plt.show()

print("✅ 图片已保存")
