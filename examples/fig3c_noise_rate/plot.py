"""Plot Fig 3(c): SA and LF1 success rate and similarity vs noise rate tau.

Requires res_published/summary.json.
"""
import json, os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_FILE = os.path.join(SCRIPT_DIR, 'res_published', 'summary.json')
FIG_DIR = os.path.join(SCRIPT_DIR, 'fig')
os.makedirs(FIG_DIR, exist_ok=True)

with open(SUMMARY_FILE, 'r') as f:
    d = json.load(f)

err_list = d['err_list']
sa_success = d['sa_success']
lf1_success = d['lf1_success']
sa_sim, lf1_sim = d['sa_sim'], d['lf1_sim']

plt.rcParams.update({'font.size': 14})

fig, ax1 = plt.subplots(figsize=(6, 4.5))
ax1.set_xlabel(r'Noise Rate $\tau$')
ax1.set_ylabel('Success Rate', color='#0c5da5')
l1 = ax1.plot(err_list, sa_success, marker='o', ls='-', color='#0c5da5', label='SA success')
l2 = ax1.plot(err_list, lf1_success, marker='o', ls='--', color='#0c5da5', label='LF1 success', fillstyle='none')
ax1.tick_params(axis='y', labelcolor='#0c5da5')
ax1.set_ylim(0, 1.1)

ax2 = ax1.twinx()
ax2.set_ylabel('Average Similarity', color='#548235')
l3 = ax2.plot(err_list, sa_sim, marker='s', ls='-', color='#548235', label='SA similarity')
l4 = ax2.plot(err_list, lf1_sim, marker='s', ls='--', color='#548235', label='LF1 similarity', fillstyle='none')
ax2.tick_params(axis='y', labelcolor='#548235')
ax2.set_ylim(0, 1.1)

lines = l1 + l2 + l3 + l4
labels = [ln.get_label() for ln in lines]
ax1.legend(lines, labels, loc='best')

fig.tight_layout()
out_path = os.path.join(FIG_DIR, 'fig3c.pdf')
fig.savefig(out_path, dpi=300)
fig.savefig(out_path.replace('.pdf', '.png'), dpi=300)
plt.close(fig)
print(f'Saved {out_path}')