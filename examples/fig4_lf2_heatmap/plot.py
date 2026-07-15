"""Plot Fig 4: SA vs LF2 difference heatmap."""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'res_published', 'heatmap_results.npz')
FIG_DIR = os.path.join(SCRIPT_DIR, 'fig')
os.makedirs(FIG_DIR, exist_ok=True)

data = np.load(DATA_FILE)
m_list = data['m_list'].tolist()
err_list = data['err_list'].tolist()
sa_heatmap = data['sa_heatmap']
lf2_heatmap = data['lf2_heatmap']

sa_dict, lf2_dict = {}, {}
for i, m in enumerate(m_list):
    for j, err in enumerate(err_list):
        sa_dict[(m, err)] = float(sa_heatmap[i, j])
        lf2_dict[(m, err)] = float(lf2_heatmap[i, j])

m_inv = m_list[::-1]
diff_mat = np.empty((len(err_list), len(m_inv)))
for i, e in enumerate(err_list):
    for j, f in enumerate(m_inv):
        diff_mat[i, j] = sa_dict.get((f, e), np.nan) - lf2_dict.get((f, e), np.nan)

diff_cmap = sns.diverging_palette(10, 150, s=80, l=50, as_cmap=True, center='light')
plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(diff_mat, xticklabels=m_inv,
            yticklabels=[f'{e:.2f}' for e in err_list],
            cmap=diff_cmap, center=0, vmin=0, vmax=1.0, annot=False, ax=ax,
            cbar_kws={'label': 'Success diff (SA − LF2)'})
ax.invert_yaxis()
ax.set_xlabel(r'Number of Samples $m$')
ax.set_ylabel(r'Noise Rate $\tau$')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
fig.tight_layout()
out_path = os.path.join(FIG_DIR, 'fig4_diff.pdf')
fig.savefig(out_path, dpi=300)
fig.savefig(out_path.replace('.pdf', '.png'), dpi=300)
plt.close(fig)
print(f'Saved {out_path}')