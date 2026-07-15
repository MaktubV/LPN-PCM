"""Plot Fig 3(d-f): SA, BKW (LF1), and difference heatmaps."""
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
bkw_heatmap = data['bkw_heatmap']

sa_dict, bkw_dict = {}, {}
for i, m in enumerate(m_list):
    for j, err in enumerate(err_list):
        sa_dict[(m, err)] = float(sa_heatmap[i, j])
        bkw_dict[(m, err)] = float(bkw_heatmap[i, j])

m_inv = m_list[::-1]
plt.rcParams.update({'font.size': 14})


def plot_heatmap(mat_dict, fname, cmap, label, vmin=0, vmax=1):
    mat = np.empty((len(err_list), len(m_inv)))
    for i, e in enumerate(err_list):
        for j, f in enumerate(m_inv):
            mat[i, j] = mat_dict.get((f, e), np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mat, xticklabels=m_inv,
                yticklabels=[f'{e:.2f}' for e in err_list],
                cmap=cmap, vmin=vmin, vmax=vmax, annot=False, ax=ax,
                cbar_kws={'label': label})
    ax.invert_yaxis()
    ax.set_xlabel(r'Number of Samples $m$')
    ax.set_ylabel(r'Noise Rate $\tau$')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, fname)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=300)
    plt.close(fig)
    print(f'Saved {out_path}')


# Panel (d): SA heatmap
plot_heatmap(sa_dict, 'fig3d_sa.pdf', 'crest', 'Success Rate')

# Panel (e): BKW heatmap
plot_heatmap(bkw_dict, 'fig3e_bkw.pdf', 'crest', 'Success Rate')

# Panel (f): difference heatmap
diff_cmap = sns.diverging_palette(10, 150, s=80, l=50, as_cmap=True, center='light')
diff_mat = np.empty((len(err_list), len(m_inv)))
for i, e in enumerate(err_list):
    for j, f in enumerate(m_inv):
        diff_mat[i, j] = sa_dict.get((f, e), np.nan) - bkw_dict.get((f, e), np.nan)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(diff_mat, xticklabels=m_inv,
            yticklabels=[f'{e:.2f}' for e in err_list],
            cmap=diff_cmap, center=0, vmin=0, vmax=1.0, annot=False, ax=ax,
            cbar_kws={'label': 'Success diff (SA − BKW)'})
ax.invert_yaxis()
ax.set_xlabel(r'Number of Samples $m$')
ax.set_ylabel(r'Noise Rate $\tau$')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
fig.tight_layout()
out_path = os.path.join(FIG_DIR, 'fig3f_diff.pdf')
fig.savefig(out_path, dpi=300)
fig.savefig(out_path.replace('.pdf', '.png'), dpi=300)
plt.close(fig)
print(f'Saved {out_path}')
print('All figures generated in fig/')