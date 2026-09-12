"""Plot the existing QRBM-CNN CSV results without retraining.

Usage: python visualize_results.py --result-dir fraud_results_ulb_fixed
Dependencies: pip install numpy pandas matplotlib
Charts use English labels for cross-platform font compatibility.
pr_auc / val_pr_auc in this repository are average precision (AP).
No test ROC/PR curve is fabricated from aggregate scores.
"""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FULL = 'QRBM + CNN + Focal + Adaptive Threshold'
FOCAL = 'QRBM + CNN + Focal'
MODELS = ['Decision Tree (No Prune)', 'Naive Bayes',
          'Logistic (Unweighted)', 'Logistic (Balanced)', 'MLP (32,16)',
          'Plain CNN', 'QRBM + CNN', FULL]
LABELS = ['Decision tree', 'Naive Bayes', 'Logistic (unweighted)',
          'Logistic (balanced)', 'MLP (32,16)', 'Plain CNN',
          'QRBM + CNN', 'QRBM + CNN + Focal + threshold']
HISTORIES = [('Plain_CNN_history.csv', 'Plain CNN'),
             ('QRBM_plus_CNN_history.csv', 'QRBM + CNN'),
             ('QRBM_plus_CNN_plus_Focal_history.csv', 'QRBM + CNN + Focal')]
COLORS = ['#3973a5', '#d58c32', '#329282']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'fraud_results_ulb_fixed')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    src = args.result_dir.resolve()
    out = (args.output_dir or src / 'figures').resolve()
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    notes = [
        'Data are from one recorded split/run; no confidence intervals are available.',
        'Repository pr_auc and val_pr_auc mean average precision (AP), not trapezoidal PR area.',
        'Main comparison excludes two DummyClassifier rows: thresholding constant scores made both predict every case as fraud.',
        'The original CSV is unchanged. The named Focal-only row is not a valid fixed-threshold ablation in the current training code.',
        'Loss functions have different scales: compare loss trajectories within each panel only.',
        'QRBM currently uses classical simulated annealing; these plots do not demonstrate quantum speedup.',
        'No sample-level test scores are saved, so test ROC and PR curves are not generated.'
    ]

    def save(fig, name):
        for ext in ('png', 'svg'):
            fig.savefig(out / f'{name}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print('Saved:', name)

    data = pd.read_csv(src / 'model_comparison.csv')
    required = {'model', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'tn', 'fp', 'fn', 'tp'}
    if not required.issubset(data.columns):
        raise ValueError(f'Missing columns: {required - set(data.columns)}')
    if data['model'].duplicated().any():
        raise ValueError('Duplicate model names in model_comparison.csv')
    data = data.set_index('model')
    for name, row in data.iterrows():
        den = 2 * row.tp + row.fp + row.fn
        expected_f1 = 2 * row.tp / den if den else 0
        if not np.isclose(expected_f1, row.f1, atol=1e-7):
            raise ValueError(f'F1 does not match confusion counts: {name}')
    if FULL not in data.index:
        raise ValueError(f'Missing model: {FULL}')
    if FOCAL in data.index:
        cols = ['precision', 'recall', 'f1', 'threshold', 'tn', 'fp', 'fn', 'tp']
        if np.allclose(data.loc[FULL, cols].astype(float), data.loc[FOCAL, cols].astype(float)):
            notes.append('Focal and full-model rows are identical, including threshold; show the full-model row once, without claiming threshold ablation gains.')
    chosen = [m for m in MODELS if m in data.index]
    labels = [LABELS[MODELS.index(m)] for m in chosen]
    table = data.loc[chosen].copy()
    table.rename(columns={'pr_auc': 'average_precision'}).to_csv(out / 'comparison_for_report.csv')

    # Fig 1: full 0-1 axes, including weak results; no selected metric winner claims.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True, layout='constrained')
    for ax, metric, title in zip(axes, ['f1', 'pr_auc', 'roc_auc'], ['F1', 'Average precision (AP)', 'ROC-AUC']):
        values = table[metric].to_numpy()
        ax.barh(np.arange(len(chosen)), values,
                color=['#329282' if m == FULL else '#3973a5' for m in chosen])
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_title(title)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.text(value + .015, i, f'{value:.4f}', va='center', fontsize=9)
    axes[0].set_yticks(np.arange(len(chosen)), labels)
    axes[0].invert_yaxis()
    fig.suptitle('Test-set model comparison | one recorded run', fontsize=14)
    save(fig, 'fig1_model_comparison')

    # Fig 2: exact counts + true-class-normalized color, avoiding majority-class domination.
    r = data.loc[FULL]
    cm = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=int)
    normalized = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 5.5), layout='constrained')
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}\n({normalized[i,j]:.3%})', ha='center', va='center',
                    fontsize=15, color='white' if normalized[i,j] > .55 else '#172435')
    ax.set_xticks([0, 1], ['Normal', 'Fraud'])
    ax.set_yticks([0, 1], ['Normal', 'Fraud'])
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title(f'Full model: test confusion matrix\nThreshold = {float(r.threshold):.6f}')
    fig.colorbar(im, ax=ax, label='Proportion within true class')
    save(fig, 'fig2_confusion_matrix')

    histories = []
    for filename, label in HISTORIES:
        if (src / filename).exists():
            histories.append((pd.read_csv(src / filename), label))
        else:
            notes.append(f'Missing optional history: {filename}')
    if histories:
        fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
        for (h, label), color in zip(histories, COLORS):
            ax.plot(h.epoch, h.val_pr_auc, label=label, color=color)
            selected_name = FULL if label == FOCAL else label
            if selected_name in data.index and pd.notna(data.loc[selected_name, 'best_epoch']):
                epoch = int(data.loc[selected_name, 'best_epoch'])
                selected = h[h.epoch == epoch]
                if len(selected):
                    ax.scatter([epoch], selected.val_pr_auc, color=color, s=55, zorder=4)
                    notes.append(f'{label}: saved selected epoch {epoch}; validation AP {selected.val_pr_auc.iloc[0]:.6f}.')
        ax.set(xlabel='Epoch', ylabel='Validation AP', ylim=(0, 1), title='Validation AP during training | dots mark saved selected epochs')
        ax.grid(alpha=.2)
        ax.legend(loc='lower right')
        save(fig, 'fig3_validation_ap')

        fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4), squeeze=False, layout='constrained')
        for ax, (h, label), color in zip(axes[0], histories, COLORS):
            ax.plot(h.epoch, h.train_loss, color=color)
            ax.set(title=label, xlabel='Epoch', ylabel='Training loss', ylim=(0, None))
            ax.grid(alpha=.2)
        fig.suptitle('Training losses | separate scales and different objectives')
        save(fig, 'fig4_training_loss')
        for h, label in histories:
            zero = h[(h.val_threshold == .5) & (h.val_recall == 0)]
            if len(zero):
                notes.append(f'{label}: {len(zero)} logged epochs have zero validation recall at threshold 0.5. These are not test results.')

    if (src / 'qrbm_history.csv').exists():
        h = pd.read_csv(src / 'qrbm_history.csv')
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        ax.plot(h.epoch, h.objective, marker='o', color='#76599b')
        ax.axhline(0, color='gray', lw=.8)
        ax.set(xlabel='Epoch', ylabel='Recorded objective (including regularization)',
               title='RBM pretraining objective | not a reconstruction-error curve')
        ax.grid(alpha=.2)
        save(fig, 'fig5_rbm_objective')

    # Optional: existing threshold tables are VALIDATION data only.
    threshold_files = sorted(src.glob('*_threshold.csv'))
    for path in threshold_files:
        t = pd.read_csv(path).dropna(subset=['threshold', 'precision', 'recall', 'f1'])
        t = t.sort_values('threshold').reset_index(drop=True)
        if t.empty:
            continue
        best = int(t.f1.to_numpy().argmax())
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        for col, color in zip(['precision', 'recall', 'f1'], COLORS):
            ax.plot(t.threshold, t[col], label=col.title(), color=color, lw=1)
        ax.axvline(t.threshold.iloc[best], ls='--', color='#444444',
                   label=f'Validation best F1 threshold: {t.threshold.iloc[best]:.6f}')
        ax.set(xlabel='Threshold', ylabel='Validation score', xlim=(0, 1), ylim=(0, 1.02),
               title='Validation threshold sensitivity: ' + path.stem.replace('_', ' '))
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
        save(fig, 'validation_' + path.stem)
    if not threshold_files:
        notes.append('No threshold CSVs in the selected input directory: optional validation threshold plots skipped.')
    (out / 'plot_notes.txt').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    for note in notes:
        print('Note:', note)
    print('Output directory:', out)


if __name__ == '__main__':
    main()
