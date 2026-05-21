import os, sys, csv, time, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.channels  import (gen_tdl_channel, gen_tdl_random,
                                SCENARIOS, SNAMES, TIME_STEPS)
from shared.baselines import (ls_estimate, mmse_estimate,
                                sadlcs_postprocess,
                                add_snr_channel, snr_aware_blend)


# ─── Custom layer for NCRN ────────────────────────────────────
class LastFrame(tf.keras.layers.Layer):
    def call(self, x): return x[:, -1, :, :, :]
    def get_config(self): return super().get_config()


# ─── Configuration ────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, '..', 'models')
OUTPUT_DIR = os.path.join(HERE, '..', 'outputs', 'figures')
CSV_DIR    = os.path.join(HERE, '..', 'outputs', 'csv')

SNR_RANGE  = [-15, -10, -5, 0, 5, 10, 15, 20]

# Sample counts (reduced if --quick)
N_TDL  = 500
N_SCEN = 1000

MODEL_PATHS = {
    'sa_dlcs_tdl_64': 'sa_dlcs_tdl_64.keras',
    'ncrn_tdl_64':    'ncrn_tdl_64.keras',
}


# ─── Plot style ───────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif',
                          'Liberation Serif'],
    'mathtext.fontset':  'stix',
    'font.size':         9,
    'axes.titlesize':    9,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   7.5,
    'legend.frameon':    True,
    'legend.framealpha': 0.92,
    'legend.edgecolor':  '0.5',
    'legend.fancybox':   False,
    'axes.linewidth':    0.7,
    'axes.edgecolor':    '0.2',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  3.0,
    'ytick.major.size':  3.0,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'lines.linewidth':   1.0,
    'lines.markersize':  3.5,
    'grid.linewidth':    0.4,
    'grid.linestyle':    ':',
    'grid.alpha':        0.45,
    'grid.color':        '0.6',
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.facecolor': 'white',
    'savefig.dpi':       600,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

# ─── Method styling ───────────────────────────────────────────
COLORS = {
    'ls':     '#000000',   # black
    'mmse':   '#1B5E20',   # dark green
    'sadlcs': '#0D3B66',   # dark blue
    'ncrn':   '#7B1A1A',   # dark red
}
LINESTYLES = {
    'ls':     '--',
    'mmse':   '-.',
    'sadlcs': '-',
    'ncrn':   '-',
}
MARKERS = {
    'ls':     '^',
    'mmse':   'D',
    'sadlcs': 's',
    'ncrn':   'o',
}
LABELS = {
    'ls':     'LS',
    'mmse':   'MMSE',
    'sadlcs': 'Sa-DLCS',
    'ncrn':   'NCRN',
}
METHOD_ORDER = ['ls', 'mmse', 'sadlcs', 'ncrn']


def _style_for(m):
    """Return plot kwargs for a single curve."""
    is_ncrn = (m == 'ncrn')
    return dict(
        color=COLORS[m],
        marker=MARKERS[m],
        linestyle=LINESTYLES[m],
        linewidth=1.4 if is_ncrn else 0.9,
        markersize=4.2 if is_ncrn else 3.4,
        markerfacecolor=COLORS[m] if is_ncrn else 'white',
        markeredgecolor=COLORS[m],
        markeredgewidth=0.75,
        label=LABELS[m],
    )


def _save_fig(fig, name):
    """Save figure as both vector PDF and 600-dpi PNG."""
    pdf_path = os.path.join(OUTPUT_DIR, f'{name}.pdf')
    png_path = os.path.join(OUTPUT_DIR, f'{name}.png')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
                facecolor='white')
    fig.savefig(png_path, format='png', dpi=600, bbox_inches='tight',
                facecolor='white')
    print(f"  → {name}.pdf  +  {name}.png")


# ─── Model cache & retrace-free predict ───────────────────────
_MODEL_CACHE      = {}
_PREDICT_FN_CACHE = {}

def _make_predict_fn(model):
    @tf.function(reduce_retracing=True)
    def _predict(x):
        return model(x, training=False)
    return _predict

def _predict_one(model, x_np):
    if id(model) not in _PREDICT_FN_CACHE:
        _PREDICT_FN_CACHE[id(model)] = _make_predict_fn(model)
    fn   = _PREDICT_FN_CACHE[id(model)]
    x_tf = tf.constant(np.expand_dims(x_np, 0))
    return fn(x_tf).numpy()[0]

def load_model(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    fname = MODEL_PATHS.get(key)
    if not fname:
        _MODEL_CACHE[key] = None
        return None
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        print(f"  [WARN] Missing: {fname}")
        _MODEL_CACHE[key] = None
        return None
    try:
        m = tf.keras.models.load_model(
            path, custom_objects={'LastFrame': LastFrame},
            safe_mode=False)
        _MODEL_CACHE[key] = m
        print(f"  [OK]   Loaded {fname}")
        return m
    except Exception as e:
        print(f"  [WARN] Failed to load {fname}: {e}")
        _MODEL_CACHE[key] = None
        return None


# ─── Evaluation pipeline ──────────────────────────────────────
def evaluate_methods(Nr, Nt, snr_range, n_samples, channel_fn):
    """Returns {method: {'nmse_db', 'nmse_lin', 'mse_lin'}} per SNR."""
    sadlcs_m  = load_model('sa_dlcs_tdl_64')
    ncrn_m    = load_model('ncrn_tdl_64')
    needs_snr = (ncrn_m is not None and ncrn_m.input_shape[-1] == 3)

    results        = {m: {'errs': [], 'mses': []} for m in METHOD_ORDER}
    denoms_per_snr = []

    for snr in snr_range:
        ns = np.sqrt(10**(-snr/10))
        for m in METHOD_ORDER:
            results[m]['errs'].append([])
            results[m]['mses'].append([])
        denoms_per_snr.append([])

        for _ in range(n_samples):
            h  = channel_fn(Nr, Nt)
            hn = (h + np.random.normal(0, ns, h.shape)).astype(np.float32)
            hl = hn[-1]
            hr = h[-1]
            denom = np.sum(hr**2) + 1e-12
            denoms_per_snr[-1].append(denom)

            pred = ls_estimate(hl)
            e    = np.sum((hr - pred)**2)
            results['ls']['errs'][-1].append(e)
            results['ls']['mses'][-1].append(e / (Nr * Nt))

            pred = mmse_estimate(hl, snr, Nt)
            e    = np.sum((hr - pred)**2)
            results['mmse']['errs'][-1].append(e)
            results['mmse']['mses'][-1].append(e / (Nr * Nt))

            if sadlcs_m is not None:
                pr   = _predict_one(sadlcs_m, hn)
                pred = sadlcs_postprocess(pr, hl)
                e    = np.sum((hr - pred)**2)
                results['sadlcs']['errs'][-1].append(e)
                results['sadlcs']['mses'][-1].append(e / (Nr * Nt))

            if ncrn_m is not None:
                ncin = add_snr_channel(hn, snr) if needs_snr else hn
                raw  = _predict_one(ncrn_m, ncin)
                pred = snr_aware_blend(raw, hl, snr_db_known=snr)
                e    = np.sum((hr - pred)**2)
                results['ncrn']['errs'][-1].append(e)
                results['ncrn']['mses'][-1].append(e / (Nr * Nt))

    out = {}
    for m in METHOD_ORDER:
        if not results[m]['errs'][0]:
            continue
        nmse_db, nmse_lin, mse_lin = [], [], []
        for i in range(len(snr_range)):
            denom = sum(denoms_per_snr[i]) + 1e-12
            errs  = results[m]['errs'][i]
            if not errs:
                nmse_db.append(None); nmse_lin.append(None)
                mse_lin.append(None); continue
            nl = sum(errs) / denom
            nmse_lin.append(nl)
            nmse_db.append(10 * np.log10(nl + 1e-12))
            mse_lin.append(np.mean(results[m]['mses'][i]))
        out[m] = {'nmse_db': nmse_db, 'nmse_lin': nmse_lin,
                  'mse_lin': mse_lin}
    return out


# ─── Legend helper ────────────────────────────────────────────
def _add_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    ncol = 2 if len(handles) >= 3 else 1
    defaults = dict(loc='upper right', ncol=ncol,
                    columnspacing=1.0, handlelength=1.8,
                    handletextpad=0.5, borderpad=0.5)
    defaults.update(kwargs)
    ax.legend(handles, labels, **defaults)


# ─── PLOT 1: Stage II TDL averaged ────────────────────────────
def plot_stage2(n_samples):
    print("\n  ─── Stage II TDL averaged, 64×64 ───")
    Nr, Nt = 64, 64

    if not any(os.path.exists(os.path.join(MODELS_DIR, MODEL_PATHS[k]))
               for k in MODEL_PATHS):
        print("  [SKIP] No TDL models found.")
        return

    res = evaluate_methods(Nr, Nt, SNR_RANGE, n_samples, gen_tdl_random)

    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for m in METHOD_ORDER:
        if m not in res:
            continue
        valid = [(x, v) for x, v in zip(SNR_RANGE, res[m]['nmse_lin'])
                 if v is not None and v > 0]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax.semilogy(xs, ys, **_style_for(m))

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('NMSE')
    ax.grid(True, which='both')
    ax.minorticks_on()
    _add_legend(ax)
    plt.tight_layout(pad=0.3)
    _save_fig(fig, 'nmse_tdl_64')
    plt.close(fig)
    export_csv(res, SNR_RANGE, os.path.join(CSV_DIR, 'tdl_64.csv'))


# ─── PLOT 2–5: Per-scenario panels ────────────────────────────
def _extract_env(sname):
    s = sname.upper()
    for needle, label in [('INH','InH'),('UMI','UMi'),
                           ('UMA','UMa'),('RMA','RMa')]:
        if needle in s:
            return label
    return None

def _scenario_title(sname, cfg, group):
    env  = _extract_env(sname)
    v    = cfg.get('v', '?')
    fc   = cfg.get('fc', None)
    fc_s = f"{fc/1e9:.1f}\u202fGHz" if fc else '?'
    head = f"{env} {group}" if env else sname.replace('_', ' ')
    return f"{head} ({v}\u202fkm/h, {fc_s})"

def _sort_key(sname):
    order = {'InH': 0, 'UMi': 1, 'UMa': 2, 'RMa': 3, None: 9}
    return (order[_extract_env(sname)], sname)


def plot_per_scenario(group, n_samples):
    Nr, Nt = 64, 64
    sc_in_group = sorted([n for n, c in SCENARIOS.items()
                          if c['group'] == group], key=_sort_key)
    if not sc_in_group:
        return
    print(f"\n  ─── Per-scenario {group} ({len(sc_in_group)} panels) ───")

    sadlcs_m  = load_model('sa_dlcs_tdl_64')
    ncrn_m    = load_model('ncrn_tdl_64')
    if ncrn_m is None:
        print("  [WARN] NCRN model not found — skipping.")
        return
    needs_snr = (ncrn_m.input_shape[-1] == 3)

    n = len(sc_in_group)
    if   n <= 2: nrow, ncol = 1, n
    elif n <= 4: nrow, ncol = 2, 2
    elif n <= 6: nrow, ncol = 2, 3
    else:        ncol = 3; nrow = (n + ncol - 1) // ncol

    figw = 3.6 if ncol == 1 else (2.7 * ncol + 0.6)
    figh = 2.7 * nrow + 0.6
    fig, axes = plt.subplots(nrow, ncol, figsize=(figw, figh),
                              sharex=True, sharey=True)
    axes = np.atleast_2d(axes).flatten()

    for i, sname in enumerate(sc_in_group):
        ax  = axes[i]
        cfg = SCENARIOS[sname]
        print(f"    {sname}")

        nmse_lin             = {k: [] for k in METHOD_ORDER}
        ncrn_low, ncrn_high  = [], []

        for snr in SNR_RANGE:
            ns            = np.sqrt(10**(-snr/10))
            errs          = {k: [] for k in METHOD_ORDER}
            denom_sum     = 0.0
            ncrn_per_sample = []

            for _ in range(n_samples):
                h  = gen_tdl_channel(sname, Nr, Nt)
                hn = (h + np.random.normal(0, ns, h.shape)).astype(np.float32)
                hl = hn[-1]; hr = h[-1]
                d  = np.sum(hr**2) + 1e-12
                denom_sum += d

                errs['ls'].append(np.sum((hr - ls_estimate(hl))**2))
                errs['mmse'].append(np.sum((hr - mmse_estimate(hl, snr, Nt))**2))

                if sadlcs_m is not None:
                    pr = _predict_one(sadlcs_m, hn)
                    sa = sadlcs_postprocess(pr, hl)
                    errs['sadlcs'].append(np.sum((hr - sa)**2))

                ncin = add_snr_channel(hn, snr) if needs_snr else hn
                raw  = _predict_one(ncrn_m, ncin)
                p4   = snr_aware_blend(raw, hl, snr_db_known=snr)
                e_n  = np.sum((hr - p4)**2)
                errs['ncrn'].append(e_n)
                ncrn_per_sample.append(e_n / d)

            for k in METHOD_ORDER:
                if errs[k]:
                    nl = sum(errs[k]) / (denom_sum + 1e-12)
                    nmse_lin[k].append(nl)
                else:
                    nmse_lin[k].append(None)

            arr     = np.array(ncrn_per_sample) + 1e-14
            log_arr = np.log10(arr)
            mu, sd  = float(np.mean(log_arr)), float(np.std(log_arr))
            ncrn_low.append(10 ** (mu - sd))
            ncrn_high.append(10 ** (mu + sd))

        snr_arr = np.array(SNR_RANGE)

        if not any(v is None for v in nmse_lin['ncrn']):
            ax.fill_between(snr_arr, ncrn_low, ncrn_high,
                            color=COLORS['ncrn'], alpha=0.10, linewidth=0)

        for m in METHOD_ORDER:
            valid = [(x, v) for x, v in zip(snr_arr, nmse_lin[m])
                     if v is not None and v > 0]
            if not valid:
                continue
            xs, ys = zip(*valid)
            ax.semilogy(xs, ys, **_style_for(m))

        ax.set_title(_scenario_title(sname, cfg, group), fontsize=8.5, pad=3)
        ax.grid(True, which='both')
        ax.minorticks_on()
        if i % ncol == 0:
            ax.set_ylabel('NMSE')
        if i // ncol == nrow - 1:
            ax.set_xlabel('SNR (dB)')

        export_csv_perscenario(
            snr_arr, nmse_lin,
            os.path.join(CSV_DIR, f'tdl_{sname}_64.csv'))

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels,
                   loc='lower center', ncol=len(handles),
                   bbox_to_anchor=(0.5, -0.02), frameon=False,
                   columnspacing=1.4, handlelength=2.0, handletextpad=0.5)
        plt.tight_layout(rect=[0, 0.05, 1, 1], pad=0.3, h_pad=0.6, w_pad=0.6)
    else:
        plt.tight_layout(pad=0.3)

    _save_fig(fig, f'tdl_scenarios_{group}')
    plt.close(fig)


# ─── CSV export ───────────────────────────────────────────────
def export_csv(res, snr_range, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ms   = [m for m in METHOD_ORDER if m in res]
    cols = (['SNR (dB)'] +
            [f'{LABELS[m]} NMSE (dB)' for m in ms] +
            [f'{LABELS[m]} NMSE (lin)' for m in ms])
    with open(save_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, snr in enumerate(snr_range):
            row = [snr]
            for m in ms:
                v = res[m]['nmse_db'][i] if i < len(res[m]['nmse_db']) else None
                row.append(f'{v:+.3f}' if v is not None else 'N/A')
            for m in ms:
                v = res[m]['nmse_lin'][i] if i < len(res[m]['nmse_lin']) else None
                row.append(f'{v:.6e}' if v is not None else 'N/A')
            w.writerow(row)
    print(f"  → CSV {os.path.basename(save_path)}")


def export_csv_perscenario(snr_range, nmse_lin, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ms = [m for m in METHOD_ORDER if nmse_lin.get(m)]
    with open(save_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['SNR (dB)'] +
                   [f'{LABELS[m]} NMSE (dB)' for m in ms] +
                   [f'{LABELS[m]} NMSE (lin)' for m in ms])
        for i, snr in enumerate(snr_range):
            row = [int(snr)]
            for m in ms:
                v = nmse_lin[m][i] if i < len(nmse_lin[m]) else None
                row.append(f'{10*np.log10(v+1e-12):+.3f}'
                           if (v is not None and v > 0) else 'N/A')
            for m in ms:
                v = nmse_lin[m][i] if i < len(nmse_lin[m]) else None
                row.append(f'{v:.6e}' if v is not None else 'N/A')
            w.writerow(row)


# ─── Main ─────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description='Generate TDL NMSE figures for NCRN.')
    p.add_argument('--quick', action='store_true',
                   help='Reduce sample counts 4× for fast preview.')
    p.add_argument('--skip', nargs='+', default=[],
                   choices=['stage2', 'scenarios'],
                   help='Skip individual figure groups.')
    args = p.parse_args()

    n_tdl  = N_TDL  // 4 if args.quick else N_TDL
    n_scen = N_SCEN // 4 if args.quick else N_SCEN

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CSV_DIR,    exist_ok=True)

    print("=" * 70)
    print("  TDL CHECKING — NMSE Figures (64×64)")
    print(f"  Methods : LS · MMSE · Sa-DLCS · NCRN")
    print(f"  Models  : {MODELS_DIR}")
    print(f"  Output  : {OUTPUT_DIR}  (PDF + 600 dpi PNG)")
    print(f"  Mode    : {'QUICK' if args.quick else 'FULL'} "
          f"(N_tdl={n_tdl}, N_scen={n_scen})")
    print("=" * 70)

    t0 = time.time()

    if 'stage2' not in args.skip:
        print("\n[1/5] Stage II — averaged TDL NMSE")
        plot_stage2(n_tdl)

    if 'scenarios' not in args.skip:
        print("\n[2/5] Per-scenario panels (LOS / NLOS / O2I / HST)")
        for grp in ['LOS', 'NLOS', 'O2I', 'HST']:
            plot_per_scenario(grp, n_scen)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Done — {elapsed/60:.1f} min")
    print(f"  Figures : {OUTPUT_DIR}")
    print(f"  CSVs    : {CSV_DIR}")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()