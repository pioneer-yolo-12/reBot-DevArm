"""Offline pilot audit, frozen cross-profile prediction and mirror comparison."""
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parents[2]))
from friction_id.recording import load_run
from friction_id.paired_dynamic import Options, pairs, fit_paired, validate_paired
from friction_id.analysis import prepare, Gates
from preview_dynamic_excitation import features, identifiability


def rms(x):
    return float(np.sqrt(np.mean(x*x)))


def main():
    output = BASE/'review'
    output.mkdir(exist_ok=True)
    data = []; sources = []; summary = []
    for name in ('J2_up08_down20', 'J2_up20_down08'):
        p = next((BASE/name).glob('*.npz')); sources.append(p)
        meta, feedback, commands = load_run(p)
        with np.load(p, allow_pickle=False) as z:
            annotations = [json.loads(str(x)) for x in z['annotations_json']]
        fold = next(x for x in annotations if x.get('kind') == 'fold_verified')
        off = next(x for x in annotations if x.get('kind') == 'disabled_after_fold')
        assert meta['run_status'] == 'complete' and commands['ok'].all() and not feedback['status'].any()
        assert fold['t'] < off['t']
        d, info = pairs(p, Options(motion_scope='full'))
        ref, _ = features(meta)
        ref_da = np.interp(d['q'], ref['q'], ref['da'])
        a, segments = prepare(meta, feedback, commands, Gates())
        full = (a['segment'] >= 0) & np.isin(a['phase'], ['accel', 'constant', 'decel'])
        stats = dict(name=name, pairing=info, matrix_diagnostic_only=identifiability(d, .0498666),
            reference_da_rms=rms(ref_da), actual_da_rms=rms(d['da']),
            actual_reference_da_correlation=float(np.corrcoef(d['da'], ref_da)[0, 1]),
            original_method_D_segments=segments, fold=fold, disabled=off,
            max_tracking_error=float(abs(a['q'][full]-a['q_ref'][full]).max()),
            peak_temperature=[float(feedback[feedback['joint'] == j]['temperature'].max()) for j in range(1, 7)])
        summary.append(stats); data.append((d, ref_da))
        np.savez_compressed(output/(name+'_pairs.npz'), **d, reference_da=ref_da)
    left, right = data[0][0], data[1][0]
    q = np.intersect1d(left['q'], right['q'])
    li = np.searchsorted(left['q'], q); ri = np.searchsorted(right['q'], q)
    da = left['da'][li]-right['da'][ri]
    reference_da = data[0][1][li]-data[1][1][ri]
    delta_D = left['D'][li]-right['D'][ri]
    mirror = dict(bins=len(q), reference_delta_da_rms=rms(reference_da), actual_delta_da_rms=rms(da),
        actual_reference_delta_da_correlation=float(np.corrcoef(da, reference_da)[0, 1]),
        delta_D_rms=rms(delta_D), delta_D_reference_delta_da_correlation=float(np.corrcoef(delta_D, reference_da)[0, 1]),
        delta_dv_rms=rms(left['dv'][li]-right['dv'][ri]),
        delta_df_rms=rms(left['df'][li]-right['df'][ri]))
    model_path = fit_paired([sources[0]], output/'fit', Options(motion_scope='full'))
    model = json.loads(model_path.read_text())
    validation = validate_paired(model_path, sources[1], output/'validation')
    report = dict(pilots=summary, mirror=mirror, training_excitation_issues=model['excitation_issues'],
        models=[dict(kind=m['kind'], names=m['names'], coefficients=m['coefficients'], issues=m['issues'],
                     training_metrics=m['training_metrics'], heldout_metrics=validation['results'][m['kind']]) for m in model['models']],
        compensation_allowed=False, limits=['Two single-speed pilots, not independent parameter identification.',
            'Mirror subtraction assumes unchanged angle-related effects and load/temperature conditions.',
            'Matrix rank or acceleration RMS does not establish that designed excitation dominates measured ripple.'])
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    for (d, ref), name in zip(data, ('up .8 / down 2', 'up 2 / down .8')):
        axes[0].plot(d['q'], d['da'], lw=.8, label=name+' measured')
        axes[0].plot(d['q'], ref, lw=1.5, label=name+' reference')
    axes[1].plot(q, da, label='measured mirror delta')
    axes[1].plot(q, reference_da, label='reference mirror delta')
    axes[2].plot(q, delta_D, label='torque half-difference: run A - run B')
    axes[0].set_ylabel('Acceleration half-difference (rad/s²)')
    axes[1].set_ylabel('Between-run delta (rad/s²)')
    axes[2].set_ylabel('Unverified protocol torque')
    for ax in axes:
        ax.set_xlabel('J2 actual angle (rad)'); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output/'mirror_comparison.png', dpi=140)
    print(json.dumps(dict(mirror=mirror,models=report['models'],training_excitation_issues=report['training_excitation_issues']), indent=2))


if __name__ == '__main__':
    main()
