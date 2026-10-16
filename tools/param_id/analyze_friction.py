#!/usr/bin/env python3
"""Offline tracking assessment, Method D/C/E identification and held-out validation."""
import argparse
import json
from pathlib import Path
from friction_id.analysis import Gates, analyze, validate, plot_report, compare_compensation
from friction_id.common import Gravity
from friction_id.recording import recover


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='action',required=True)
    for action in ['fit','validate']:
        p=sub.add_parser(action)
        p.add_argument('input',type=Path)
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--gravity',action='store_true',help='Compute C using local model; hash must match acquisition')
        if action=='fit':
            p.add_argument('--gates',type=Path,help='JSON Gates overrides; record changes between experiments')
            p.add_argument('--plot',action='store_true')
        else: p.add_argument('--candidate',type=Path,required=True)
    p=sub.add_parser('recover');p.add_argument('journal',type=Path);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('compare');p.add_argument('baseline',type=Path);p.add_argument('trial',type=Path);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('dynamic-fit',help='Single-axis nonuniform-motion diagnostics; never deployable')
    p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window',type=float,default=.20)
    p.add_argument('--period-min',type=float,default=.03);p.add_argument('--period-max',type=float,default=.06)
    p=sub.add_parser('dynamic-validate',help='Frozen prediction on a separate raw run')
    p.add_argument('input',type=Path);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',action='store_true')
    p=sub.add_parser('changed-start',help='Compare absolute angle versus experiment-start alignment')
    p.add_argument('reference',type=Path);p.add_argument('trial',type=Path)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    args=ap.parse_args(argv)
    if args.action=='recover':
        recover(args.journal,args.output); print(args.output);return
    if args.action=='compare':
        compare_compensation(args.baseline,args.trial,args.output);print(args.output);return
    if args.action=='changed-start':
        from friction_id.shifted import compare_shift,plot_shift
        report=compare_shift(args.model,args.reference,args.trial,args.output,Gravity())
        if args.plot:plot_shift(args.output/'changed_start_profiles.npz',args.output)
        print(json.dumps(dict(directions=report['directions'],absolute_angle_alignment_supported=
            report['absolute_angle_alignment_supported'],compensation_allowed=False),indent=2));return
    if args.action in ('dynamic-fit','dynamic-validate'):
        from friction_id.dynamic import fit_dynamic,validate_dynamic,plot_dynamic
        gravity=Gravity()
        if args.action=='dynamic-fit':
            path=fit_dynamic(args.input,args.output,gravity,dict(window=args.window,
                period_min=args.period_min,period_max=args.period_max))
            model=json.loads(path.read_text())
            print(json.dumps(dict(model=str(path),models=model['models'],compensation_allowed=False),indent=2));return
        report=validate_dynamic(args.model,args.input,args.output,gravity)
        if args.plot:plot_dynamic(args.output/'validation_predictions.npz',args.output)
        print(json.dumps(dict(conclusion=report['conclusion'],results=report['results'],
            compensation_allowed=False),indent=2));return
    gravity=Gravity() if args.gravity else None
    if gravity:
        from friction_id.recording import load_run
        meta,_,_=load_run(args.input)
        if meta['conditions']['model_hash']!=gravity.model_hash: raise ValueError('Gravity model hash differs from acquisition')
    if args.action=='validate':
        print(validate(args.candidate,args.input,args.output,gravity)); return
    overrides=json.loads(args.gates.read_text()) if args.gates else {}
    report,_=analyze(args.input,args.output,Gates(**overrides),gravity)
    if args.plot: plot_report(report,args.output/'aligned.npz',args.output)
    print(json.dumps(dict(segments=report['segments'],fits=report['fits'],eligible_for_validation=report['eligible_for_validation']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
