#!/usr/bin/env python3
"""Replot a saved M6 validation.json. Does not import torch, fit or evaluate."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args=parser.parse_args()
    report=json.loads(args.report.read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    curve=report['learning_curve']
    axes[0].plot([x['epoch'] for x in curve],[1000*x['validation_rmse_m'] for x in curve])
    axes[0].axhline(report['config']['gate']['predictor_rmse_m']*1000,color='red',ls='--')
    axes[0].set(xlabel='Epoch',ylabel='Validation RMSE (mm)',title='Checkpoint selection')
    axes[1].bar(['Learned','Training mean','Oracle physics'],
                [report[k]['rmse_m']*1000 for k in ('learned_all','training_mean','teacher')])
    axes[1].set(ylabel='Held-out RMSE (mm)',title='Narrow distribution; oracle uses truth')
    values=list(report['per_serve_rmse_m'].values())
    axes[2].plot(range(1,len(values)+1),[v*1000 for v in values],'.')
    axes[2].set(xlabel='Held-out serve index',ylabel='RMSE (mm)',title='Serves, NOT independent training runs')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=140)
    plt.close(fig)


if __name__=='__main__':
    main()
