#!/usr/bin/env python3
"""Draw the M7 diagnostic from existing JSON only; never loads models or trains."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('.agents/runs/foundation/m7'))
    parser.add_argument('--output',type=Path,default=Path('doc/notes/images/m7_diagnostic.png'))
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=args.runs
    direct=json.loads((root/'control_direct.json').read_text())
    fixed=json.loads((root/'control.json').read_text())
    ab=json.loads((root/'ab_summary.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    values=[1000*direct['trials'][0]['position_error_m'],1000*fixed['trials'][0]['position_error_m']]
    axes[0,0].bar(['Initial limiter','Cubic interpolation'],values,color=['#bf6c53','#327da6'])
    for i,v in enumerate(values):axes[0,0].text(i,v+.6,f'{v:.2f} mm',ha='center')
    axes[0,0].axhline(2,color='black',ls='--',label='M5 2 mm gate');axes[0,0].legend()
    axes[0,0].set(ylabel='TCP error (mm)',title='48 Hz control integration')
    for ax,key,title in zip([axes[0,1],axes[1,0],axes[1,1]],['hit_rate','legal_return_rate','target_success_rate'],
                            ['Paddle contact','Legal return','Target box']):
        for i,group in enumerate(['A','B']):
            rows=[r for r in ab['runs'] if r['group']==group]
            values=[100*r['metrics'][key] for r in rows]
            offsets=[(j-(len(rows)-1)/2)*.07 for j in range(len(rows))]
            ax.scatter([i+x for x in offsets],values,s=55,label=f'{group}: {len(rows)} training seeds')
        baseline=ab.get('zero_action_baseline')
        if baseline is not None:ax.axhline(100*baseline[key],color='black',ls='--',label='Zero action (16 serves)')
        ax.set(xticks=[0,1],xticklabels=['A: sparse','B: + hit guidance'],ylim=(-5,105),ylabel='Fixed evaluation serves (%)',title=title)
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Small A/B diagnostic — 6,144 transitions per seed; not a superiority claim')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=140);plt.close(fig)


if __name__=='__main__':main()
