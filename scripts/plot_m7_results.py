#!/usr/bin/env python3
"""Draw the M7 diagnostic from existing JSON only; never loads models or trains."""
import argparse
import json
from pathlib import Path


def update_comparison(root, output):
    """Separate numerical update stability from final task performance."""
    import matplotlib.pyplot as plt
    diagnostic=json.loads((root/'update_diagnostic_summary.json').read_text())
    old=json.loads((root/'ab_summary.json').read_text())
    new=json.loads((root/'ab_lr1e5_summary.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    colors={.001:'#bb493a',.0001:'#b18829',.00001:'#287d9b'}
    for index,trial in enumerate(diagnostic['runs']):
        for c in trial['comparisons']:
            axes[0,0].plot(range(1,21),[x['after']['kl'] for x in c['steps']],color=colors[c['initial_lr']],alpha=.65,
                          label=f"Initial LR {c['initial_lr']:g}" if index==0 else None)
    axes[0,0].set(yscale='log',xlabel='Optimizer step in first update',ylabel='KL on entire rollout',title='Identical data, weights and RNG; 6 trials each')
    axes[0,0].axhline(.01,color='black',ls='--',label='Adaptive target');axes[0,0].legend(fontsize=8)
    for report,color,label in [(old,colors[.001],'Original 1e-3'),(new,colors[.00001],'Candidate 1e-5')]:
        for i,r in enumerate(report['runs']):
            axes[0,1].plot([x['iteration'] for x in r['history']],[x['policy_kl'] for x in r['history']],color=color,alpha=.65,label=label if i==0 else None)
    axes[0,1].set(yscale='log',xlabel='PPO update',ylabel='KL after update',title='Full short run; same 6,144 transitions / trial')
    axes[0,1].axhline(.01,color='black',ls='--');axes[0,1].legend(fontsize=8)
    for ax,key,title in [(axes[1,0],'hit_rate','Paddle contact'),(axes[1,1],'legal_return_rate','Legal return')]:
        for i,(report,color,label) in enumerate([(old,colors[.001],'Original'),(new,colors[.00001],'Candidate')]):
            for j,group in enumerate(['A','B']):
                rows=sorted([r for r in report['runs'] if r['group']==group],key=lambda r:r['training_seed'])
                ax.scatter([j*2+i+(k-1)*.10 for k in range(len(rows))],[r['metrics'][key]*100 for r in rows],color=color,s=45,label=label if j==0 else None)
        ax.axhline(new['zero_action_baseline'][key]*100,color='black',ls='--',label='Zero action')
        ax.set(xticks=[0,1,2,3],xticklabels=['A old','A new','B old','B new'],ylim=(-5,105),ylabel='Same 16 evaluation serves (%)',title=title)
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('M7 learning-rate diagnostic: stability is not task success')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=140);plt.close(fig)


def net_diagnostic(root, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    r=json.loads((root/'analysis/net_replay.json').read_text())
    c=r['ball_config'];top=c['table_top_m']+c['net_height_m'];radius=c['radius_m']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    colors={'A_7101':'#287d9b','B_7101':'#48986a','A_7103':'#bb493a','B_7103':'#b18829'}
    for name,trial in r['runs'].items():
        for i,row in enumerate(trial['records']):
            points=[x for x in row['trace'] if x[0]>=row['release']['t'] and x[0]<row['end']['before']['t']]
            points=[[row['release']['t']]+row['release']['ball'][:3]]+points+[[row['end']['before']['t']]+row['end']['before']['ball'][:3]]
            for ax in axes:
                ax.plot([v[1] for v in points],[v[3] for v in points],color=colors[name],alpha=.5,lw=.9,label=name if i==0 else None)
    for ax in axes:
        ax.axhline(top+radius,color='grey',ls=':',label='Net top + ball radius')
        ax.add_patch(Rectangle((-c['net_thickness_m']/2,c['table_top_m']),c['net_thickness_m'],c['net_height_m'],color='black'))
        ax.axhline(c['table_top_m'],color='black',lw=.8)
        ax.set(xlabel='X toward opponent (m)',ylabel='Ball centre Z (m)');ax.grid(alpha=.2)
    axes[0].set(xlim=(-1.3,.85),ylim=(.75,1.75),title='Measured outgoing flight; same 16 serves')
    axes[0].legend(fontsize=8)
    row=r['runs']['A_7103']['records'][9]
    p=row['end']['before']['ball']
    axes[1].add_patch(Circle((p[0],p[2]),radius,fill=False,color=colors['A_7103'],lw=2))
    axes[1].set(xlim=(-.07,.07),ylim=(.84,1.03),title='Net detail: sphere extends beyond its centre')
    axes[1].set_aspect('equal',adjustable='box')
    axes[1].text(-.067,1.018,'Circle: A_7103, serve env9, before net hit',fontsize=7)
    fig.suptitle('Seed 7103: slower forward flight and insufficient net clearance')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=140);plt.close(fig)


def serve_comparison(root, output):
    """Plot all fixed checkpoints and legal landings from the frozen study."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    report=json.loads((root/'serve_study/final_summary.json').read_text())
    if report['status']!='PASS':raise ValueError('Final evaluation incomplete')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    colors={'narrow':'#287d9b','challenge':'#bb493a'}
    for ax,evaluation in zip(axes[:2],['narrow','challenge']):
        for index,condition in enumerate(colors):
            rows=[r for r in report['runs'] if r['training_condition']==condition]
            for offset,row in enumerate(rows):
                rate=row['evaluations'][evaluation]['legal_return_rate']*100
                ax.scatter(index+(offset-1)*.12,rate,color=colors[condition],s=45)
                ax.annotate(str(row['seed']), (index+(offset-1)*.12,rate),xytext=(0,7),textcoords='offset points',fontsize=7,ha='center')
        zero=report['baseline'][evaluation]['legal_return_rate']*100
        ax.axhline(zero,color='black',ls='--',label=f'Zero action: {zero:g}%')
        ax.set(xticks=[0,1],xticklabels=['Train +/-1.5 cm','Train +/-12 cm'],ylim=(-5,115),ylabel='Legal returns (%)',
               title=f'Evaluate {evaluation}: same 32 serves')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axes[2]
    for condition,color in colors.items():
        points=[ep['landing_xy_m'] for row in report['runs'] if row['training_condition']==condition
                for ep in row['evaluations']['challenge']['episodes'] if ep['landing_xy_m'] is not None]
        ax.scatter([p[0] for p in points],[p[1] for p in points],s=12,alpha=.5,color=color,label=f'Train {condition}')
    zero=[ep['landing_xy_m'] for ep in report['baseline']['challenge']['episodes'] if ep['landing_xy_m'] is not None]
    ax.scatter([p[0] for p in zero],[p[1] for p in zero],s=16,marker='x',color='black',label='Zero action')
    ax.add_patch(Rectangle((.55,-.2),.4,.4,fill=False,color='green',ls='--',label='Target box'))
    ax.set(xlim=(0,1.37),ylim=(-.7625,.7625),xlabel='Opponent side X (m)',ylabel='Y (m)',title='Challenge: legal landings only')
    ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Single factor: training serve width | 6,144 transitions per seed | rally net contact v2')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=140);plt.close(fig)


def guidance_comparison(root, output):
    """Show A/B outcomes separately from the development feasibility witness."""
    import matplotlib.pyplot as plt
    report=json.loads((root/'m74/final_summary.json').read_text())
    development=json.loads((root/'m74/development.json').read_text())
    if report['status']!='PASS' or development['status']!='PASS':raise ValueError('Incomplete study')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    colors={'A':'#287d9b','B':'#bb493a'}
    for ax,condition in zip(axes[:2],['legacy','active']):
        for index,group in enumerate(colors):
            rows=[r for r in report['runs'] if r['group']==group]
            for offset,row in enumerate(rows):
                rate=row['evaluations'][condition]['legal_return_rate']*100
                x=index+(offset-1)*.12
                ax.scatter(x,rate,color=colors[group],s=45)
                ax.annotate(str(row['seed']),(x,rate),xytext=(0,7),textcoords='offset points',fontsize=7,ha='center')
        zero=report['baseline'][condition]['legal_return_rate']*100
        ax.axhline(zero,color='black',ls='--',label=f'Zero action: {zero:g}%')
        ax.set(xticks=[0,1],xticklabels=['A: basic reward','B: + hit guidance'],ylim=(-5,115),ylabel='Legal returns (%)',
               title=f'{condition.capitalize()} serves: same 32 per policy')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    for mode,color,label in [('zero','#777777','Zero action'),('script','#48986a','Feasible joint motion')]:
        for index,trace in enumerate(development['runs'][mode]['traces'].values()):
            points=[]
            for row in trace:
                if points and row[0]<=points[-1][0]:break  # Exclude automatic reset.
                points.append(row)
            axes[2].plot([r[0] for r in points],[r[3] for r in points],color=color,alpha=.5,lw=1,label=label if index==0 else None)
    axes[2].set(xlabel='Time after serve (s)',ylabel='Racket TCP height (m)',title='Development witnesses (not PPO)')
    axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
    fig.suptitle('M7.4: 6,144 transitions per seed | Joint actions | PACE apex guidance unchanged')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=140);plt.close(fig)


def return_comparison(root, output, combined=False):
    import matplotlib.pyplot as plt
    folder=root/('m74d' if combined else 'm74c')
    report=json.loads((folder/'final_summary.json').read_text())
    gate=json.loads((folder/('combined_gate.json' if combined else 'return_gate.json')).read_text())
    if report['status']!='PASS' or gate['status']!='PASS':raise ValueError('Incomplete C comparison')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    colors={'initial':'#777777','A':'#287d9b','B':'#bb493a','C':'#48986a'}
    if combined:colors['D']='#9467bd'
    for ax,metric,title in zip(axes[:2],['hit_rate','legal_return_rate'],['Effective paddle contact','Legal return']):
        for index,group in enumerate(colors):
            rows=[r for r in report['runs'] if r['group']==group]
            for offset,row in enumerate(rows):
                value=row['evaluations']['active'][metric]*100
                x=index+(offset-1)*.15
                ax.scatter(x,value,color=colors[group],s=40)
                ax.annotate(str(row['seed'])[-1],(x,value),xytext=(0,6),textcoords='offset points',fontsize=8,ha='center')
        ax.axhline(report['baseline']['active'][metric]*100,color='black',ls='--',label='Zero action')
        ax.set(xticks=range(len(colors)),xticklabels=['Initial','A','B: hit','C: return']+(['D: both'] if combined else []),ylim=(-5,115),ylabel='Same 32 active serves (%)',title=title)
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    values=[gate['p95']['landing_m']*1000,gate['p95']['net_height_m']*1000]
    axes[2].bar(['Landing','Net height'],values,color='#287d9b')
    axes[2].axhline(20,color='black',ls='--',label='20 mm gate')
    for i,v in enumerate(values):axes[2].text(i,v+.4,f'{v:.2f}',ha='center')
    axes[2].set(ylabel='P95 error (mm)',ylim=(0,24),title='Pre-training free-flight validation')
    axes[2].legend(fontsize=8)
    fig.suptitle(('D combined guidance' if combined else 'C return guidance')+' | 6,144 transitions / seed | Labels 1/2/3 = seeds 7101/7102/7103')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=140);plt.close(fig)


def predictor_interface(root,output):
    import matplotlib.pyplot as plt
    folder=root/'m74e'
    offline=json.loads((folder/'offline/validation.json').read_text())
    smoke=json.loads((folder/'smoke.json').read_text())
    if offline['status']!='PASS' or smoke['status']!='PASS':raise ValueError('Incomplete E interface validation')
    history=smoke['runs'][0]['history']
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    keys=['learned_common','training_mean_common','estimated_physics','constant_velocity']
    values=[offline[k]['rmse_m']*1000 for k in keys]
    axes[0].bar(['MLP','Train mean','Physics','CV'],values,color=['#9467bd','#888888','#287d9b','#bb493a'])
    axes[0].set_yscale('log');axes[0].axhline(30,color='black',ls='--',label='30 mm gate')
    for i,v in enumerate(values):axes[0].text(i,v*1.12,f'{v:.1f}',ha='center',fontsize=8)
    axes[0].set(title='Offline test: common valid windows',ylabel='3D RMSE (mm, log scale)');axes[0].legend(fontsize=8)
    x=[r['iteration'] for r in history]
    coverage=[r['guidance_coverage'] for r in history]
    axes[1].bar(x,[100*r['hint_used_steps']/max(1,r['prehit_steps']) for r in coverage],color='#9467bd')
    axes[1].set(title='Online from scratch: actual hint use',xlabel='PPO update',ylabel='Pre-hit actions with valid hint (%)',ylim=(0,105),xticks=x)
    axes[2].plot(x,[r['release_events'] for r in coverage],'o-',label='Release events')
    axes[2].plot(x,[r['return_valid_events'] for r in coverage],'o-',label='Valid return forecasts')
    axes[2].set(title='Training return-guidance coverage',xlabel='PPO update',ylabel='Events in rollout',xticks=x);axes[2].legend(fontsize=8)
    fig.suptitle('E data + interface validation | offline model is NOT loaded into online smoke')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=140);plt.close(fig)


def predictor_curve(root,output):
    import matplotlib.pyplot as plt
    report=json.loads((root/'m74e/learning_curve.json').read_text())
    if report['status']!='PASS':raise ValueError('Incomplete predictor curve diagnosis')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for run in report['runs']:
        h=run['history'];x=[r['iteration'] for r in h];label=f"D behavior {run['behavior_seed']}"
        axes[0].plot(x,[r['validation_rmse_m'] for r in h],label=label)
        axes[1].plot(x,[r['update']['optimizer_steps'] for r in h],label=label)
        axes[2].plot(x,[100*r['coverage']['hint_steps']/max(1,r['coverage']['prehit_steps']) for r in h],label=label)
    axes[0].axhline(.03,color='black',ls='--',label='30 mm reference')
    axes[0].axhline(report['train_mean_validation_rmse_m'],color='gray',ls=':',label='Training mean apex')
    axes[0].set(yscale='log',title='Fixed validation accuracy',ylabel='3D RMSE (m, log scale)')
    axes[0].legend(fontsize=7)
    axes[1].set(title='Actual optimization work',ylabel='Cumulative optimizer steps')
    axes[2].set(title='Hints available before hit',ylabel='Pre-hit steps (%)',ylim=(0,105))
    for ax in axes:ax.set(xlabel='MSE update opportunity',xticks=[1,5,10,15,20]);ax.grid(alpha=.15)
    fig.suptitle('20-update predictor diagnosis | frozen D does NOT consume hints | no PPO updates')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=140);plt.close(fig)


def predictor_batch(root,output):
    import matplotlib.pyplot as plt
    report=json.loads((root/'m74e/batch_comparison.json').read_text())
    if report['status']!='PASS' or report['process_exit_code']!=0:
        raise ValueError('Incomplete batch comparison')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,run in zip(axes,report['runs']):
        h=run['history'];x=[r['iteration'] for r in h]
        for batch,color in [(1024,'#bd5c43'),(32,'#257b9e')]:
            rows=h if batch==1024 else [r['batch32'] for r in h]
            ax.plot(x,[r['validation_rmse_m']*1000 for r in rows],color=color,
                    label=f"Batch {batch}: {rows[-1]['update']['optimizer_steps']} optimizer steps")
        ax.axhline(30,color='black',ls='--',label='30 mm reference')
        ax.axhline(report['train_mean_validation_rmse_m']*1000,color='gray',ls=':',label='Training mean apex')
        ax.set(title=f"Fixed D behavior {run['behavior_seed']}",yscale='log',
               xlabel='MSE update opportunity',ylabel='Validation 3D RMSE (mm)',xticks=[1,5,10,15,20])
        ax.grid(alpha=.15);ax.legend(fontsize=7)
    fig.suptitle('Same data + initialization + 20 opportunities | batch size only | unequal compute | no PPO')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=140);plt.close(fig)


def predictor_audit(root,output):
    import numpy as np
    import matplotlib.pyplot as plt
    from airbot_tt.pace_predictor import windows
    import torch
    base=root/'m74e';r=json.loads((base/'data_audit/cross.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    seeds=[7101,7102,7103];colors=['#555555','#b75c48','#247ba0','#a88d39']
    for panel,dataset in enumerate(['buffer','validation']):
        ax=axes.flat[panel];xx=np.arange(3)
        for j,label in enumerate(['offline','batch1024','batch32','mean']):
            values=[]
            for seed in seeds:
                name=label if label in ('offline','mean') else f'online_{seed}_b{1024 if label=="batch1024" else 32}'
                row=next(z for z in r['cross'] if z['model']==name and z['dataset']==(f'buffer_{seed}' if dataset=='buffer' else 'validation'))
                values.append(row['teacher']['rmse_m']*1000)
            ax.bar(xx+(j-1.5)*.2,values,width=.19,color=colors[j],label=label)
        ax.axhline(30,color='black',ls='--',lw=1);ax.set(yscale='log',xticks=xx,xticklabels=seeds,
          ylabel='3D RMSE to teacher (mm)',xlabel='Frozen D behavior',title='Archived own buffer' if panel==0 else 'Fixed validation (1,223 windows)')
        ax.legend(fontsize=8,ncol=2);ax.grid(axis='y',alpha=.15)
    ax=axes[1,0];phases=['pre_bounce','across_bounce','post_bounce'];xx=np.arange(3)
    for j,seed in enumerate(seeds):
        values=[r['validation_by_phase'][f'online_{seed}_b32'][p]['teacher']['rmse_m']*1000 for p in phases]
        ax.bar(xx+(j-1)*.23,values,width=.22,label=f'Batch32 / D {seed}')
    ax.plot(xx,[r['validation_by_phase']['offline'][p]['teacher']['rmse_m']*1000 for p in phases],'ko--',label='Offline model')
    ax.axhline(30,color='black',ls=':',lw=1);ax.set(yscale='log',xticks=xx,xticklabels=['Before bounce','Across bounce','After bounce'],
      ylabel='3D RMSE to teacher (mm)',title='Validation error concentrates after bounce');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
    ax=axes[1,1];d=dict(np.load(base/'data/validation.npz'))
    vx,_,ends=windows(d['state'][:,:3],d['time'],d['serve'],np.ones(len(d['time']),bool),d['target'],d['valid'])
    v=vx.reshape(-1,5,3)[:,-1]
    state=torch.load(base/'batch_comparison/predictor_batch32_D_7102.pt',map_location='cpu',weights_only=True)
    b=torch.cat([a for a,y in state['buffer']]).numpy().reshape(-1,5,3)[:,-1]
    ax.scatter(v[:,0],v[:,2],s=8,alpha=.25,color='#ca8f35',label='Validation endpoints')
    ax.scatter(b[:,0],b[:,2],s=5,alpha=.25,color='#247ba0',label='D7102 archived buffer endpoints')
    ax.axvline(b[:,0].min(),ls='--',color='#247ba0',lw=1,label='Buffer minimum X')
    ax.set(xlabel='Local X (m)',ylabel='Local Z, ground origin (m)',title='Online buffer ends before the later flight');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Archived inference audit | teacher error throughout | no new training or heldout access')
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('.agents/runs/foundation/m7'))
    parser.add_argument('--output',type=Path,default=Path('doc/notes/images/m7_diagnostic.png'))
    parser.add_argument('--update-comparison',action='store_true',help='Plot paired update diagnostic and subsequent short run')
    parser.add_argument('--net-diagnostic',action='store_true',help='Plot measured release-to-net replay')
    parser.add_argument('--serve-comparison',action='store_true',help='Plot frozen serve-width comparison and legal landings')
    parser.add_argument('--guidance-comparison',action='store_true',help='Plot M7.4 A/B and separate feasibility witnesses')
    parser.add_argument('--return-comparison',action='store_true',help='Plot C comparison and physical return-prediction validation')
    parser.add_argument('--combined-comparison',action='store_true',help='Plot D combined-guidance comparison')
    parser.add_argument('--predictor-interface',action='store_true',help='Plot new-distribution E data and online coverage')
    parser.add_argument('--predictor-curve',action='store_true',help='Plot bounded 20-update predictor diagnosis')
    parser.add_argument('--predictor-batch',action='store_true',help='Plot paired batch 1024 versus 32 diagnosis')
    parser.add_argument('--predictor-audit',action='store_true',help='Plot archived predictor data/fit diagnosis')
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=args.runs
    if args.predictor_audit:
        predictor_audit(root,args.output)
        return
    if args.predictor_batch:
        predictor_batch(root,args.output)
        return
    if args.predictor_curve:
        predictor_curve(root,args.output)
        return
    if args.predictor_interface:
        predictor_interface(root,args.output)
        return
    if args.combined_comparison:
        return_comparison(root,args.output,combined=True)
        return
    if args.return_comparison:
        return_comparison(root,args.output)
        return
    if args.guidance_comparison:
        guidance_comparison(root,args.output)
        return
    if args.serve_comparison:
        serve_comparison(root,args.output)
        return
    if args.net_diagnostic:
        net_diagnostic(root,args.output)
        return
    if args.update_comparison:
        update_comparison(root,args.output)
        return
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
