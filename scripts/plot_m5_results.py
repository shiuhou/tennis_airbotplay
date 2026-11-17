"""Plot measured M5 landing points and optional recorded PhysX trajectories."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np


def plot(validation, output, recording=None):
    report=json.loads(Path(validation).read_text())
    if report['status']!='PASS':raise ValueError('Expected completed validation report.')
    if recording:
        movie=json.loads(Path(recording).read_text())
        if movie['status']!='PASS':raise ValueError('Expected completed recording.')
    else:movie=None
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#07869b','#ba7900','#b8458a'];labels=['左側','中央','右側']
    fig=plt.figure(figsize=(12,8),facecolor='#f7f9fc')
    grid=fig.add_gridspec(3,2,width_ratios=[1.35,1],hspace=.6,wspace=.4)
    ax=fig.add_subplot(grid[:,0]);ax.set_facecolor('#f7f9fc')
    ax.add_patch(Rectangle((-.7625,-1.37),1.525,2.74,facecolor='#e4f1ed',edgecolor='#58756c',lw=2))
    ax.plot([-.7625,.7625],[0,0],color='#38444b',lw=3)
    ax.text(.80,0,'球網',va='center')
    ax.scatter([0],[-1.275],s=200,marker='s',color='#4b5563',zorder=6)
    ax.annotate('桌端固定底座',(0,-1.275),xytext=(-.88,-1.61),arrowprops={'arrowstyle':'-','color':'#4b5563'})
    for i,goal in enumerate(report['config']['goals_xy_m']):
        gx,gy=goal;color=colors[i]
        trials=[t for t in report['trials'] if t['goal']==goal]
        points=np.array([next(e['position_m'] for e in t['events'] if e['kind']=='legal_return')[:2] for t in trials if t['outcome']=='legal_return'])
        ax.add_patch(Rectangle((gy-.2,gx-.2),.4,.4,facecolor=color,alpha=.14,edgecolor=color,lw=2))
        ax.scatter(points[:,1],points[:,0],s=12,color=color,zorder=5)
        ax.text(gy,1.08,labels[i],color=color,ha='center',weight='bold')
        zoom=fig.add_subplot(grid[i,1]);zoom.set_facecolor('white')
        zoom.axhline(0,color='#cbd5e1',lw=.8);zoom.axvline(0,color='#cbd5e1',lw=.8)
        zoom.scatter((points[:,1]-gy)*1000,(points[:,0]-gx)*1000,s=40,color=color,alpha=.8,edgecolors='white',lw=.6)
        zoom.scatter([0],[0],marker='+',s=130,color='#111827',lw=1.5)
        zoom.set(xlim=(-35,35),ylim=(-35,35),xlabel='橫向偏差 (mm)',ylabel='遠近偏差 (mm)')
        zoom.set_title(f'{labels[i]}目標：{len(points)}/{len(trials)} 合法且落桌；框內 {sum(t["target_success"] for t in trials)}/{len(trials)}',loc='left',color=color,fontsize=11)
        zoom.grid(alpha=.12)
        if movie:
            measured=movie['trials'][i]
            samples=measured['samples'];p=np.array([s['ball'][:3] for s in samples]);times=np.array([s['time_s'] for s in samples])
            hit=next(e for e in measured['events'] if e['kind']=='paddle_contact')
            before=times<hit['time_s'];after=~before
            if i==0:
                ax.plot(p[before,1],p[before,0],color='#65758c',ls='--',lw=1.5,label='來球（實測）')
                ax.scatter([hit['position_m'][1]],[hit['position_m'][0]],s=80,marker='*',color='#111827',zorder=7)
                ax.annotate('擊球', (hit['position_m'][1],hit['position_m'][0]),xytext=(.36,-1.12),arrowprops={'arrowstyle':'->'},fontsize=11)
            ax.plot(p[after,1],p[after,0],color=color,lw=2,label=f'{labels[i]}回球')
    ax.set(xlim=(-1.,1.),ylim=(-1.8,1.45),xlabel='橫向 Y (m)',ylabel='跨網 X (m)')
    ax.set_aspect('equal');ax.set_title('球桌俯視：三條回球路線與目標框',loc='left',fontsize=13,pad=12)
    if movie:ax.legend(loc='lower right',fontsize=9,frameon=False)
    fig.suptitle('M5 回球實測｜36 次驗證，落點集中在目標中心附近',x=.08,ha='left',fontsize=18,weight='bold')
    fig.text(.08,.925,'色框為 40 × 40 cm 目標區；右側放大至中心 ±35 mm，黑色十字是目標中心。',fontsize=11,color='#475569')
    fig.text(.08,.025,'點：獨立驗證的 36 次落點。線：另錄同一來球、三目標的 PhysX 實測軌跡（非預測線）。\n固定底座／預先規劃動作／限定高反彈來球；本圖不代表真機或任意來球表現。' if movie else '點為獨立驗證實測落點；固定底座、限定高反彈來球。',fontsize=10,color='#475569')
    fig.subplots_adjust(top=.86,bottom=.12)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160,facecolor=fig.get_facecolor());plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation',required=True,type=Path)
    parser.add_argument('--recording',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();plot(args.validation,args.output,args.recording)
