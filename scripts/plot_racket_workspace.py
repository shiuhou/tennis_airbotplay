"""Render static sampled racket workspace; does not control a robot."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]


def comparison(args):
    report=json.loads((args.runs/'comparison.json').read_text())
    fig,axes=plt.subplots(3,3,figsize=(15,11),sharex=True,sharey=True,facecolor='#f8fafc')
    for row in report['rows']:
        r=report['protocol']['retreat_m'].index(row['retreat_m']);c=report['protocol']['radii_rad'].index(row['radius_rad'])
        ax=axes[r,c];path=Path(row['path']);path=path if path.is_absolute() else ROOT/path
        d=np.load(path.parent/'samples.npz');a=d['current_contact_xyz']
        a=a[np.linspace(0,len(a)-1,min(16000,len(a))).astype(int)]
        ax.scatter(a[:,0],a[:,2],s=2,alpha=.22,c=['#83b3ca','#248caa','#32667f'][c])
        ax.plot([-1.37,.15],[.76,.76],color='#438269',lw=2)
        ax.scatter(-1.275-row['retreat_m'],.76,marker='s',s=45,c='#172c40')
        ax.axvline(-1.37,color='#438269',ls='--',lw=.8)
        ax.set(xlim=(-2.2,-.55),ylim=(.42,1.75));ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
        result=row['result'];tip=result['contact_bounds_m'][1][0]
        ax.set_title(f"±{row['radius_rad']:.1f} rad；後退 {row['retreat_m']*100:.0f} cm",fontsize=12)
        ax.text(.03,.96,f"最前 X={tip:.3f} m\n碰撞姿態 {result['sampled_collision']}/{result['sampled']}",transform=ax.transAxes,va='top',fontsize=10)
        if r==2:ax.set_xlabel('X：桌長方向 (m)')
        if c==0:ax.set_ylabel('Z：離地高度 (m)')
    fig.suptitle('動作半徑 × 底座後退：更大範圍不等於更容易接球',fontsize=20,x=.06,ha='left',y=.98,weight='bold')
    fig.text(.06,.935,f"同一組 {2**report['protocol']['power']:,} Sobol 樣本＋ready，每組重算碰撞；點為正拍面潛在球心接觸位置的側視投影。",fontsize=11)
    fig.text(.06,.055,'綠線為桌面，虛線為己方桌端 X=−1.37m；底座後退組假設另有支架，保持 Z=0.76m，支架本身未建模。\n後退讓範圍向桌後移，可能減少碰桌，卻犧牲向前接球距離。這裡未驗證限時到達、拍速、合法回球或實體支撐。',fontsize=11,color='#475569')
    fig.subplots_adjust(top=.87,bottom=.14,hspace=.32,wspace=.15)
    output=args.output or ROOT/'doc/notes/images/racket_workspace_comparison.png'
    fig.savefig(output,dpi=140,facecolor=fig.get_facecolor());plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=ROOT/'.agents/runs/foundation/workspace')
    p.add_argument('--output',type=Path)
    p.add_argument('--compare',action='store_true')
    args=p.parse_args();report=json.loads((args.runs/'workspace.json').read_text());data=np.load(args.runs/'samples.npz')
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':11,'axes.unicode_minus':False})
    if args.compare:return comparison(args)
    args.output=args.output or ROOT/'doc/notes/images/racket_workspace.png'
    base=np.array(report['protocol']['base_position_m'])
    full=data['full_contact_xyz'];current=data['current_contact_xyz'];tcp=data['current_tcp']
    # Uniform deterministic visual thinning only; bounds/statistics use all samples.
    def thin(a,n=18000):return a[np.linspace(0,len(a)-1,min(len(a),n)).astype(int)]
    full,current=thin(full),thin(current)
    fig=plt.figure(figsize=(14,11),facecolor='#f8fafc')
    dims=[(0,1,'俯視：桌長 × 桌寬'),(0,2,'側視：桌長 × 離地高度'),(1,2,'端視：桌寬 × 離地高度')]
    axes=[]
    labels=['X：桌長方向 (m)','Y：桌寬方向 (m)','Z：離地高度 (m)']
    for k,(i,j,title) in enumerate(dims):
        ax=fig.add_subplot(2,2,k+1);axes.append(ax)
        ax.scatter(full[:,i],full[:,j],s=2,c='#bcc5ce',alpha=.35,rasterized=True)
        ax.scatter(current[:,i],current[:,j],s=2,c='#248caa',alpha=.24,rasterized=True)
        ax.scatter(tcp[:,i],tcp[:,j],s=2,c='#e89936',alpha=.4,rasterized=True)
        ax.scatter(base[i],base[j],s=75,c='#192b3e',marker='s',zorder=5)
        if j==1:
            ax.add_patch(Rectangle((-1.37,-.7625),2.74,1.525,fill=False,ec='#438269',lw=1.6))
            ax.plot([0,0],[-.7625,.7625],c='#438269',ls='--')
        elif i==0:
            ax.plot([-1.37,1.37],[.76,.76],c='#438269',lw=2)
            ax.plot([0,0],[.76,.9125],c='#438269',lw=2)
        else:
            ax.plot([-.7625,.7625],[.76,.76],c='#438269',lw=2)
        ax.set(xlabel=labels[i],ylabel=labels[j],title=title);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
    ax=fig.add_subplot(2,2,4,projection='3d')
    a=thin(full,6000);b=thin(current,6000)
    ax.scatter(*a.T,s=1,c='#bcc5ce',alpha=.12)
    ax.scatter(*b.T,s=2,c='#248caa',alpha=.3)
    ax.scatter(*tcp.T,s=1,c='#e89936',alpha=.2)
    table=np.array([[-1.37,-.7625,.76],[1.37,-.7625,.76],[1.37,.7625,.76],[-1.37,.7625,.76],[-1.37,-.7625,.76]])
    ax.plot(*table.T,c='#438269');ax.scatter(*base,s=65,c='#192b3e',marker='s')
    ax.view_init(elev=24,azim=-55);ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='空間位置：底座固定在己方桌端')
    ax.set_box_aspect((3.5,2,2))
    handles=[Line2D([],[],ls='',marker='o',color=c,label=t,markersize=7) for c,t in
        [('#bcc5ce','完整關節範圍：潛在球心接觸點'),('#248caa','目前 ±0.6 rad：潛在球心接觸點'),('#e89936','目前 ±0.6 rad：拍心 TCP')]]
    handles.append(Line2D([],[],ls='',marker='s',color='#192b3e',label='底座',markersize=7))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.925),ncol=2,frameon=False,fontsize=11)
    fig.suptitle('目前安裝方式：球拍能擺到哪裡？球心可能在哪裡碰到正拍面？',fontsize=19,x=.06,ha='left',y=.985,weight='bold')
    fig.text(.06,.945,f"原 URDF／固定器／球拍 CAD；每範圍 {report['protocol']['poses_per_box']:,} 個 Sobol 姿態＋ready，排除手臂自身、桌、網與地面碰撞。",fontsize=11,color='#475569')
    fig.text(.06,.035,'灰／藍點是正拍面樣本沿法向外移球半徑 20 mm；不含拍柄、拍邊或背面。各點有對應關節姿態及法向。\n這是靜態採樣，不是可及時接球或合法回球的保證；投影重疊不代表相同3D位置，空白也不代表不可達。\n底座採倉庫設定 '+str(base.tolist())+' m，尚未實測對位。其他安裝位置須重算碰撞，不能只平移這張圖。',fontsize=10,color='#475569')
    fig.subplots_adjust(top=.835,bottom=.14,wspace=.25,hspace=.3)
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=145,facecolor=fig.get_facecolor());plt.close(fig)


if __name__=='__main__':main()
