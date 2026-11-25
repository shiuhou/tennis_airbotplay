"""Plot the completed Dataset-to-AIRBOT audit; never trains or replays physics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.colors import ListedColormap,BoundaryNorm

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=ROOT/'.agents/runs/foundation/deepmind_audit')
    parser.add_argument('--output',type=Path,default=ROOT/'doc/notes/images/deepmind_feasibility.png')
    args=parser.parse_args();base=args.runs
    rows=json.loads((base/'transformed.json').read_text());summary=json.loads((base/'summary.json').read_text())
    assert summary['status']=='PASS'
    from matplotlib import font_manager
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(15,10),facecolor='#f8fafc');colors=['#db974a','#3e7b99']
    ax=axes[0,0];ax.add_patch(Rectangle((-1.37,-.7625),2.74,1.525,facecolor='#e3efec',edgecolor='#78928a'))
    ax.axvline(0,color='#61717b',lw=1)
    for kind,color in zip(('serves','rallies'),colors):
        x=np.array([r['state'] for r in rows if r['kind']==kind]);ax.scatter(x[:,0],x[:,1],s=2,alpha=.15,color=color,label=f'{kind}: {len(x):,}')
    ax.scatter([-1.275],[0],s=60,color='#111827',marker='s',label='固定底座')
    ax.scatter([.5],[0],s=60,color='#c7473d',marker='D',label='原人工發球位置')
    ax.set(xlabel='AIRBOT X：桌長 (m)',ylabel='AIRBOT Y：桌寬 (m)',xlim=(-1.5,2.5),ylim=(-1,1),title='初始位置：統一來球方向後');ax.legend(fontsize=8,loc='lower left',ncol=2);ax.grid(alpha=.15)
    ax=axes[0,1]
    for kind,color in zip(('serves','rallies'),colors):
        x=np.array([r['state'][3:] for r in rows if r['kind']==kind]);ax.hist(np.linalg.norm(x,axis=1),bins=np.linspace(2,14,49),density=True,histtype='step',lw=2,color=color,label=kind)
    ax.axvline(np.linalg.norm([-3.3,0,-1.4]),ls='--',color='#c7473d',label='原人工來球速率')
    ax.set(xlabel='初始速率 (m/s)',ylabel='密度',title='速率中位數 5.49 m/s；5–95%：3.90–7.68');ax.legend(fontsize=8);ax.grid(alpha=.15)
    ax=axes[0,2];spin=np.linalg.norm([r['omega'] for r in rows],axis=1)
    ax.hist(spin,bins=np.linspace(0,370,75),color='#7e8fbc',alpha=.85);ax.axvline(10,color='#c7473d',ls='--',label='低旋篩選 ≤10 rad/s')
    ax.set(xlabel='角速度模長 (rad/s)',ylabel='筆數（對數）',yscale='log',title='旋轉中位數 48.7 rad/s；沒有完全零旋紀錄');ax.legend(fontsize=8);ax.grid(alpha=.15)
    ax=axes[1,0];arr=summary['arrivals'];state=np.array([r['state'] for r in arr]);times=np.array([r['time_s'] for r in arr])
    dots=ax.scatter(state[:,1],state[:,2],c=times,s=16,cmap='viridis',alpha=.8,vmin=.3,vmax=1.)
    from airbot_tt.tool import racket_model
    model=racket_model(ROOT/'configs/racket.json');nom=json.loads((ROOT/'configs/strike.json').read_text())['ready_q_rad']
    shot=json.loads((ROOT/'configs/single_shot.json').read_text());ready_z=model.fk(nom)[2,3]+shot['base_position_m'][2]
    ax.axhline(ready_z,ls='--',color='#c7473d',label='原準備拍心高度（非可達域）')
    ax.set(xlabel='攔截平面 Y (m)',ylabel='離地高度 Z (m)',title='到 X=−1.1m 的位置；顏色=到達秒數');ax.legend(fontsize=7,loc='lower right');ax.grid(alpha=.15)
    bar=fig.colorbar(dots,ax=ax,fraction=.04,pad=.02);bar.ax.set_title('秒',fontsize=8)
    ax=axes[1,1];counts=[summary['raw_count'],summary['low_spin_count'],summary['eligible_count'],summary['position_command_candidates'],summary['certified_return_count']]
    labels=['原始初始狀態','保守低旋篩選','符合目前任務階段','命令預檢候選','實際合法回球見證']
    ax.barh(range(5),counts,color=['#8c9caa','#768f9d','#5f91a0','#3685a0','#b84d4a'])
    for i,n in enumerate(counts):ax.text(max(n,1)*1.15,i,f'{n:,}',va='center',weight='bold')
    ax.set(yticks=range(5),yticklabels=labels,xscale='log',xlim=(.5,60000),title='分層審查：各層分母不同，不是回球率');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    ax=axes[1,2];matrix=np.array(summary['matrix_hit_legal']);value=matrix[:,:,0]+matrix[:,:,1]
    cmap=ListedColormap(['#e5e7eb','#3b82a0','#27a36f']);norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    ax.imshow(value,cmap=cmap,norm=norm,aspect='auto')
    for i in range(len(value)):
        for j in range(len(value[0])):
            if value[i,j]:ax.text(j,i,'碰' if value[i,j]==1 else '合法',ha='center',va='center',fontsize=8,color='white')
    ys=[name.replace('script_','腳本 ').replace('M5_reference_via_M7','M5固定參考') for name in summary['matrix_labels']]
    ax.set(xticks=range(len(summary['matrix_serve_ids'])),xticklabels=summary['matrix_serve_ids'],yticks=range(len(ys)),yticklabels=ys,
       xlabel='原始來球 ID',title='固定動作交叉重播：能碰球，仍沒有合法回球')
    ax.tick_params(axis='x',labelrotation=45,labelsize=8);ax.tick_params(axis='y',labelsize=8)
    ax.legend(handles=[Patch(color='#e5e7eb',label='未有效碰拍'),Patch(color='#3b82a0',label='碰到但回球失敗'),Patch(color='#27a36f',label='合法回球')],fontsize=7,loc='upper left',bbox_to_anchor=(0,-.23),ncol=2)
    fig.suptitle('DeepMind → AIRBOT：資料分布很廣，本輪尚未找到合法回球見證',x=.055,ha='left',fontsize=19,weight='bold')
    fig.text(.055,.925,'343筆低旋／階段相容候選完成有限搜尋；8個專用動作能碰球，7次落回己方、1次出界。未改 PPO，也未接入訓練。',color='#475569',fontsize=11)
    fig.text(.055,.025,'原始資料：Google LLC (2025), Competitive Robot Table Tennis: Initial Ball States, CC BY 4.0，commit 54cf18e4。\n來源 github.com/google-deepmind/competitive_robot_table_tennis；座標轉換／篩選／圖為本工程衍生。到達／反彈為零旋近似模型結果，非完整實測軌跡。未找到見證不等於物理不可達。',fontsize=8,color='#475569')
    fig.subplots_adjust(top=.865,bottom=.17,hspace=.4,wspace=.45);args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=140,facecolor=fig.get_facecolor());plt.close(fig)


if __name__=='__main__':main()
