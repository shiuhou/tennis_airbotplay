"""Plot/animate measured same-serve M5/zero/PPO traces, using simplified body outlines."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
LABELS=['M5 預先規劃','零動作','PPO A7101：合法','PPO B7101：失敗']
COLORS=['#087f8c','#667085','#2966c0','#c44942']


def load(folder):
    reports=[json.loads((folder/f'{name}.json').read_text()) for name in ('m5','policies')]
    gates=[json.loads((folder.parent/f'motion_{name}.json').read_text()) for name in ('m5','policies')]
    assert all(r['status']=='PASS' for r in reports+gates) and all(r['process_exit_code']==0 for r in gates)
    assert reports[0]['protocol']['serve']==reports[1]['protocol']['serve']
    assert reports[0]['protocol']['source_sha256']==reports[1]['protocol']['source_sha256']
    runs=reports[0]['runs']+reports[1]['runs']
    assert [r['name'] for r in runs]==['M5','zero','A_7101','B_7101']
    return runs


def paddle(row):
    p=np.asarray(row['tcp']);rotation=Rotation.from_quat(np.asarray(row['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
    theta=np.linspace(0,2*np.pi,41)
    head=p+np.outer(.085*np.cos(theta),rotation[:,0])+np.outer(.075*np.sin(theta),rotation[:,1])
    handle=p+np.outer([-.08,-.23],rotation[:,0])
    return head,handle


def style():
    from matplotlib import font_manager
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})


def figure(runs,output):
    style();fig=plt.figure(figsize=(16,9),facecolor='#f8fafc')
    grid=fig.add_gridspec(2,4,height_ratios=[1.05,1],hspace=.4,wspace=.25)
    for i,(run,color,label) in enumerate(zip(runs,COLORS,LABELS)):
        ax=fig.add_subplot(grid[0,i]);trace=run['trace'];ball=np.array([z['ball'][:3] for z in trace]);tcp=np.array([z['tcp'] for z in trace])
        hit=run['impact'];t=np.array([z['t'] for z in trace]);prior=t<hit['time_s']
        ax.fill_between([-1.37,1.37],.68,.76,color='#cde1df');ax.plot([0,0],[.76,.9125],color='#364152',lw=3)
        ax.plot(ball[prior,0],ball[prior,2],ls='--',color='#b3bbc5',lw=1.4)
        ax.plot(ball[~prior,0],ball[~prior,2],color=color,lw=2.5,label='碰拍後球路')
        ax.plot(tcp[:,0],tcp[:,2],color='#169b62',lw=1.4,label='球拍中心軌跡')
        row=hit['pre_sample'];head,handle=paddle(row);ax.plot(head[:,0],head[:,2],color=color,lw=2);ax.plot(handle[:,0],handle[:,2],color=color,lw=3)
        p=np.array(row['tcp']);n=np.array(row['normal']);v=np.array(row['tcp_velocity'])
        ax.annotate('',xy=(p+.22*n)[[0,2]],xytext=p[[0,2]],arrowprops=dict(arrowstyle='->',color='#c026a2',lw=2))
        ax.annotate('',xy=(p+.22*v)[[0,2]],xytext=p[[0,2]],arrowprops=dict(arrowstyle='->',color='#16a34a',lw=2))
        land=run['post_hit_table_contact'];lp=np.array(land['ball'][:3]);ax.scatter(lp[0],lp[2],marker='X',s=65,color=color,zorder=5)
        ax.set(xlim=(-1.5,1.4),ylim=(.65,2.3),xlabel='X：己方 ← → 對手 (m)',ylabel='離地高度 Z (m)' if i==0 else '',title=label)
        ax.grid(alpha=.15)
        text=f"碰拍前拍速 {hit['tcp_speed_m_s']:.2f} m/s\n拍速 X = {row['tcp_velocity'][0]:+.2f} m/s\n落桌 X/Y = {lp[0]:+.3f} / {lp[1]:+.3f} m"
        ax.text(.03,.98,text,transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        if i==0:ax.legend(loc='upper right',bbox_to_anchor=(1, .71),fontsize=8)
    ax=fig.add_subplot(grid[1,:2]);ax.add_patch(Rectangle((-1.37,-.7625),2.74,1.525,fc='#e1eeeb',ec='#708c86'))
    ax.plot([0,0],[-.7625,.7625],color='#364152',lw=2)
    ax.add_patch(Rectangle((.55,-.2),.4,.4,fc='#078f70',alpha=.12,ec='#078f70',lw=2));ax.plot(.75,0,'+',ms=12,color='#078f70')
    for run,color,label in zip(runs,COLORS,LABELS):
        trace=run['trace'];p=np.array([z['ball'][:3] for z in trace]);t=np.array([z['t'] for z in trace]);mask=t>=run['impact']['time_s']
        ax.plot(p[mask,0],p[mask,1],color=color,label=label,lw=2)
        land=run['landing_xy_m'];ax.scatter(*land,marker='X',s=75,color=color,zorder=5)
        offset=(7,12) if run['name']=='M5' else ((7,-17) if run['name']=='zero' else (7,7))
        ax.annotate(f"({land[0]:.2f}, {land[1]:.2f})",land,xytext=offset,textcoords='offset points',color=color,fontsize=9)
    ax.set(xlim=(-1.5,1.4),ylim=(-.85,.85),xlabel='X (m)',ylabel='Y (m)',title='實際落桌：綠框為 M5 中央目標；合法回球 ≠ 命中目標')
    ax.set_aspect('equal');ax.legend(fontsize=8,ncol=2,loc='lower right');ax.grid(alpha=.15)
    ax=fig.add_subplot(grid[1,2:])
    for run,color,label in zip(runs,COLORS,LABELS):
        t=[z['t'] for z in run['trace']];v=np.array([z['tcp_velocity'] for z in run['trace']])
        ax.plot(t,v[:,0],color=color,lw=1.8,label=label)
        hit=run['impact'];ax.scatter(hit['time_s'],hit['pre_sample']['tcp_velocity'][0],color=color,s=50,zorder=5)
    ax.axhline(0,color='#7d8590',lw=1);ax.axvspan(.58,.65,color='#e9c46a',alpha=.15)
    ax.set(xlabel='發球後時間 (s)',ylabel='拍速 X 分量 (m/s)',title='拍速的方向：正值朝對手；負值在退讓')
    ax.grid(alpha=.15);ax.legend(fontsize=9,ncol=2)
    fig.suptitle('同一顆球，四種動作｜960 Hz PhysX 實測重繪',x=.06,ha='left',fontsize=20,weight='bold')
    fig.text(.06,.925,'上排側視：灰虛線為來球；彩線為碰拍後球路；綠線／箭頭為拍心移動／速度；洋紅箭頭為拍面法線。',color='#475569')
    fig.text(.06,.025,'拍速取首次碰拍前一個物理樣本；落桌取碰拍後首次桌接觸（含失敗落回己方）。球拍外形簡化，姿態與軌跡來自實測。\n這是原 M5 validation_0 的單球行為示例，不是成功率排名；A7101 在此合法，但未命中 M5 中央目標。',fontsize=10,color='#475569')
    fig.subplots_adjust(top=.87,bottom=.13);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=140,facecolor=fig.get_facecolor());plt.close(fig)


def video(runs,output):
    from matplotlib.animation import FFMpegWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio_ffmpeg
    style();plt.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
    fig=plt.figure(figsize=(12.8,8),facecolor='#f8fafc');axes=[];artists=[]
    for i,(run,color,label) in enumerate(zip(runs,COLORS,LABELS)):
        ax=fig.add_subplot(2,2,i+1,projection='3d');axes.append(ax)
        table=[(-1.37,-.7625,.76),(1.37,-.7625,.76),(1.37,.7625,.76),(-1.37,.7625,.76)]
        ax.add_collection3d(Poly3DCollection([table],facecolors='#d5e9e4',edgecolors='#7faaa0',alpha=.65))
        net=[(0,-.7625,.76),(0,.7625,.76),(0,.7625,.9125),(0,-.7625,.9125)]
        ax.add_collection3d(Poly3DCollection([net],facecolors='#556577',alpha=.25))
        ax.plot([.55,.95,.95,.55,.55],[-.2,-.2,.2,.2,-.2],[.763]*5,color='#008460',lw=1.2)
        ax.set(xlim=(-1.55,1.45),ylim=(-.85,.85),zlim=(.65,2.25),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=label)
        ax.set_box_aspect((3,1.7,1.6));ax.view_init(elev=23,azim=-66);ax.tick_params(labelsize=7)
        arm,=ax.plot([],[],[],color='#5a6069',lw=4,marker='o',ms=3)
        ball,=ax.plot([],[],[],marker='o',ms=6,color='#f28c28',ls='')
        tail,=ax.plot([],[],[],color=color,lw=1.5,alpha=.8)
        head,=ax.plot([],[],[],color=color,lw=2.5)
        handle,=ax.plot([],[],[],color=color,lw=3)
        normal,=ax.plot([],[],[],color='#c026a2',lw=2)
        speed,=ax.plot([],[],[],color='#16a34a',lw=2)
        landing,=ax.plot([],[],[],marker='X',ms=7,color=color,ls='')
        text=ax.text2D(.01,.90,'',transform=ax.transAxes,fontsize=9)
        artists.append((arm,ball,tail,head,handle,normal,speed,landing,text))
    fig.subplots_adjust(left=.02,right=.98,top=.91,bottom=.07,wspace=.02,hspace=.16)
    fig.text(.04,.95,'同球同步重播｜0.25 倍速｜依實測姿態重繪，機械臂與球拍為簡化外形',fontsize=14,weight='bold')
    fig.text(.04,.025,'橙球：實際球位置；彩線：球路；洋紅：拍面法線；綠：TCP 速度方向。每組到終止時定格。',fontsize=10,color='#475569')
    zooms=[]
    for i,(run,color) in enumerate(zip(runs,COLORS)):
        p=np.array(run['impact']['pre_sample']['tcp'])
        zoom=fig.add_axes([.378 if i%2==0 else .868,.645 if i<2 else .19,.108,.18],facecolor='#f1f5f9')
        zoom.set(xlim=(p[0]-.22,p[0]+.25),ylim=(p[2]-.23,p[2]+.23),title='碰拍區側視')
        zoom.set_aspect('equal');zoom.set_xticks([]);zoom.set_yticks([]);zoom.title.set_fontsize(8)
        zhead,=zoom.plot([],[],color=color,lw=2);zhandle,=zoom.plot([],[],color=color,lw=2)
        zball,=zoom.plot([],[],'o',color='#f28c28',ms=7)
        zn,=zoom.plot([],[],color='#c026a2',lw=1.5);zv,=zoom.plot([],[],color='#16a34a',lw=1.5)
        zooms.append((zhead,zhandle,zball,zn,zv))
    fps=15;source_hz=60;duration=max(r['duration_s'] for r in runs)
    frames=int(np.ceil(duration*source_hz))+1;output.parent.mkdir(parents=True,exist_ok=True)
    writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','24','-movflags','+faststart'])
    with writer.saving(fig,str(output),dpi=100):
        for frame in range(frames+fps):
            t=min(frame/source_hz,duration)
            for run,art,zoom in zip(runs,artists,zooms):
                trace=run['trace'];times=np.array([r['t'] for r in trace]);j=max(0,np.searchsorted(times,t,side='right')-1)
                row=trace[j];arm,ball,tail,head,handle,normal,speed,landing,text=art
                def line(artist,points):artist.set_data_3d(*np.asarray(points).T)
                line(arm,row['arm']);line(ball,[row['ball'][:3]])
                line(tail,[z['ball'][:3] for z in trace[:j+1]])
                h,b=paddle(row);line(head,h);line(handle,b)
                p=np.array(row['tcp']);line(normal,[p,p+.18*np.array(row['normal'])]);line(speed,[p,p+.15*np.array(row['tcp_velocity'])])
                zh,zb,zball,zn,zv=zoom
                for artist,points in [(zh,h),(zb,b),(zball,[row['ball'][:3]]),(zn,[p,p+.15*np.array(row['normal'])]),(zv,[p,p+.15*np.array(row['tcp_velocity'])])]:
                    points=np.asarray(points);artist.set_data(points[:,0],points[:,2])
                ended=t>=run['duration_s'];land=run['landing_xy_m']
                if ended and land:line(landing,[[*land,.78]])
                current=f"t={min(t,run['duration_s']):.3f}s  拍速={np.linalg.norm(row['tcp_velocity']):.2f} m/s"
                if ended:current+=f"\n{'合法回球' if run['outcome']=='legal_return' else '落回己方：失敗'}  落桌({land[0]:.2f}, {land[1]:.2f})"
                elif t>=run['impact']['time_s']:current+='\n已碰拍'
                text.set_text(current)
            writer.grab_frame()
    plt.close(fig)
    return dict(fps=fps,frames=frames+fps,duration_s=(frames+fps)/fps,playback_speed=.25,width=1280,height=800)


def bc_video(report, output, poster=None):
    """Three synchronized measured runs; no interpolation of missing outcomes."""
    from matplotlib.animation import FFMpegWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import imageio_ffmpeg
    if report.get('status') != 'PASS' or report.get('process_exit_code') != 0:
        raise ValueError('BC replay must have a completed PASS report and exit 0')
    runs = report['runs']
    if [run['name'] for run in runs] != ['script', 'frozen_bc', 'closed_loop_bc']:
        raise ValueError('Expected original script, frozen BC actions, and closed-loop BC')
    for run in runs:
        trace = run['trace']
        if len(trace) < 2 or any(a['t'] >= b['t'] for a, b in zip(trace, trace[1:])):
            raise ValueError('Replay samples must advance in time')
        np.testing.assert_allclose(trace[0]['ball'], runs[0]['trace'][0]['ball'], atol=2e-6, rtol=0)
    style()
    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    colors = ['#087f8c', '#326ab4', '#bd4c40']
    titles = ['原接觸腳本', 'BC 預先算好後播放', 'BC 自主控制']
    subtitles = ['按既有腳本執行', '不讀即時觀測', '每一步讀即時觀測']
    outcome_names = {'wrong_bounce': '落回己方', 'unexpected_contact': '非預期接觸',
        'arm_fault': '保護性停止', 'miss': '漏接', 'out': '出界', 'illegal_hit': '非法接觸',
        'legal_return': '合法回球', 'timeout': '超時', 'net': '碰網'}
    fig = plt.figure(figsize=(16, 9), facecolor='#f8fafc')
    grid = fig.add_gridspec(2, 3, left=.035, right=.985, top=.815, bottom=.12,
                          hspace=.10, wspace=.16, height_ratios=[1.22, 1])
    fig.text(.035, .955, '同一顆球：預先播放與自主控制的差別', fontsize=23, weight='bold', color='#172b42')
    clock = fig.text(.985, .957, '', fontsize=17, ha='right', color='#172b42')
    fig.text(.035, .914,
        f"來球 ID {report['selected_raw_id']} · BC seed {report['seed']} · 0.25 倍速 · 960 Hz PhysX 實測重繪，簡化外形",
        fontsize=12, color='#475569')
    fig.text(.035, .050, '橙色：球　洋紅：拍面法線　綠色：拍心速度方向　灰虛線：同時刻原腳本拍面',
             fontsize=12, color='#475569')
    fig.text(.035, .022, '下排放大己方球拍區；每組回合結束後定格。這是同一案例的行為比較，接觸成功與合法回球分開標示。',
             fontsize=11, color='#475569')
    artists = []
    for i, (run, color, title, subtitle) in enumerate(zip(runs, colors, titles, subtitles)):
        ax = fig.add_subplot(grid[0, i], projection='3d')
        table = [(-1.37, -.7625, .76), (1.37, -.7625, .76), (1.37, .7625, .76), (-1.37, .7625, .76)]
        net = [(0, -.7625, .76), (0, .7625, .76), (0, .7625, .9125), (0, -.7625, .9125)]
        ax.add_collection3d(Poly3DCollection([table], facecolors='#cce4de', edgecolors='#779e95', alpha=.48))
        ax.add_collection3d(Poly3DCollection([net], facecolors='#5b6c80', alpha=.30))
        ax.set(xlim=(-1.8, 1.45), ylim=(-.85, .85), zlim=(.6, 1.7))
        ax.set_box_aspect((3.25, 1.7, 1.1)); ax.view_init(elev=25, azim=-62)
        ax.tick_params(labelsize=7, pad=0)
        ax.set_title(title + '\n' + subtitle, fontsize=14, color=color, pad=13)
        arm, = ax.plot([], [], [], color='#596675', lw=4, marker='o', ms=3)
        ball, = ax.plot([], [], [], marker='o', ms=7, color='#f28c28', ls='')
        tail, = ax.plot([], [], [], color='#e3a054', lw=1.4)
        head, = ax.plot([], [], [], color=color, lw=2.6)
        handle, = ax.plot([], [], [], color=color, lw=3)
        normal, = ax.plot([], [], [], color='#c026a2', lw=2)
        velocity, = ax.plot([], [], [], color='#159663', lw=2)
        status = ax.text2D(.02, .02, '', transform=ax.transAxes, fontsize=11, color='#26394e',
                           bbox=dict(facecolor='#f8fafc', edgecolor='none', alpha=.85))
        close = fig.add_subplot(grid[1, i])
        close.axhspan(.69, .76, facecolor='#d5e9e4')
        close.plot([-1.37, .2], [.76, .76], color='#7faaa0', lw=2)
        close.plot([-1.37, -1.37], [.65, .76], color='#7faaa0', lw=2)
        close.set(xlim=(-1.82, -.63), ylim=(.63, 1.57), xlabel='前後方向 X (m)',
                  ylabel='高度 Z (m)' if i == 0 else '')
        close.set_aspect('equal'); close.grid(alpha=.12); close.tick_params(labelsize=8)
        reference, = close.plot([], [], '--', color='#9aa4b3', lw=1.8, alpha=.9)
        reference_handle, = close.plot([], [], '--', color='#9aa4b3', lw=1.8, alpha=.9)
        zarm, = close.plot([], [], color='#596675', lw=3, marker='o', ms=3)
        zhead, = close.plot([], [], color=color, lw=2.8)
        zhandle, = close.plot([], [], color=color, lw=3)
        zball, = close.plot([], [], 'o', color='#f28c28', ms=8)
        zn, = close.plot([], [], color='#c026a2', lw=2)
        zv, = close.plot([], [], color='#159663', lw=2)
        text = close.text(.025, .97, '', transform=close.transAxes, va='top', fontsize=10,
                          bbox=dict(facecolor='white', edgecolor='none', alpha=.82))
        artists.append(dict(top=(arm, ball, tail, head, handle, normal, velocity, status),
                            close=(zarm, zhead, zhandle, zball, zn, zv, reference, reference_handle, text)))
    times = [np.asarray([row['t'] for row in run['trace']]) for run in runs]
    duration = max(run['duration_s'] for run in runs)
    fps, source_hz = 20, 80
    moving_frames = int(np.ceil(duration * source_hz)) + 1
    frame_count = moving_frames + fps

    def update(t):
        clock.set_text(f'仿真時間 {t:.3f} s')
        ref_index = max(0, np.searchsorted(times[0], min(t, runs[0]['duration_s']), side='right') - 1)
        ref_head, ref_handle = paddle(runs[0]['trace'][ref_index])
        for run, art, ts in zip(runs, artists, times):
            sample_time = min(t, run['duration_s'])
            j = max(0, np.searchsorted(ts, sample_time, side='right') - 1)
            row = run['trace'][j]
            arm, ball, tail, head, handle, normal, velocity, status = art['top']
            p = np.asarray(row['tcp']); h, b = paddle(row)
            normal_points = np.array([p, p + .18 * np.asarray(row['normal'])])
            velocity_points = np.array([p, p + .15 * np.asarray(row['tcp_velocity'])])
            points = (row['arm'], [row['ball'][:3]], [z['ball'][:3] for z in run['trace'][:j+1]],
                      h, b, normal_points, velocity_points)
            for artist, value in zip((arm, ball, tail, head, handle, normal, velocity), points):
                artist.set_data_3d(*np.asarray(value).T)
            zarm, zh, zb, zball, zn, zv, reference, reference_handle, text = art['close']
            for artist, value in ((zarm, row['arm']), (zh, h), (zb, b), (zball, [row['ball'][:3]]),
                                  (zn, normal_points), (zv, velocity_points)):
                value = np.asarray(value); artist.set_data(value[:, 0], value[:, 2])
            if run['name'] != 'script':
                reference.set_data(ref_head[:, 0], ref_head[:, 2])
                reference_handle.set_data(ref_handle[:, 0], ref_handle[:, 2])
            ended = t >= run['duration_s'] - 1e-6
            hit = run['impact'] is not None and t >= run['impact']['time_s']
            if ended:
                contact = '有效碰拍' if run['actual_hit'] else '未有效碰拍'
                status.set_text(contact + ' · ' + outcome_names.get(run['outcome'], run['outcome']) + '（定格）')
            else:
                status.set_text('已碰拍' if hit else '等待接觸')
            speed = np.linalg.norm(row['tcp_velocity'])
            label = f'拍心速度 {speed:.2f} m/s'
            if hit:
                label += f"\n首次碰拍 {run['impact']['time_s']:.3f} s"
            if ended and run['landing_xy_m'] is not None:
                x, y = run['landing_xy_m']; label += f'\n落桌 X/Y {x:+.2f} / {y:+.2f} m'
            text.set_text(label)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '22', '-movflags', '+faststart'])
    with writer.saving(fig, str(output), dpi=100):
        for frame in range(frame_count):
            update(min(frame / source_hz, duration))
            writer.grab_frame()
    if poster is not None:
        update(min(.85, duration))
        poster.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(poster, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return dict(fps=fps, frames=frame_count, duration_s=frame_count/fps, playback_speed=.25,
                width=1600, height=900, source_time_s=duration,
                rendering='Measured PhysX positions/orientations, simplified outlines; no synthetic outcome')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=ROOT/'.agents/runs/foundation/m7/motion_comparison')
    parser.add_argument('--output',type=Path,default=ROOT/'doc/notes/images/m7_motion_comparison.png')
    parser.add_argument('--video',type=Path,help='Optional local H.264 movie; keep outside tracked assets')
    parser.add_argument('--bc-report',type=Path,help='Completed measured script/frozen-BC/closed-loop-BC comparison')
    parser.add_argument('--poster',type=Path,help='Optional local preview image for the BC video')
    args=parser.parse_args()
    if args.bc_report is not None:
        if args.video is None:parser.error('--bc-report requires --video')
        metadata_path=args.video.with_name(args.video.stem+'_visuals.json')
        outputs=[args.video,metadata_path]+([args.poster] if args.poster is not None else [])
        resolved=[path.resolve() for path in outputs]
        if args.bc_report.resolve() in resolved or len(set(resolved))!=len(resolved):
            parser.error('BC report, video, poster and visual metadata must have distinct paths')
        report=json.loads(args.bc_report.read_text())
        result=dict(status='PASS',video=str(args.video),
                    source_sha256=hashlib.sha256(args.bc_report.read_bytes()).hexdigest(),
                    renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    video_metadata=bc_video(report,args.video,args.poster))
        if args.poster is not None:result['poster']=str(args.poster)
        metadata_path.write_text(json.dumps(result,indent=2)+'\n')
        return
    if args.poster is not None:parser.error('--poster requires --bc-report')
    inputs=[args.runs/f'{name}.json' for name in ('m5','policies')]
    inputs += [args.runs.parent/f'motion_{name}.json' for name in ('m5','policies')]
    outputs=[args.output,args.runs/'visuals.json']+([args.video] if args.video is not None else [])
    if {path.resolve() for path in inputs} & {path.resolve() for path in outputs} or len({path.resolve() for path in outputs})!=len(outputs):
        parser.error('Motion reports, figure, video and visual metadata must have distinct paths')
    runs=load(args.runs);figure(runs,args.output)
    result=dict(status='PASS',figure=str(args.output),source_sha256={name:hashlib.sha256((args.runs/f'{name}.json').read_bytes()).hexdigest() for name in ('m5','policies')})
    if args.video:result.update(video=str(args.video),video_metadata=video(runs,args.video))
    (args.runs/'visuals.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
