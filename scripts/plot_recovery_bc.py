#!/usr/bin/env python3
"""Render the fixed-five recovery study from completed physical rollout records."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from plot_deepmind_ppo import OUTCOMES, ROOT, SEEDS, SERVE_IDS, ordered_episodes


def set_font():
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()


def draw(cases, fit, output):
    indexed = {(row['raw_id'], row['switch_tick']): row for row in cases}
    ticks = (6, 9, 12)
    if len(cases) != 15 or set(indexed) != {(ball, tick) for ball in SERVE_IDS for tick in ticks}:
        raise ValueError('Expected each of the five balls at each declared switch tick exactly once')
    set_font()
    colors = {'qualified': '#116a62', 'hit': '#c2e1de', 'miss': '#e8edf0'}
    ink = '#20333f'
    columns = 2 if fit else 1
    fig, axes = plt.subplots(1, columns, figsize=(7.5 * columns, 6), squeeze=False, facecolor='white')
    fig.subplots_adjust(left=.075, right=.97, top=.72, bottom=.16, wspace=.23)
    ax = axes[0, 0]
    for y, ball in enumerate(SERVE_IDS):
        for x, tick in enumerate(ticks):
            row = indexed[ball, tick]
            if row['qualifies']:
                color, status, pen = colors['qualified'], '碰拍達標', 'white'
            elif row['valid_contact']:
                color, status, pen = colors['hit'], '碰拍・偏離小', ink
            else:
                color, status, pen = colors['miss'], '未碰拍', ink
            ax.add_patch(Rectangle((x-.49, y-.48), .98, .96, color=color, lw=0))
            ax.text(x, y-.15, f"{status}  ·  {row['switch_q_rmse_rad']:.4f} rad",
                    ha='center', va='center', color=pen, fontsize=11)
            ax.text(x, y+.21, OUTCOMES[row['fault']], ha='center', va='center', color=pen, fontsize=9)
    ax.set_title('原腳本從目前狀態接管', loc='left', pad=35, fontweight='bold')
    if fit:
        runs = fit['runs']
        if tuple(row['seed'] for row in runs) != SEEDS:
            raise ValueError('The autonomous evaluation must include all three original seeds')
        policy = axes[0, 1]
        policy.set_title('修正資料 BC 自主控制', loc='left', pad=35, fontweight='bold')
        seed_labels = []
        for x, run in enumerate(runs):
            episodes = ordered_episodes(run['evaluation'])
            count = sum(row['hit'] for row in episodes)
            if count != run['contacts']:
                raise ValueError('Reported contact count differs from scored episodes')
            seed_labels.append(f"Seed {run['seed']} ({count}/5)")
            for y, row in enumerate(episodes):
                color = colors['qualified'] if row['hit'] else colors['miss']
                pen = 'white' if row['hit'] else ink
                policy.add_patch(Rectangle((x-.49, y-.48), .98, .96, color=color, lw=0))
                policy.text(x, y-.15, '有效碰拍' if row['hit'] else '未碰拍', ha='center', va='center', color=pen, fontsize=11)
                policy.text(x, y+.21, OUTCOMES[row['outcome']], ha='center', va='center', color=pen, fontsize=9)
        policy.set_xticks(range(3), labels=seed_labels)
    ax.set_xticks(range(3), labels=[f'{tick/48*1000:g} ms' for tick in ticks])
    for panel in axes.flat:
        panel.set(xlim=(-.5, 2.5), ylim=(4.5, -.5), yticks=range(5), yticklabels=SERVE_IDS)
        panel.xaxis.tick_top()
        panel.tick_params(axis='both', length=0, pad=8, labelsize=11)
        for spine in panel.spines.values():
            spine.set_visible(False)
    covered = len({row['raw_id'] for row in cases if row['qualifies']})
    contacts = sum(row['valid_contact'] for row in cases)
    legal = sum(row['legal'] for row in cases)
    title = '接管時刻與自主策略的逐球碰拍結果' if fit else '自主 BC 偏離後，由原腳本接管'
    fig.text(.045, .945, title, fontsize=19, color=ink)
    fig.text(.05, .872, f'五顆訓練球，每球三個接管時刻。'
             f' 接管後有效碰拍 {contacts}/15，合法回球 {legal}/15。', fontsize=11, color=ink)
    fig.text(.05, .092, f'接管當步的六軸關節角 RMSE ≥ 0.01 rad 且救回碰拍，'
             f'共有 {covered}/5 顆球達標（啟動修正資料 BC 需 3 顆）。', fontsize=10, color=ink)
    if fit:
        legal_bc = sum(row['outcome']=='legal_return' for run in runs for row in run['evaluation']['episodes'])
        eligible = sum(run['contact_gate_pass'] for run in runs)
        fig.text(.05, .044, f'原始 BC 三組均為 0/5。修正後自主接觸門檻為每組至少 3/5，'
                 f'共有 {eligible}/3 組達標，合法回球 {legal_bc}/15。', fontsize=10, color=ink)
    fig.savefig(output, dpi=175, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def verify_sha(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'The recorded source digest differs for {path}')


def read_completed(path):
    report = json.loads(path.read_text())
    if report['status'] != 'PASS' or report['process_exit_code'] != 0:
        raise ValueError(f'Expected a completed physical rollout report at {path}')
    return report


def draw_probe(directory, report, output):
    protocol = json.loads((directory/'protocol.json').read_text())
    if protocol['schema'] != 'airbot-recovery-probe-v1' or protocol['switch_tick'] != 12:
        raise ValueError('Expected the fixed tick-12 recovery probe protocol')
    ref_path, = [Path(path) for path in protocol['input_sha256'] if Path(path).name == 'recovery_traces.npz']
    verify_sha(ROOT/ref_path, protocol['input_sha256'][str(ref_path)])
    set_font()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(left=.13, right=.97, top=.75, bottom=.17, hspace=.24)
    colors = {6: '#007c78', 9: '#a4540a', 12: '#65529b'}
    markers = {6: 'o', 9: 's', 12: '^'}
    styles = {'autonomous': '-', 'takeover': (0, (4, 2.5))}
    fields = [
        ('measured_q_minus_nominal', '關節角\nrad'),
        ('measured_qd', '關節角速度\nrad/s'),
        ('previous_action', '前一步動作\n無量綱'),
    ]
    rows = {(r['reference_switch_tick'], r['condition']): r for r in report['comparison']}
    if len(rows) != 6 or len(report['comparison']) != 6:
        raise ValueError('Expected one comparison per reference and probe condition')
    ball = protocol['raw_id']
    switch = protocol['switch_tick']
    tick_ms = 1000 / 48
    with np.load(directory/'traces.npz') as traces, np.load(ROOT/ref_path) as refs:
        for i, tick in enumerate(colors):
            prefix = f't{tick}_{ball}_'
            for name, style in styles.items():
                row = rows[tick, name]
                indices = np.arange(row['first_tick'], row['last_tick'] + 1)
                if indices[0] != switch or len(indices) != row['samples']:
                    raise ValueError('The comparison window differs from the probe report')
                for archive, key in ((traces, f'{name}_'), (refs, prefix)):
                    np.testing.assert_array_equal(archive[key+'tick'][indices], indices)
                    if not archive[key+'prehit'][indices].all():
                        raise ValueError('Comparison includes a post-contact sample')
                for ax, (field, _) in zip(axes, fields):
                    diff = traces[f'{name}_{field}'][indices] - refs[prefix+field][indices]
                    rmse = np.sqrt(np.mean(diff.astype(float)**2, axis=1))
                    # Stagger the symbols so coincident references remain identifiable.
                    ax.plot(indices*tick_ms, rmse, color=colors[tick], linestyle=style, lw=1.55)
                    ax.plot(indices[2*i::7]*tick_ms, rmse[2*i::7], color=colors[tick], ls='none',
                            marker=markers[tick], ms=3.8, zorder=3)

    ink = '#213846'
    for ax, (_, label) in zip(axes, fields):
        ax.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=12, color=ink)
        ax.axvline(switch*tick_ms, color='#627581', linestyle=':', lw=1, zorder=0)
        ax.grid(axis='y', color='#dbe3e7', lw=.7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#aab8bf')
        ax.tick_params(colors=ink)
    axes[-1].set_xticks(np.arange(switch, max(r['last_tick'] for r in rows.values()) + 1, 6)*tick_ms)
    axes[-1].set_xlabel('自發球起算的決策時間（ms）')
    legend = [Line2D([], [], color=colors[t], marker=markers[t], ms=5,
                     label=f'舊軌跡 {t*tick_ms:g} ms 接管') for t in colors]
    legend.extend(Line2D([], [], color=ink, linestyle=style, lw=1.8, label=label)
                  for style, label in ((styles['autonomous'], '新策略自主'), (styles['takeover'], '新策略轉腳本')))
    fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.04, .847), frameon=False,
               ncol=5, fontsize=9, columnspacing=1.5)
    episodes = {r['condition']: r['episode'] for r in report['cases']}
    outcomes = []
    for name, label in (('autonomous', '自主'), ('takeover', '250 ms 轉腳本')):
        episode = episodes[name]
        contact = f"{1000*episode['motion']['impact']['time_s']:.0f} ms 碰拍，" if episode['hit'] else ''
        outcomes.append(f"{label}：{contact}{OUTCOMES[episode['outcome']]}")
    fig.text(.045, .948, f'球 {ball} · 修正 BC {protocol["seed"]} 與三條舊接管軌跡的差距',
             fontsize=19, color=ink)
    fig.text(.046, .887, '  ｜  '.join(outcomes), fontsize=12, color=ink)
    fig.text(.046, .078, '每點為同一決策時刻兩條軌跡的六軸 RMSE。虛直線為本次接管時刻，各組畫到雙方最後一個共同碰拍前樣本。',
             fontsize=9.2, color=ink)
    fig.text(.046, .043, '接管後皆依同一腳本給動作，下一步起三組前動作差距重疊。', fontsize=9.2, color=ink)
    fig.savefig(output, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def remaining_cases(directory, report):
    protocol = json.loads((directory/'protocol.json').read_text())
    if protocol['schema'] != 'airbot-recovery-remaining-v1' or protocol['switch_tick'] != 12:
        raise ValueError('Expected the fixed four-case tick-12 protocol')
    if len(report['runs']) != 4 or any(
            (run['seed'], run['raw_id']) != (spec['seed'], spec['raw_id'])
            for run, spec in zip(report['runs'], protocol['cases'], strict=True)):
        raise ValueError('Cases must match the four predeclared seed and ball pairs')
    verify_sha(directory/'candidates.pt', report['candidates_sha256'])
    inputs = protocol['input_sha256']
    reuse = report['reused_9378']
    old_path = Path(protocol['prior_probe'])
    if reuse['replayed_this_run'] or reuse['report'] != str(old_path) or reuse['report_sha256'] != inputs[str(old_path)]:
        raise ValueError('Expected the linked 9378 probe receipt')
    verify_sha(old_path, reuse['report_sha256'])
    old = read_completed(old_path)
    old_protocol_path, old_traces = (old_path.with_name(n) for n in ('protocol.json', 'traces.npz'))
    verify_sha(old_protocol_path, old['protocol_sha256'])
    verify_sha(old_traces, old['traces_sha256'])
    old_protocol = json.loads(old_protocol_path.read_text())
    old_id = old_protocol['raw_id']
    if (old_protocol['seed'], old_id, old_protocol['switch_tick']) != (7101, 9378, protocol['switch_tick']):
        raise ValueError('Linked probe has an unexpected seed, ball or takeover tick')
    replay_path, = [Path(k) for k in inputs if Path(k).name == 'replay_comparison.json']
    verify_sha(ROOT/replay_path, inputs[str(replay_path)])
    script = read_completed(ROOT/replay_path)['bank_evaluations']['script']
    script_impact_s = {e['raw_id']: e['motion']['impact']['time_s'] for e in script['episodes']}
    episodes = {r['condition']: r['episode'] for r in old['cases']}
    with np.load(old_traces) as data:
        switch_s = float(data['takeover_elapsed_s'][protocol['switch_tick']])
    impact = episodes['takeover']['motion']['impact']
    historical = dict(seed=old_protocol['seed'], raw_id=old_id, episodes=episodes, comparison=old['comparison'],
                      reference_remaining_s=script_impact_s[old_id]-protocol['switch_tick']/48,
                      actual_takeover_delay_s=impact['time_s']-switch_s if impact else None)
    return [historical, *report['runs']]


def draw_remaining(directory, report, output):
    runs = remaining_cases(directory, report)
    set_font()
    ink = '#20333f'
    fig, (table, errors) = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True,
                                       gridspec_kw={'width_ratios': [1.52, 1]})
    fig.subplots_adjust(left=.035, right=.97, bottom=.24, top=.75, wspace=.07)
    table.set(xlim=(0, 1), ylim=(len(runs)-.5, -.5))
    table.axis('off')
    xs = (.015, .31, .53, .815)
    for x, label in zip(xs, ('球 ID／seed', '自主控制', '原腳本接管', '碰拍餘時（ms）')):
        table.text(x, 1.06, label, transform=table.transAxes, ha='left' if x==xs[0] else 'center',
                   color=ink, fontsize=11)
    errors.set_title('250 ms 當步的策略與腳本請求差距', loc='left', fontsize=12, color=ink, pad=13)
    styles = [('即時觀測', '#1b3946', 'D'), ('舊 125 ms', '#007c78', 'o'),
              ('舊 187.5 ms', '#a4540a', 's'), ('舊 250 ms', '#65529b', '^')]
    for y, run in enumerate(runs):
        # The two continuations share this pre-action state at tick 12.
        rows = {r['reference_switch_tick']: r['switch'] for r in run['comparison'] if r['condition']=='autonomous'}
        points = [rows[6]['new_actor_request_vs_script_rmse'],
                  *(rows[t]['reference_new_actor_vs_label_rmse'] for t in (6, 9, 12))]
        for i, (value, (_, color, marker)) in enumerate(zip(points, styles)):
            errors.plot(value, y+(i-1.5)*.105, marker=marker, color=color, ms=6, ls='none')
        for panel in (table, errors):
            panel.axhline(y+.5, color='#dbe3e7', lw=.7, zorder=0)
        reused = ' · 沿用' if y == 0 else ''
        table.text(xs[0], y-.13, str(run['raw_id']), fontsize=12, va='center', color=ink)
        table.text(xs[0], y+.19, f"{run['seed']}{reused}", fontsize=9, va='center', color='#526771')
        for x, name in zip(xs[1:3], ('autonomous', 'takeover')):
            episode = run['episodes'][name]
            table.text(x, y-.14, '有效碰拍' if episode['hit'] else '未碰拍', va='center', ha='center',
                       fontsize=11, color='#00745f' if episode['hit'] else ink)
            table.text(x, y+.19, OUTCOMES[episode['outcome']], va='center', ha='center', color=ink, fontsize=9)
        delay = run['actual_takeover_delay_s']
        measured = '—' if delay is None else f'{1000*delay:.0f}'
        table.text(xs[3], y, f"{1000*run['reference_remaining_s']:.0f} ／ {measured}",
                   va='center', ha='center', color=ink, fontsize=11)
    errors.set_xscale('log')
    marks = [.005, .01, .02, .05, .1, .2, .5]
    errors.set_xticks(marks, labels=[f'{m:g}' for m in marks])
    errors.set_xlabel('六軸動作 RMSE（無量綱，對數軸）', color=ink)
    errors.grid(axis='x', color='#dbe3e7', lw=.7)
    errors.set_axisbelow(True)
    errors.tick_params(axis='y', left=False, labelleft=False)
    errors.spines[['left', 'right', 'top']].set_visible(False)
    handles = [Line2D([], [], color=c, marker=m, ls='none', label=s) for s, c, m in styles]
    errors.legend(handles=handles, loc='upper left', bbox_to_anchor=(-.035, -.22), ncol=4,
                  fontsize=8.8, frameon=False, handletextpad=.25, columnspacing=1)
    new_hits = report['takeover_contacts']
    fig.text(.04, .94, '修正 BC 五個自主漏接案例在 250 ms 接管後的結果', fontsize=20, color=ink)
    fig.text(.04, .862, f"本次新增四案、八回合，腳本接管救回 {new_hits}/4 球，合法回球 {report['legal_returns']}。"
             '首列沿用先前的 9378 配對。', fontsize=11.5, color=ink)
    fig.text(.04, .095, '碰拍餘時依序為原腳本全程運行的碰拍時間減 250 ms，以及接管後到實際碰拍的時間。', fontsize=10, color=ink)
    fig.text(.04, .042, '右圖比較同一策略在即時觀測與三條舊接管軌跡觀測上的請求。四點均取 250 ms，圖例中的舊時刻標示原軌跡的接管時間。', fontsize=9.7, color=ink)
    fig.savefig(output, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def hit_legal(episodes):
    return sum(e['hit'] for e in episodes), sum(e['outcome'] == 'legal_return' for e in episodes)


def refinement_data(ppo):
    c2 = Path(ppo['runs'][0]['actor_initialization']['directory'])
    manifest_path = c2/'initialization.json'
    manifest = json.loads(manifest_path.read_text())
    verify_sha(c2/'protocol.json', manifest['protocol_sha256'])
    protocol = json.loads((c2/'protocol.json').read_text())
    if manifest['schema'] != 'airbot-contact-refinement-v1' or protocol['recipe']['candidate'] != 2:
        raise ValueError('Expected PPO initialized by the second BC refinement candidate')
    base = c2.parent.parent
    paths = (base/'bc_run.json', base/'recovery_bc/fit/results.json',
             c2.parent/'candidate_1/results.json', c2/'results.json')
    stages = []
    for path in paths:
        expected = manifest['evaluation_sha256'] if path == paths[-1] else protocol['input_sha256'][str(path)]
        verify_sha(path, expected)
        bc = read_completed(path)
        if tuple(r['seed'] for r in bc['runs']) != SEEDS:
            raise ValueError(f'Expected all three paired seeds in {path}')
        evaluations = [ordered_episodes(r['evaluation']) for r in bc['runs']]
        counts = [hit_legal(e)[0] for e in evaluations]
        if counts != [r['contacts'] for r in bc['runs']] or (min(counts) >= 3) != (bc['contact_gate'] == 'PASS'):
            raise ValueError(f'Contact gate differs from physical episodes in {path}')
        stages.append(evaluations)
    if tuple(r['training_seed'] for r in ppo['runs']) != SEEDS:
        raise ValueError('Expected three final PPO policies')
    finals = []
    for r, initial in zip(ppo['runs'], stages[-1]):
        verify_sha(manifest_path, r['actor_initialization']['initialization_sha256'])
        if r['status'] != 'PASS' or r['group'] != 'A' or r['transitions'] != 6144:
            raise ValueError('Expected each final A-group 6144-transition PPO evaluation')
        final = ordered_episodes(r['evaluation'])
        for current, replay in ((initial, r['initial_evaluation']), (final, r['replay_evaluation'])):
            if any((a['hit'], a['outcome']) != (b['hit'], b['outcome'])
                   for a, b in zip(current, ordered_episodes(replay))):
                raise ValueError('PPO initialization or checkpoint replay differs on a scored ball')
        finals.append(final)
    stages.append(finals)
    noise = read_completed(c2/'noise.json')
    verify_sha(manifest_path, noise['initialization_sha256'])
    counts = {}
    for r in noise['runs']:
        trials = r['trials']
        if [t['repetition'] for t in trials] != list(range(10)) or any(t['noise_std'] != .3 for t in trials):
            raise ValueError('Expected ten paired noise repetitions per policy')
        counts[r['seed'], r['kind']] = hit_legal([e for t in trials for e in ordered_episodes(t)])
    if noise['total'] != 300 or set(counts) != {(s, k) for s in SEEDS for k in ('original', 'refined_bc')}:
        raise ValueError('Paired noise report is incomplete')
    return stages, counts


def draw_refinement(ppo, output):
    stages, noise = refinement_data(ppo)
    set_font()
    ink = '#20333f'
    colors = ['#e9eef1', '#f5d5a0', '#78c8ae']
    labels = ['原始 BC', '舊修正 BC', '候選 1（新增資料）', '候選 2（餘弦衰減）', '候選 2 ＋ PPO']
    fig, panels = plt.subplots(1, 3, figsize=(15.2, 6.1), sharey=True)
    fig.subplots_adjust(left=.13, right=.985, top=.71, bottom=.30, wspace=.075)
    for j, (ax, seed) in enumerate(zip(panels, SEEDS)):
        for i, stage in enumerate(stages):
            episodes = stage[j]
            for col, e in enumerate(episodes):
                level = 2 if e['outcome'] == 'legal_return' else int(e['hit'])
                ax.add_patch(Rectangle((col-.49, i-.48), .98, .96, color=colors[level], lw=0))
                outcome = '保護終止' if e['outcome'] == 'arm_fault' else OUTCOMES[e['outcome']]
                ax.text(col, i, outcome, ha='center', va='center', fontsize=8.5, color=ink)
            hits, legal = hit_legal(episodes)
            ax.text(5, i, f'{hits} / {legal}', ha='center', va='center', fontsize=11, weight='bold', color=ink)
        ax.set(xlim=(-.5, 5.5), ylim=(4.5, -.5), xticks=range(6), xticklabels=[*SERVE_IDS, '碰／合'],
               yticks=range(5), yticklabels=labels)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, labelsize=9, pad=5, colors=ink)
        ax.set_title(f'Seed {seed}', fontsize=13, loc='left', pad=12, color=ink)
        for spine in ax.spines.values():
            spine.set_visible(False)
        before, after = (noise[seed, k] for k in ('original', 'refined_bc'))
        ax.text(.01, -.21, f'原始初值 {before[0]} / {before[1]}    →    候選 2   {after[0]} / {after[1]}',
                transform=ax.transAxes, fontsize=10.5, color=ink)
    fig.text(.035, .938, '從示範模仿到短程 PPO 的五球閉環結果', fontsize=20, color=ink)
    final_hits = sum(hit_legal(e)[0] for e in stages[-1])
    final_legal = sum(hit_legal(e)[1] for e in stages[-1])
    fig.text(.036, .858, f'候選 2 通過三組各至少 3/5 有效碰拍的 BC 門檻。每組再經 6,144 次 PPO 互動後，'
             f'最終共 {final_hits}/15 有效碰拍、{final_legal}/15 合法回球。', fontsize=12, color=ink)
    fig.text(.035, .213, '配對噪聲', fontsize=11.5, color=ink)
    legend = [Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in zip(colors, ('未有效碰拍', '有效碰拍後失分', '合法回球'))]
    fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(.029, .085), frameon=False, ncol=3,
               fontsize=10, handlelength=1.2, handletextpad=.4, columnspacing=1.3)
    fig.text(.4, .105, '每列為固定最後模型。各組的「碰／合」欄記有效碰拍數與合法回球數，分母均為 5。', fontsize=10, color=ink)
    fig.text(.036, .048, '配對動作噪聲診斷（標準差 0.3）也列有效碰拍數／合法回球數，每個 seed、每種初值各 50 回合。'
             '全圖使用相同五顆訓練球。', fontsize=10, color=ink)
    fig.savefig(output, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    view = parser.add_mutually_exclusive_group()
    view.add_argument('--probe', action='store_true', help='Draw the two-episode comparison against old recovery traces')
    view.add_argument('--remaining', action='store_true', help='Draw four additional cases alongside the linked first probe')
    view.add_argument('--refinement', action='store_true', help='Draw the refinement study from its completed PPO report')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    image_name = 'deepmind_five_recovery.png'
    if args.probe:
        image_name = 'deepmind_recovery_probe.png'
    if args.remaining:
        image_name = 'deepmind_remaining_recovery.png'
    if args.refinement:
        image_name = 'deepmind_bc_refinement.png'
    output = args.output or ROOT/'doc/notes/images'/image_name
    if output.resolve().suffix.lower() != '.png':
        parser.error('Expected a PNG output path')
    report = read_completed(args.report)
    if args.refinement:
        draw_refinement(report, output)
        return
    trace_name = 'traces.npz' if args.probe or args.remaining else 'recovery_traces.npz'
    verify_sha(args.report.parent/'protocol.json', report['protocol_sha256'])
    verify_sha(args.report.parent/trace_name, report['traces_sha256'])
    if args.remaining:
        draw_remaining(args.report.parent, report, output)
        return
    if args.probe:
        draw_probe(args.report.parent, report, output)
        return
    fit = None
    if report['coverage_gate'] == 'PASS':
        fit_path = args.report.parent / 'fit/results.json'
        fit = json.loads(fit_path.read_text())
        if fit['status'] != 'PASS' or fit['process_exit_code'] != 0:
            raise ValueError('A completed autonomous evaluation with successful exit is required')
    draw(report['cases'], fit, output)


if __name__ == '__main__':
    main()
