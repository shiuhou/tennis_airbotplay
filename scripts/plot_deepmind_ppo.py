#!/usr/bin/env python3
"""Draw the fixed-five DeepMind A study from JSON; no model loading or training."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SERVE_IDS = (9378, 10818, 5255, 12250, 14852)
SEEDS = (7101, 7102, 7103)
OUTCOMES = {
    'unexpected_contact': '非預期接觸', 'arm_fault': '手臂故障',
    'net': '碰網終止', 'illegal_hit': '非法擊球',
    'wrong_bounce': '反彈違規', 'legal_return': '合法回球',
    'miss': '漏接', 'out': '出界', 'timeout': '超時',
}


def ordered_episodes(evaluation):
    rows = evaluation['episodes']
    if len(rows) != len(SERVE_IDS):
        raise ValueError('Each evaluation must contain exactly the five scored serves')
    indexed = {row['raw_id']: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(SERVE_IDS):
        raise ValueError('Evaluation IDs differ from the fixed training-side bank')
    if any(row['outcome'] not in OUTCOMES for row in rows):
        raise ValueError('Evaluation contains an unknown or incomplete outcome')
    return [indexed[raw_id] for raw_id in SERVE_IDS]


def draw(report_path, output):
    report = json.loads(report_path.read_text())
    if report['status'] != 'PASS':
        raise ValueError('Training report is incomplete; do not plot it as a completed study')
    runs = sorted(report['runs'], key=lambda row: row['training_seed'])
    if tuple(row['training_seed'] for row in runs) != SEEDS:
        raise ValueError('Expected exactly the three predeclared training seeds')
    if any(row['group'] != 'A' or row['transitions'] != 6144 for row in runs):
        raise ValueError('Report differs from the A-only, 6144-transitions protocol')
    rows = [ordered_episodes(report['zero_action_baseline'])]
    labels = ['零動作']
    for run in runs:
        rows.extend([ordered_episodes(run['initial_evaluation']), ordered_episodes(run['evaluation'])])
        labels.extend([f"{run['training_seed']}：未訓練", f"{run['training_seed']}：訓練後"])
    levels = np.array([[2 if row['outcome'] == 'legal_return' else int(row['hit'])
                        for row in episodes] for episodes in rows])
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False})
    fig = plt.figure(figsize=(14, 8.5), facecolor='#f8fafc')
    grid = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], wspace=.38, hspace=.45)
    matrix = fig.add_subplot(grid[:, 0])
    colors = ['#e2e8f0', '#e8b76b', '#389475']
    matrix.imshow(levels, cmap=ListedColormap(colors), norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3), aspect='auto')
    for i, episodes in enumerate(rows):
        for j, row in enumerate(episodes):
            matrix.text(j, i, OUTCOMES[row['outcome']], ha='center', va='center',
                        fontsize=10, color='white' if levels[i, j] == 2 else '#243746')
    summaries = [f"{label}  [{sum(r['outcome']=='legal_return' for r in episodes)}/5]"
                 for label, episodes in zip(labels, rows)]
    matrix.set(xticks=range(5), xticklabels=SERVE_IDS, yticks=range(7), yticklabels=summaries,
               xlabel='原始來球 ID（五顆均屬 train）', title='確定性評測：每格一球；列末為合法數／5')
    matrix.set_xticks(np.arange(-.5, 5, 1), minor=True)
    matrix.set_yticks(np.arange(-.5, 7, 1), minor=True)
    matrix.grid(which='minor', color='white', linewidth=2)
    matrix.tick_params(which='minor', bottom=False, left=False)
    for y in (.5, 2.5, 4.5):
        matrix.axhline(y, color='#94a3b8', lw=1)
    matrix.legend(handles=[Patch(color=color, label=label) for color, label in zip(colors,
                  ['未有效碰拍', '碰拍但回球失敗', '合法回球'])],
                  loc='upper center', bbox_to_anchor=(.42, .145), bbox_transform=fig.transFigure,
                  ncol=3, fontsize=9, frameon=False)
    seed_colors = ['#287d9b', '#b67536', '#8868ad']
    for axis, metric, title in [(fig.add_subplot(grid[0, 1]), 'training_hit_rate', '訓練累積有效碰拍比例'),
                                (fig.add_subplot(grid[1, 1]), 'training_legal_rate', '訓練累積合法回球比例')]:
        for run, color in zip(runs, seed_colors):
            history = run['history']
            x = [entry['transitions'] for entry in history]
            y = [np.nan if entry.get(metric) is None else entry[metric] * 100 for entry in history]
            axis.plot(x, y, color=color, lw=2, label=str(run['training_seed']))
        axis.set(xlim=(0, 6144), ylim=(-3, 103), xlabel='環境 transitions', ylabel='已結束訓練 episodes (%)', title=title)
        axis.set_xticks([0, 2048, 4096, 6144])
        axis.grid(alpha=.2)
        axis.legend(title='training seed', fontsize=9, title_fontsize=9, frameon=False)
    fig.suptitle('五顆真實初態來球：Actor 能否學會完整回球？',
                 x=.04, ha='left', y=.98, fontsize=19, weight='bold')
    fig.text(.04, .928, 'A 組基本事件獎勵｜原安裝／ready／±0.6 rad｜3 seeds × 6,144 transitions｜固定最後 checkpoint',
             fontsize=11, color='#475569')
    fig.text(.04, .035, '左圖：確定性均值動作；零動作 5 球＋各 seed 初始化／訓練後各 5 球，共 35 個主要評測 episode。\n'
             '右圖：隨機探索期間的累積訓練比例，分母依已結束回合而變；不能與左圖當成同一評測。\n'
             '五顆球同時用於訓練與檢查，沒有新 held-out 或泛化聲明；原始旋轉保留在資料中，模擬採零旋近似。',
             fontsize=9, color='#475569')
    fig.subplots_adjust(left=.19, right=.97, top=.855, bottom=.215)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_bc(directory, output, ppo_directory=None):
    """Plot the three paired BC seeds, preserving a failed gate as NOT_RUN PPO."""
    result = json.loads((directory / 'results.json').read_text())
    baseline = json.loads((directory / 'baseline_replay.json').read_text())
    noise = json.loads((directory / 'noise.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    if result['status'] != 'PASS' or baseline['status'] != 'PASS':
        raise ValueError('BC collection and baseline replay must be completed before plotting')
    runs = [json.loads((directory / f'bc_{seed}.json').read_text()) for seed in SEEDS]
    originals = {row['seed']: row for row in baseline['runs']}
    if set(originals) != set(SEEDS) or tuple(row['seed'] for row in runs) != SEEDS:
        raise ValueError('BC figure requires the three original paired seeds')
    if protocol['recipe']['minimum_contacts_per_seed'] != 3:
        raise ValueError('Unexpected BC contact gate')
    gate_pass = all(row['contacts'] >= 3 for row in runs)
    if gate_pass != (result['contact_gate'] == 'PASS'):
        raise ValueError('BC contact counts disagree with the recorded gate')
    if ppo_directory is not None and not gate_pass:
        raise ValueError('Do not plot PPO as executed after a failed BC contact gate')
    ppo_runs = {}
    if ppo_directory is not None:
        initialization_sha = hashlib.sha256((directory / 'initialization.json').read_bytes()).hexdigest()
        for seed in SEEDS:
            row = json.loads((ppo_directory / f'A_{seed}' / 'result.json').read_text())
            if row['status'] != 'PASS' or row['training_seed'] != seed or row['group'] != 'A' or row['transitions'] != 6144:
                raise ValueError('BC PPO result differs from the fixed paired protocol')
            if row.get('actor_initialization', {}).get('seed') != seed:
                raise ValueError('PPO result is missing the paired BC initialization')
            if row['actor_initialization'].get('initialization_sha256') != initialization_sha:
                raise ValueError('PPO result belongs to a different BC initialization experiment')
            ppo_runs[seed] = row

    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False})
    fig = plt.figure(figsize=(17, 10.5), facecolor='#f8fafc')
    grid = fig.add_gridspec(3, 2, width_ratios=[1.55, 1], wspace=.35, hspace=.72)
    matrix = fig.add_subplot(grid[:, 0])
    rows, labels = [], []
    for seed, run in zip(SEEDS, runs):
        for name, evaluation in (
                ('原 PPO', originals[seed]['final']),
                ('BC', run['evaluation']),
                ('BC＋PPO', ppo_runs[seed]['evaluation'] if seed in ppo_runs else None)):
            episodes = ordered_episodes(evaluation) if evaluation is not None else None
            rows.append(episodes)
            counts = f"{sum(r['hit'] for r in episodes)}／{sum(r['outcome']=='legal_return' for r in episodes)}" if episodes else '—／—'
            labels.append(f'{seed}  {name}  [{counts}]')
    levels = np.array([[-1] * 5 if episodes is None else
                       [2 if row['outcome'] == 'legal_return' else int(row['hit']) for row in episodes]
                       for episodes in rows])
    colors = ['#ffffff', '#e2e8f0', '#e8b76b', '#389475']
    matrix.imshow(levels, cmap=ListedColormap(colors), norm=BoundaryNorm([-1.5, -.5, .5, 1.5, 2.5], 4), aspect='auto')
    for i, episodes in enumerate(rows):
        if episodes is None:
            label = 'NOT_RUN：BC 門檻未通過' if not gate_pass else 'NOT_RUN：未提供 PPO 結果'
            matrix.text(2, i, label, ha='center', va='center', fontsize=10, color='#64748b',
                        bbox=dict(facecolor='white', edgecolor='none', pad=4))
            continue
        for j, row in enumerate(episodes):
            # This protective simulation event is not a claim of hardware failure.
            outcome = '保護性終止' if row['outcome'] == 'arm_fault' else OUTCOMES[row['outcome']]
            matrix.text(j, i, outcome, ha='center', va='center', fontsize=9,
                        color='white' if levels[i, j] == 2 else '#243746')
    matrix.set(xticks=range(5), xticklabels=SERVE_IDS, yticks=range(9), yticklabels=labels,
               xlabel='原始來球 ID（同一組五顆 train 球）', title='閉環均值動作：列末 [有效接觸數／合法回球數]，各以 5 球計')
    matrix.set_xticks(np.arange(-.5, 5, 1), minor=True)
    matrix.set_yticks(np.arange(-.5, 9, 1), minor=True)
    matrix.grid(which='minor', color='white', linewidth=2)
    matrix.tick_params(which='minor', bottom=False, left=False)
    for y in (2.5, 5.5):
        matrix.axhline(y, color='#94a3b8', lw=1.5)

    seed_colors = ['#287d9b', '#b67536', '#8868ad']
    loss_axis = fig.add_subplot(grid[0, 1])
    for row, color in zip(runs, seed_colors):
        history = row['fitting']['history']
        loss_axis.plot([entry['update'] for entry in history],
                       [entry['loss'] for entry in history], color=color, lw=2, label=str(row['seed']))
    loss_axis.set(yscale='log', xlabel='BC optimizer 更新次數', ylabel='五條示範等權 MSE', title='示範動作擬合：固定 2,000 次更新')
    loss_axis.grid(alpha=.2)
    loss_axis.legend(ncol=3, fontsize=8, frameon=False)

    fit_axis = fig.add_subplot(grid[1, 1])
    x = np.arange(5)
    for offset, row, color in zip((-.24, 0, .24), runs, seed_colors):
        errors = [row['fitting']['per_trajectory_rmse'][str(i)] for i in range(5)]
        fit_axis.bar(x + offset, errors, width=.23, color=color, label=str(row['seed']))
    fit_axis.set(xticks=x, xticklabels=SERVE_IDS, ylabel='動作 RMSE（無單位）', title='最後 Actor 對示範的動作誤差；不是閉環成功率')
    fit_axis.tick_params(axis='x', labelsize=8)
    fit_axis.grid(axis='y', alpha=.2)

    noise_axis = fig.add_subplot(grid[2, 1])
    has_rollout_diagnostic = (directory / 'rollout_diagnostic.json').exists()
    if noise['status'] == 'PASS':
        repetitions = noise['repetitions_per_ball']
        indexed = {(row['seed'], row['kind']): row for row in noise['runs']}
        if set(indexed) != {(seed, kind) for seed in SEEDS for kind in ('original', 'bc')}:
            raise ValueError('Noise figure requires all paired original and BC policies')
        counts, noise_labels = [], []
        for seed in SEEDS:
            for kind, label in (('original', '原始'), ('bc', 'BC')):
                trials = indexed[seed, kind]['trials']
                if len(trials) != repetitions:
                    raise ValueError('Incomplete action-noise repetitions')
                counts.append(np.array([[r['hit'] for r in ordered_episodes(t)] for t in trials]).sum(axis=0))
                noise_labels.append(f'{seed} {label}')
        counts = np.asarray(counts)
        noise_axis.imshow(counts, cmap='Blues', vmin=0, vmax=repetitions, aspect='auto')
        for i in range(6):
            for j in range(5):
                noise_axis.text(j, i, str(int(counts[i, j])), ha='center', va='center', fontsize=8,
                                color='white' if counts[i, j] > repetitions / 2 else '#243746')
        noise_axis.set(xticks=range(5), xticklabels=SERVE_IDS, yticks=range(6), yticklabels=noise_labels,
                       title=f"探索 std={protocol['recipe']['noise_std']}：每格有效接觸數／{repetitions}")
        noise_axis.tick_params(labelsize=8)
    elif has_rollout_diagnostic:
        diagnostic = json.loads((directory / 'rollout_diagnostic.json').read_text())
        if diagnostic['status'] != 'PASS' or diagnostic['ppo_updates'] != 0 or diagnostic['bc_updates'] != 0:
            raise ValueError('Expected a completed replay diagnostic without training updates')
        selected_id = SERVE_IDS[0]
        reference_times = []
        with np.load(directory / 'rollout_diagnostic.npz', allow_pickle=False) as traces:
            for seed, color in zip(SEEDS, seed_colors):
                matches = [row for row in diagnostic['rows'] if row['seed'] == seed and row['raw_id'] == selected_id]
                if len(matches) != 1:
                    raise ValueError('Missing or duplicate paired rollout diagnostic')
                reference_time = matches[0]['reference_contact_time_s']
                reference_times.append(reference_time)
                trace = traces[f'{seed}_{selected_id}']
                before_contact = trace[trace[:, 0] < reference_time]
                if len(before_contact) != matches[0]['compared_precontact_steps'] or not np.isfinite(before_contact[:, :5]).all():
                    raise ValueError('Rollout trace disagrees with the pre-contact report')
                noise_axis.plot(before_contact[:, 0], before_contact[:, 3], color=color, lw=1.7, label=str(seed))
                noise_axis.plot(before_contact[:, 0], before_contact[:, 4], color=color, lw=1.5, ls='--')
        if not np.allclose(reference_times, reference_times[0], atol=1e-6, rtol=0):
            raise ValueError('The paired seed traces must share the same script contact time')
        noise_axis.axvline(reference_times[0], color='#94a3b8', lw=1, ls=':')
        noise_axis.set(yscale='log', xlim=(0, reference_times[0] + .01),
                       xlabel='時間 (s)；實線：自身觀測／虛線：示範觀測', ylabel='請求動作 RMSE',
                       title=f'首個 ID {selected_id}：原腳本碰球前的動作偏離')
        noise_axis.grid(alpha=.2)
        noise_axis.legend(ncol=3, fontsize=8, frameon=False)
    else:
        collection = json.loads((directory / 'collection.json').read_text())
        script_rows = ordered_episodes(collection['evaluation'])
        noise_axis.plot(x, [row['min_ball_distance_m'] * 100 for row in script_rows],
                        color='#334155', ls='--', marker='s', lw=1.5, label='原腳本')
        for row, color in zip(runs, seed_colors):
            episodes = ordered_episodes(row['evaluation'])
            noise_axis.plot(x, [episode['min_ball_distance_m'] * 100 for episode in episodes],
                            color=color, marker='o', lw=1.5, label=str(row['seed']))
        noise_axis.set(xticks=x, xticklabels=SERVE_IDS, ylabel='距離 (cm)', title='實際閉環：球心與拍心的最近距離')
        noise_axis.tick_params(axis='x', labelsize=8)
        noise_axis.grid(alpha=.2)
        noise_axis.legend(ncol=4, fontsize=7.5, frameon=False)

    contact_counts = '、'.join(f"{row['seed']}：{row['contacts']}/5" for row in runs)
    fig.suptitle('先學會碰球，再學回球：接觸模仿初始化對照', x=.04, ha='left', y=.98, fontsize=20, weight='bold')
    fig.text(.04, .933, f"BC 啟動門檻：每個 seed 至少 3/5 接觸｜結果 {result['contact_gate']}｜{contact_counts}",
             fontsize=11, color='#475569')
    fig.text(.04, .904, '原安裝／ready／六維 ±0.6 rad／A 獎勵不變；僅 Actor 先做 BC，Critic 與初始 std 保留原值。',
             fontsize=10, color='#475569')
    fig.legend(handles=[Patch(color=color, label=label) for color, label in zip(colors[1:],
               ['未有效碰拍', '碰拍但回球失敗', '合法回球'])], loc='center', bbox_to_anchor=(.45, .142),
               ncol=3, fontsize=10, frameon=False)
    noise_note = ('噪聲診斷僅作量測，不否決已通過接觸門檻的 PPO。' if noise['status'] == 'PASS' else
                  '未通過 BC 門檻：探索噪聲診斷與後續 PPO 均 NOT_RUN。')
    if noise['status'] != 'PASS':
        noise_note += ('右下固定展示首個 ID 9378；全 15 例見筆記，不據此宣稱唯一根因。' if has_rollout_diagnostic else
                       '右下最近距離不是拍面接觸判定。')
    fig.text(.04, .035, '示範只有接觸成功：4 次反彈違規、1 次出界；BC 動作誤差低不等於能在閉環中碰球。\n'
             'BC 使用額外示範與擬合成本；條件式 PPO 沿用每 seed 6,144 transitions，不宣稱總成本相同。\n'
             '全部是同五顆已研究的 train 初態；噪聲重複也不是新來球或泛化證據。重載重播不作獨立樣本。\n' + noise_note,
             fontsize=9, color='#475569')
    fig.subplots_adjust(left=.20, right=.97, top=.845, bottom=.21)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_feedback(report_path, output):
    """Plot measured early errors and contact outcomes; no policy/model loading."""
    trace_path = report_path.with_name('feedback_traces.npz')
    protocol_path = report_path.with_name('feedback_protocol.json')
    if output.resolve() in {path.resolve() for path in (report_path, trace_path, protocol_path)}:
        raise ValueError('Figure output must not overwrite diagnostic evidence')
    report = json.loads(report_path.read_text())
    if report['status'] != 'PASS' or report.get('process_exit_code') != 0:
        raise ValueError('Feedback audit must be complete with confirmed process exit 0')
    if report['seed'] != 7101 or report['episodes'] != 20 or report['bc_updates'] or report['ppo_updates']:
        raise ValueError('Feedback figure requires the fixed seed7101 no-training experiment')
    if hashlib.sha256(trace_path.read_bytes()).hexdigest() != report['traces_sha256']:
        raise ValueError('Feedback traces differ from the completed report')
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != report['protocol_sha256']:
        raise ValueError('Feedback protocol differs from the completed report')
    names = ('baseline', 'previous_action', 'velocity', 'both')
    labels = ('自主基線', '示範前動作', '示範速度', '兩者皆替換')
    colors = ('#485568', '#267eaa', '#d18b25', '#319178')
    runs = report['conditions']
    if tuple(row['name'] for row in runs) != names:
        raise ValueError('Expected exactly the four predeclared intervention conditions')
    episodes = [ordered_episodes(run['evaluation']) for run in runs]
    expected_times = np.arange(10) / 48
    curves = {}
    with np.load(trace_path, allow_pickle=False) as traces:
        for name in names:
            for metric in ('action_rmse', 'q_rmse_rad', 'qd_rmse_rad_s'):
                series = []
                for raw_id in SERVE_IDS:
                    prefix = f'{name}_{raw_id}_'
                    times = traces[prefix + 'time_s']
                    selected = (times >= 0) & (times < .2)
                    np.testing.assert_allclose(times[selected], expected_times, atol=1e-10, rtol=0)
                    values = traces[prefix + metric][selected]
                    if not np.isfinite(values).all() or (values < 0).any():
                        raise ValueError('Early feedback errors must be complete, finite and nonnegative')
                    series.append(values)
                curves[name, metric] = np.stack(series)

    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False})
    fig = plt.figure(figsize=(15, 9), facecolor='#f8fafc')
    grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=.55, wspace=.32)
    for j, (metric, title, unit) in enumerate((
            ('action_rmse', '動作偏差', '六關節 action RMSE'),
            ('q_rmse_rad', '實測位置偏差', '六關節 RMSE（rad）'),
            ('qd_rmse_rad_s', '實測速度偏差', '六關節 RMSE（rad/s）'))):
        axis = fig.add_subplot(grid[0, j])
        for name, label, color in zip(names, labels, colors):
            values = curves[name, metric]
            axis.fill_between(expected_times * 1000, values.min(0), values.max(0), color=color, alpha=.10)
            axis.plot(expected_times * 1000, np.median(values, axis=0), color=color, label=label, lw=2)
        axis.axvline(50, color='#94a3b8', lw=1, ls='--')
        axis.set(xlim=(0, 200), ylim=(0, None), xticks=(0, 50, 100, 150, 200),
                 xlabel='策略取樣時刻（ms）', ylabel=unit, title=title)
        axis.grid(alpha=.18)
        if j == 0:
            handles, legend_labels = axis.get_legend_handles_labels()
    matrix = fig.add_subplot(grid[1, :])
    levels = np.array([[2 if row['outcome'] == 'legal_return' else int(row['hit']) for row in rows]
                       for rows in episodes])
    result_colors = ['#e2e8f0', '#e8b76b', '#389475']
    matrix.imshow(levels, cmap=ListedColormap(result_colors), norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3), aspect='auto')
    for i, rows in enumerate(episodes):
        for j, row in enumerate(rows):
            matrix.text(j, i, OUTCOMES[row['outcome']], ha='center', va='center',
                        fontsize=10, color='white' if levels[i, j] == 2 else '#243746')
    summaries = [f'{label}  [{sum(row["hit"] for row in rows)}/5]'
                 for label, rows in zip(labels, episodes)]
    matrix.set(xticks=range(5), xticklabels=SERVE_IDS, yticks=range(4), yticklabels=summaries,
               xlabel='同五顆已研究的 train 來球 ID', title='整球結果：列末為有效碰拍數／5；各格保留最後失敗或成功類別')
    matrix.set_xticks(np.arange(-.5, 5, 1), minor=True)
    matrix.set_yticks(np.arange(-.5, 4, 1), minor=True)
    matrix.grid(which='minor', color='white', linewidth=2)
    matrix.tick_params(which='minor', bottom=False, left=False)
    fig.suptitle('BC 自主回饋診斷：前動作與關節速度', x=.04, ha='left', y=.975, fontsize=20, weight='bold')
    fig.text(.04, .926, '固定 seed 7101／相同權重、五球、控制與物理｜只替換 Actor 輸入副本的完整五幀欄位｜沒有重訓',
             fontsize=11, color='#475569')
    fig.legend(handles, legend_labels, loc='center', bbox_to_anchor=(.51, .875), ncol=4, frameon=False)
    fig.legend(handles=[Patch(color=color, label=label) for color, label in zip(result_colors,
               ['未有效碰拍', '碰拍但回球失敗', '合法回球'])], loc='center', bbox_to_anchor=(.51, .136),
               ncol=3, fontsize=10, frameon=False)
    fig.text(.04, .035, '曲線為五球中位數，陰影為五球最小至最大值；不是信賴區間。虛線：50 ms 反應等待結束。\n'
             '誤差均對同時刻示範計算，位置／速度取實際量測，沒有使用替換後欄位冒充改善；最後取樣點為 187.5 ms。\n'
             '混合實際位置與示範速度可能產生不一致觀測；這是機制診斷，不是可部署政策、BC 門檻通過或泛化證據。',
             fontsize=9, color='#475569')
    fig.subplots_adjust(left=.145, right=.97, top=.81, bottom=.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT/'.agents/runs/foundation/m7/deepmind_five/training.json')
    parser.add_argument('--output', type=Path, help='Figure path; defaults to deepmind_five_{ppo,bc,feedback}.png by mode')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--bc', type=Path, help='Completed contact-BC directory; switches to the BC comparison figure')
    mode.add_argument('--feedback', type=Path, help='Completed feedback.json; plot no-training observation interventions')
    parser.add_argument('--bc-ppo', type=Path, help='Optional completed BC-initialized PPO policies directory')
    args = parser.parse_args()
    if args.bc_ppo is not None and args.bc is None:
        parser.error('--bc-ppo requires --bc')
    suffix = 'feedback' if args.feedback is not None else 'bc' if args.bc is not None else 'ppo'
    output = args.output or ROOT / 'doc/notes/images' / f'deepmind_five_{suffix}.png'
    if args.feedback is not None:
        draw_feedback(args.feedback, output)
    elif args.bc is not None:
        draw_bc(args.bc, output, args.bc_ppo)
    else:
        draw(args.report, output)


if __name__ == '__main__':
    main()
