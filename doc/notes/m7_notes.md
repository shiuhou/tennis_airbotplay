# M7：把環境接到 PPO，再檢查引導是否有用

M6 教會我們怎樣預測、怎樣計分；M7 才開始讓策略用自己的動作與環境互動。本輪範圍是 M7.1–M7.3，加上 A／B 小規模診斷，並非完成五組正式研究。

## 1. 這次要解決什麼

先回答「訓練鏈是否正確」，再回答「引導是否幫忙」。順序是：

1. 建立多份互不干擾的機械臂／球桌／球環境。
2. 確認六維 action 經驅動，確實能做出已知可行的動作。
3. 用官方 PPO 收集資料、更新 actor／critic，保存及重載模型。
4. A 不加 hit guidance；B 加同一個 apex 距離獎勵，其他條件相同。

A／B 測的是**獎勵引導**，不是 MLP 是否有用。兩組 actor 都沒有 predictor 欄位，沒有載入 M6 checkpoint；D／E 的預測提示比較留在後續。

## 2. 環境怎樣變成可以訓練

[pace_env.py](../../src/airbot_tt/pace_env.py) 使用固定版本 Isaac Lab 的 `InteractiveScene` 複製場景。各環境有自己的球、關節狀態、接觸、發球亂數、時鐘及觀測歷史；碰撞群組隔離。位置先扣除各環境的 origin，否則同一個姿態放在第二張桌上，網路會誤以為球跑遠了。

資產沒有重新畫一套：沿用原 URDF、固定件／球拍、247 g 質量近似、TCP 與驅動設定。`runtime.configure_racket_colliders` 是舊單環境與新環境共用的工具建構；`BallTestbed` 補上場景路徑參數，球桌、球網與球的物理設定仍共用。

事件由 [TensorShotEvents](../../src/airbot_tt/pace_control.py) 在 tensor 上批量處理，並與原 `ShotEvents` 對照：己方先反彈 → 合法拍面接觸 → 過網 → 對方首次落桌。撞柄、fixer、手臂、球網等不當成成功；接近球不等於碰球。

一個策略步的資料流：

```text
115 維 actor 觀測 → 六維抽樣動作 → 限制與插值 → 20 個 PhysX 子步
                                    ↓
                        接觸／事件／分項獎勵
                                    ↓
                    保存末態觀測 → 局部 reset
                                    ↓
                    原版 storage → GAE → PPO
```

actor 每幀 23 維、5 幀共 115 維：關節位置差、關節速度、前動作、球位置、量測 valid 和 age。critic 每幀 40 維、5 幀共 200 維，另外可看真值球速度、教師點、TCP、任務階段等訓練特權。actor 沒有偷看 critic。

目前是同步、無噪聲的仿真觀測。影像偵測、相機延遲、缺樣及在線 predictor 仍未接入。

## 3. 控制整合實際踩到的坑

**M5 能成功，不代表把 M5 位置直接送給新 action 就會成功。**

初版把每個位置目標當成需要靠近並煞停的目的地。M5 卻需要連續通過擊球點，帶著指定拍速走過去。兩種介面的意思不同。

實測直接重播，在 48 Hz 下產生約 **38.7 mm** TCP 位置誤差；即使提高到 960 Hz，仍約 **27.8 mm**。這表示主因是指令處理，而不是 PPO、apex 或單純取樣頻率。

曾檢查煞車距離補償及局部動作擬合。它們沒有讓原定 48 Hz 達到原 M5 門檻；未把失敗抹掉，也沒有把策略悄悄改成 960 Hz。這些補償未留在生產控制器。

現在 [PositionCommand](../../src/airbot_tt/pace_control.py) 把 action 解讀為**本策略區間結束時要求的位置**：

\[
q_{request}=q_{nominal}+s\odot clip(a,-1,1)
\]

用目前指令位置／速度作起點，從最近三筆位置要求估計終點速度，建立三次插值。每個 960 Hz 子步再限制速度與加速度，經驅動施加；不直接改機械臂狀態。

這裡的三次曲線是低階指令平滑，不是 M5 規劃器。訓練時 actor 仍只輸出六個數字，不會取得已知軌跡、IK 或未來擊球答案。

M5 中央擊球軌跡經此接口，在原 48 Hz 設定下通過原驗收：位置約 **0.91 mm**、線速度約 **24.8 mm/s**、時刻約 **0.58 ms**。尺度目前是各軸 0.6 rad 候選包絡，已確認中央軌跡最大 residual 約 0.33；不代表整個 action 空間都無碰撞，隨機動作仍由物理接觸與故障終止。

## 4. 怎樣避免「重置後的球」污染上一球

例如球因時間用完而結束，critic 應評估的是**舊球結束時的狀態**，而不是重置後又出現的新球。

因此環境先計分、保存 `terminal_actor`／`terminal_critic`，再只 reset 已結束的環境。每一步重新產生 terminated／truncated。

- 真正失敗或成功：`terminated`，不補未來價值。
- 本輪 timeout 按既定計畫視為人工截斷：`truncated`，補 `gamma × V(terminal_critic)`。這是所選任務語義，不可無條件外推到其他任務的自然截止。

[train_pace.py](../../scripts/train_pace.py) 在呼叫官方 `process_env_step` 前補一次末態價值；不傳原版 `time_outs` 欄位，避免原版再用舊 value 補第二次。測試把終止與截斷放在同一批，核對真正進入官方 storage／returns 的值。

Gaussian 的原始動作及 log probability 存進 PPO；環境的 action clip 是另外一個明確轉換，不把限幅後的動作冒充原抽樣值。

### PPO 更新實際發生在哪裡

薄接合的核心流程可以對照這幾行理解：

```python
action = algo.act(obs, critic)       # 保存舊觀測、原始動作、log_prob 與 V
obs, critic, reward, done, info = env.step(action)
value = policy.evaluate(info["terminal_critic"])
corrected = bootstrap_timeout(reward, value, info["truncated"], gamma)
algo.process_env_step(corrected, done, {})
# 收滿 24 個策略步之後：
algo.compute_returns(critic)
losses = algo.update()
```

16 個環境 × 24 步，形成一次更新的 384 筆 transition。官方 storage 用 reward、value 和結束遮罩算 returns／GAE；actor 調整動作機率，critic 擬合回報。PPO 每次做 5 epochs、4 minibatches，並限制機率比的變化。保存的是收集時真正使用的觀測，更新時不重新生成舊觀測。

例如某個動作最後導致錯誤反彈，它可能仍拿到首次碰拍分，但拿不到合法回球分。critic 與 advantage 會受整段結果影響；不能只看碰拍那一瞬間就說策略學得對。完整原版公式與對應檔案見 [PACE 資料流研讀](pace_pipeline.md)。

## 5. A／B 到底比較什麼

共用 reward：首次合法碰拍 1 分、合法回球 5 分；動作變化懲罰按 dt 計算，故障扣分。這些是 AIRBOT 單拍任務設定，不冒稱原版 T1 的完整 reward。

B 額外使用原 hit guidance 形狀：

\[
r_{hit}=\exp[-\max(d,0.15)/0.25]
\]

乘權重 2 和實際子步 dt。進入 15 cm 內就是同一個最高分平台；8 cm 不會比 12 cm 得到更多這一項分數。拍面方向和時間不對，位置分很高仍然可能漏球。

環境逐物理子步記錄 apex 有效秒數、平台停留時間、累積 guidance 和拍球最近距離；用真實事件記碰拍與回球。**不能拿 B 的總 reward 比 A 高，就說 B 學得更好。** 要比較相同互動步數下的共同任務指標。

教師目前採用 M6 CPU 參考，在每次發球初始化時求未受拍影響的 apex；後續使用剩餘時間與事件遮罩，不在每個策略步逐環境呼叫 SciPy。這是乾淨零旋來球下的初始整合，尚未完成 batched GPU 教師或感知擾動後的重估。

## 6. 短試跑如何判讀

先量得 2／16 環境的單拍吞吐，再以 16 環境進行診斷；不預設 4096 環境可以跑。48 Hz 策略、960 Hz 物理不變，5 幀跨度 83.33 ms。

- smoke：A 組，4 次 update × 24 步 × 16 環境 = 1,536 transitions；上限 30 分鐘。確認參數更新、非 NaN、episode、保存和重載，不要求學會回球。
- A／B：各 3 個 seed，每次 16 update，即每個 seed／方法 6,144 transitions。這是小規模診斷，遠少於正式訓練預算。
- 固定評測 seed 7201，每環境預先指定第一顆來球；等待每一顆都有結果。較快失敗的環境不能以第二顆球取代其他環境尚未完成的第一顆球。
- 記錄 KL、PPO clip fraction、entropy、value／surrogate loss、action clip、指令追蹤及共同任務結果。評測取 actor 均值，不需要 critic 決策。

checkpoint 保存 policy、optimizer、RNG、設定與來源，但目前沒有提供正式 resume；保存這些項目不等於已驗證續訓。此次也不包含 C／D／E、M8 或真機操作。

## 7. 重跑入口與證據

在 repo 根目錄，先完成既有 Isaac 安裝及官方 PACE 來源準備：

```bash
python3 scripts/prepare_pace.py
./run.sh --unit pace
./run.sh --mode pace-check --pace-check reference --pace-envs 1 --device cuda:0 --headless --report .agents/runs/foundation/m7/reference.json
./run.sh --mode pace-check --pace-check parity --pace-envs 1 --device cuda:0 --headless --report .agents/runs/foundation/m7/parity.json
./run.sh --mode pace-check --pace-check lifecycle --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/m7/lifecycle.json
./run.sh --mode pace-check --pace-check control --pace-envs 1 --device cuda:0 --headless --report .agents/runs/foundation/m7/control.json
./run.sh --mode pace-smoke --pace-envs 16 --device cuda:0 --headless --pace-output .agents/runs/foundation/m7/new_smoke --report .agents/runs/foundation/m7/new_smoke_summary.json
```

A／B 使用 `--mode pace-ab`，同樣指定新的 `--pace-output` 和 `--report`。訓練入口先核對三份 gate 報告為 PASS，且來源雜湊與目前一致。結果、checkpoint、逐次更新及失敗紀錄留本地 runs；不能重用已存在的訓練輸出目錄。

實際結果見下節；重跑命令本身不代表已完成訓練。


## 8. 這輪實際得到什麼（2026-11-19）

**工程鏈已接通，但這次短跑沒有得到有效回球策略。**

- 16-env 的觀測、reset 隔離、重置後 TCP 運動學、末態觀測／timeout 均 PASS；固定姿態單環境與舊 SingleShot 的同時刻球位置差約 `1.5e-11 m`，結果一致。這是特定案例交叉檢查，不是所有動作等價的證明。
- 48 Hz 中央軌跡控制 PASS；本機 16-env 單拍流程約 **85.5 policy samples/s**，包含此流程的 reset／教師成本，不是純 GPU 物理 benchmark，也不外推 4096-env。
- smoke PASS：1,536 transitions，actor／critic 更新範數分別約 1.816／1.354，checkpoint 重載輸出一致，含評測約 33.1 秒。
- A／B 各 3 seeds、每次 6,144 transitions，共 **36,864 transitions**；完整流程約 522 秒。每個環境實際只累積約 8 秒模擬經驗，所以只把它當小試，不當充分訓練。

每個 seed／方法評測相同的 16 顆預先指定來球，全部計入：

| 方法 | Training seed | 碰拍 | 合法回球 | 目標框 | 主要失敗 |
| --- | --- | --- | --- | --- | --- |
| A | 7101 | 16/16 | 0/16 | 0/16 | wrong_bounce |
| A | 7102 | 16/16 | 0/16 | 0/16 | wrong_bounce |
| A | 7103 | 0/16 | 0/16 | 0/16 | out |
| B | 7101 | 0/16 | 0/16 | 0/16 | out |
| B | 7102 | 0/16 | 0/16 | 0/16 | out |
| B | 7103 | 0/16 | 0/16 | 0/16 | out |
| 零動作固定準備姿態 | 非訓練策略 | 16/16 | 8/16 | 0/16 | 另 8 顆未合法回球 |

![M7 控制修正與小規模 A/B](images/m7_diagnostic.png)

讀這張圖時注意三件事：

1. **碰拍不等於回球。** A 的兩個 seed 能碰球，但下一次落桌順序／位置不合法，沒有完成過網後對方落桌。
2. **此輪沒有看到 B 的幫助。** 但 A 自己也有一個 seed 完全漏球，而且訓練很短；目前結論是 INCONCLUSIVE，不是已證明 apex 引導無效。
3. **零動作已能完成部分合法回球。** 目前學出的動作尚未優於這個簡單基線；不能把「網路權重改了」當成任務進步，也不能拿舊 M5 的其他評測集直接充當本次同球對照。

平台診斷也有用：六次評測都沒有進入 15 cm 最高分平台。以 seed 7101 為例，A 的有效引導平均約 0.640 秒、累積分 0.156；B 約 0.844 秒、累積分 0.219。B 的累積分較高同時伴隨更長的有效階段，不能只看累積值就說位置準備更好，更不能說已觀察到在最高分平台停留刷分。

### 保留的工程負結果

首次 smoke 的更新／保存已執行，但切換評測時，Isaac 的可變張量遇到 PyTorch InferenceMode 限制，因此整次記 FAIL。改用 `no_grad` 並補齊官方 reset 後的 `sim.forward()`，重新通過生命週期 gate 和 smoke。失敗 checkpoint／報告留本地 `smoke_initial_failure/`，不當作成功模型或額外獨立 seed。

控制器的原始煞停、頻率診斷及局部擬合失敗報告也保留；未把 480 Hz 的 30.83 mm/s 誤差放寬為通過。最終修正在指令插值，不是調整門檻或替換 apex 教師。

### 接下來應查什麼

先理解 A 的接觸後出球與 B 的漏球行為，再決定下一個對照預算。需要分辨拍面姿態／拍速、探索、動作限制與 reward 的影響；不能僅憑這輪短跑換教師或轉向其他架構。C／D／E、在線 predictor、正式多 seed 效果推論及 M8 仍未完成。

圖由 `scripts/plot_m7_results.py` 只讀既有 JSON 產生，不載模型或重訓。Git 只收這一張代表圖；逐次更新、checkpoint 和完整報告留本地。

收尾驗證：73 項分組單元通過；原 M5 development 的三條空揮拍及 9/9 合法回球通過。原 36 球獨立驗證未重跑，這次回歸不增加新的研究樣本。

另有一輪 M5 回歸在數值報告完成後，外層程序退出 143，未得到完成確認；保留為 `m5_regression_shutdown_incomplete.json`。退出原因未定位，不能把該次稱為完整 PASS。重跑同入口後三條空揮拍、9/9 回球及退出碼 0 均確認完成。
