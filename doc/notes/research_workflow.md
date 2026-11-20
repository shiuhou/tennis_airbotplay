# 研究流程：讓每次實驗都能說清楚

這次採納通用 RL repo 的重點，是把「我們改了什麼、結果能證明什麼」整理清楚。PACE → AIRBOT 仍是主線；沒有搬入 CartPole 訓練器，也沒有改用另一套 PPO。這些規則同樣適用於 M7、目標條件化、感知誤差及之後的真機研究。

## 1. 先寫一個能被推翻的問題

例如：「其他條件相同時，加入 apex 引導是否提高碰球率？」比「把 reward 改好」更容易驗證。

每次實驗在既有活動計畫中記下：

- **問題與假設**：預期哪個指標改善？什麼結果表示假設不成立？
- **共同條件**：來球分布、目標、物理步長、評測集、訓練預算、成功定義。改了這些，就不再是原來那個公平比較。
- **唯一主要變因**：例如有無 hit guidance。A–E 消融本來就允許 observation／reward 不同，不要求整份設定雜湊相同。
- **事先選定指標**：碰球率、合法回球率、落點誤差；不要看完結果才挑最漂亮的一項。
- **停止與失敗條件**：時間／資源上限、NaN、不可達、碰撞等。失敗和被拒絕的來球都需要交代。

活動待辦仍只在本地 PLAN；公開方法與結論寫進相應 milestone notes。不要再維護第二套任務看板。

## 2. 工程通過，不等於研究假設成立

兩個問題要分開回答：

| 問題 | 例子 | 如何記錄 |
| --- | --- | --- |
| 程式是否按規格工作？ | checkpoint 能載入、reset 不跨回合、評測完整跑完 | PASS／FAIL／BLOCKED／INCOMPLETE／NOT_RUN |
| 方法是否有幫助？ | 加引導是否真的比不加更有效？ | SUPPORTED／REFUTED／INCONCLUSIVE，附比較條件與不確定性 |

M6 的 11.6 mm 可以通過既定 30 mm 工程門檻；但窄分布下，訓練均值也有 13.0 mm，不能因此宣稱 predictor 已具廣泛優勢。這正是兩種結論不能混用的例子。

## 3. 已落地：按環境執行測試

在 repo 根目錄：

```bash
./run.sh --unit             # research、core、pace、camera 全部分組執行
./run.sh --unit pace        # 只檢查 PACE／M6，不啟動仿真或訓練
bash scripts/verify_groups.sh research
python3 scripts/verify.py research   # 直接使用目前 Python
```

- `core` 和 `pace` 使用既有固定 Isaac image；`camera` 使用相機獨立 venv；`research` 只需 Python 標準庫。
- 缺少環境／套件是 BLOCKED；測試有 skip 或零項是 INCOMPLETE；測試失敗是 FAIL。不能把「沒測到」當成 PASS。
- 報告在本地 `.agents/runs/foundation/verification/`。這只是單元驗證，完整 Foundation／M5／硬體驗收仍走原入口。
- 調度器退出碼：0 通過、1 測試失敗、2 未完整驗證。若同時有失敗與缺環境，以 1 回報。

程式資料流：`run.sh --unit` → `verify_groups.sh` 選環境 → `verify.py` 選測試 → `unittest` → 檢查錯誤及 skip → JSON 報告。新增測試模組時須加入分組，不能默默遺漏。

## 4. 已落地：執行紀錄與離線比較

[research.py](../../scripts/research.py) 是小型命令包裝器，不是訓練器或實驗平台。先把一份實驗條件 JSON 放在本地 runs，例如：

```json
{
  "question": "相同預算下，hit guidance 是否提高碰球率？",
  "kind": "experiment",
  "common": {
    "evaluation_set": "填入實際資料雜湊",
    "physics_hz": 960,
    "training_budget": "填入預先確定的步數",
    "success_rule": "填入成功定義與版本"
  },
  "method": {"hit_guidance": true}
}
```

這是格式示意，不是已批准的 M7 設定。`common` 是要固定的比較條件，`method` 是刻意改變的方法；具體內容需要研究者正確填寫，工具無法替你判斷實驗設計公平與否。

執行語法（`your_program.py` 代表實際研究入口，並非倉庫既有檔案）：

```bash
python3 scripts/research.py run \
  --directory .agents/runs/research/trial_a_seed1 \
  --protocol .agents/runs/research/protocol.json \
  --input configs/pace.json --timeout 600 \
  -- python3 your_program.py --output '{run_dir}/result.json'
```

程式先建立**不能重用的目錄**，記錄 commit、dirty 狀態、所有非忽略檔案的雜湊、額外指定輸入、條件、命令與啟動環境，再執行 argv；不經 shell 解讀。最後記錄退出碼、耗時和產物雜湊。超時會終止同一程序群組；Docker 等外部服務仍需自己的停止策略，不能把群組終止當成遠端工作已停止。

子程式應輸出 `result.json`；若已有其他結果檔，可用 `--result 子路徑/結果.json`。最小工程結果只要求 `status`；要做訓練比較則需要：

```json
{
  "status": "PASS",
  "training_seed": 1,
  "runtime": {"python": "實際版本", "torch": "實際版本", "device": "實際設備"},
  "metrics": {"hit_rate": 0.8}
}
```

數字只是格式例子，不是實驗結果。啟動器與工作程序可能不同，例如主機啟動 Docker；所以工作程序須回報自己的 runtime。退出碼 0 卻沒有有效結果檔，會標記 INCOMPLETE。

```bash
python3 scripts/research.py compare --metric hit_rate \
  .agents/runs/research/trial_a_seed1 .agents/runs/research/trial_b_seed1
```

比較只讀既有產物，不載入 Torch 或開始訓練。它檢查共同條件及產物是否被改動，按方法列出每次獨立訓練的值、平均及樣本標準差；同方法重複 seed 會拒絕，失敗不會默默消失。`kind: analysis` 的重畫圖不能當作一次訓練。

**目前只提供描述統計，科學結論固定為 INCONCLUSIVE。** 尚未實作 bootstrap、配對推論或顯著性檢定；不要從一兩個 seed 的平均值宣稱方法勝出。64 個 held-out 發球是一次訓練的評測樣本，並非 64 次獨立訓練。

限制也要明確：雜湊可辨認版本，不能還原未提交程式；正式可重現發表仍需保留對應程式版本及資料。紀錄不是防惡意篡改簽章；研究者若同時重寫 JSON 和雜湊，工具無法保證真實性。命令輸出存本地 `execution.log`，請使用精簡輸出並限制執行時間；尚無 log 輪替或磁碟配額管理。不要把憑據放入命令參數。

## 5. 已落地：M6 訓練、評測、畫圖分離

之前畫圖在訓練腳本尾端，要重新走訓練才能重建圖。現在：

1. `verify_m6.py --stage train`：只使用 train／validation 選 checkpoint，寫 `training.json`，不跑 held-out 推論。
2. `--stage evaluate`：核對來源、設定、資料與 checkpoint 雜湊及開發門檻，再評測；不呼叫 `fit_epoch`。
3. `plot_m6_results.py`：只讀 `validation.json`，不匯入 Torch。第三張圖使用既有逐球誤差，毋須重做推論。

以下前兩個命令須在既有 Isaac Python 環境、設定 `PYTHONPATH=src:tests` 後執行：

```bash
python scripts/verify_m6.py --stage train --output-dir .agents/runs/foundation/m6/new_trial
python scripts/verify_m6.py --stage evaluate --output-dir .agents/runs/foundation/m6/new_trial
python scripts/plot_m6_results.py .agents/runs/foundation/m6/new_trial/validation.json \
  --output .agents/runs/foundation/m6/new_trial/predictor.png
```

畫圖只需要 Matplotlib，可在任何已安裝它的環境執行。需要一鍵從主機啟動既有容器的完整流程，使用：

```bash
./verify_m6.sh --reuse-data .agents/runs/foundation/m6/new_trial
```

輸出目錄必須是新的；評測也不覆蓋既有 `validation.json`。原本 M6 數據及代表圖保留。重跑原 seed 是程式回歸，不是增加研究樣本；舊資料的重新分析也不叫新 held-out 驗證。

## 6. 後續階段怎樣使用

| 階段 | 要帶入的習慣 | 目前狀態 |
| --- | --- | --- |
| M7 PPO | 固定預算與評測；記 KL、entropy、clip fraction、value loss，以及關節飽和、接觸、NaN、吞吐 | 規約已整理；PPO 接入未開始 |
| M7 消融 | 先原方法適配，再分離 hit／return guidance、predictor；多個獨立訓練 seed；同條件比較 M5 | 描述比較工具已可用；統計推論待具體實驗 |
| M8 目標條件化 | 將訓練目標和未見目標拆開，逐目標報告；不能只看整體平均 | 設計約定，尚無實驗 |
| 誤差／sim-to-real | 分別施加延遲、感知噪聲、安裝誤差，保留單變因與負結果 | 設計約定，尚無硬體操作 |
| 長時間訓練 | 保存 optimizer、normalizer、RNG、步數、環境狀態及資料狀態，驗證恢復前後行為 | 尚未實作；執行紀錄不等於 resume |

## 7. 來源與本次驗證

參考檔保留在 [rl_engineering_v3](../../assets/upstream/rl_engineering_v3/provenance.json)，附原 LICENSE 和逐檔 SHA256。選入的三份研究／評測／驗證文件保持原文；其中相對連結屬原 repo，並非本工程操作入口。未複製其 trainer、task manager 或依賴環境。此頁才是 AIRBOT 的實際用法。

2026-11-19：研究工具 10 項、core 31 項、PACE 17 項、camera 11 項通過；缺少主機數值套件時正確回報 BLOCKED。研究工具測試用合成 fixture，涵蓋拒絕覆寫、重複 seed、產物變動、不相容條件、失敗保留、超時和 skip，不能當作 RL 效果證據。M6 分段回歸與無 Torch 繪圖結果見 ROADMAP。

M6 分開 train／evaluate 的流程回歸 PASS：學習曲線、逐球誤差、分段結果和 checks 與既有報告完全一致，held-out RMSE 仍為 11.5803 mm。禁止 Torch import 後，從原 JSON 重畫圖亦 PASS。新圖和回歸 checkpoint 只留本地 `m6/workflow_check/`，未增加 Git 圖片或重新採集仿真。
