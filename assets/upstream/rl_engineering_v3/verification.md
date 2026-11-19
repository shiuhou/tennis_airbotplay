# 按改動影響選驗證

先問「這次變動可能破壞什麼」，再挑驗證。證據足夠即停止，不以測試數量當進度。

```bash
python scripts/verify.py --list
```

| 情況 | 最小起點 | 何時擴大 |
| --- | --- | --- |
| 純文字／筆記 | 審閱內容與受影響連結，必要時命令樣例檢查 | 修改了執行指令或接口契約 |
| 局部純邏輯 | 對應單元測試或 `quick` | 跨模組輸出改變 |
| 十欄狀態工具 | `--group tasks` | 狀態與其他 CLI 接合有變 |
| runner／預算／恢復 | `--group harness` | 連接真實 adapter 變動時加一個 smoke |
| RL 數值邏輯 | `--group rl-numeric`（RL 版） | rollout／checkpoint／評測接口改變 |
| RL adapter／比較 | `--group adapter` 或 `--group rl-analysis`（RL 版） | 主要流程接合變動或正式交付 |
| 新環境首次使用 | doctor、quick；需要研究才做一個 smoke | 使用的依賴／平台尚未驗證 |
| 基礎設施整合／版本交付 | `full` 一次 | 只為具體尚未覆蓋風險追加 |

`quick`（別名 fast）不啟動 PPO 訓練，也不跑 controller 的 timeout／SIGKILL 套件。`full` 內已包含 quick 的範圍，沒有先跑 quick 再 full 的必要。指定多個 group 時選一次聯集。實際可用群組以 `--list` 為準。

RL full、rl-numeric、adapter 等明確需要 RL 的入口缺少依賴時會 BLOCKED，不把 skip 當作已完成數值／訓練驗證。一般 quick 可在未安裝 torch 的環境使用，但它只證明相應的便宜檢查。

## 重用與失效

`tasks.py verify --reuse-if-fresh` 只重用同一 feature／整體入口的有效證據，核對輸入、驗收、工具、環境和 evidence；回覆明示 REUSED。`--force` 真正再跑。研究結果不作此種快取，重跑同一 seed 也不增加獨立訓練樣本數。

STALE 不等於 FAIL，也不要求馬上重驗。保留歷史結果，完成預定修改後再驗證；若依賴變動與本輪無關，可用明確、完整的 evidence inputs 減少不必要失效。不要為了綠燈故意省略真依賴。

## 狀態與限度

PASS／FAIL 是實際受測行為；NOT_RUN／BLOCKED 必須列原因。REUSED 表示引用原證據，STALE 表示不覆蓋現況。方法有效、工程可運行、可部署與自己理解需要不同證據。

無需為註解或低風險可逆文字改動建立鏡像測試。真正測試應能暴露錯誤、界線或負例。大型訓練交給研究 runner／專案調度器，不藏在便宜的任務驗證命令裡。

模板的 Linux 本機驗證不等於 Windows、macOS、GPU、模擬器或實機通過；CI YAML 存在也不等於遠端執行成功。見 [本次驗證](template-validation.md)。

驗證退出碼：0=所選範圍 PASS；1=FAIL；2=BLOCKED 或 INCOMPLETE。缺少所需依賴為 BLOCKED；任何所選測試被 skip（包括平台特有測試）為 INCOMPLETE，不能宣稱完整通過。跨平台便宜檢查與 Linux 完整 gate 分開配置。
