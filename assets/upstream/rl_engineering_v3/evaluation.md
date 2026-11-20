# RL 評估：主路徑可以分析，結論仍要審查

`scripts/compare_trials.py` 直接讀 V3 runner 的完整 trial。舊 `scripts/rl.py compare` 只讀舊格式，不必兩套都跑。

## 抽樣單位

每個獨立 training seed 先對固定 evaluation episodes 求平均，再以 training run 作比較單位。10 個 episodes 不能冒充 10 次獨立訓練；相同候選／同 seed 的重跑也不能增加獨立 seed 數。

報告顯示兩組均值、差、per-seed 分數與 percentile bootstrap 區間。預設獨立抽樣，`--paired` 要求匹配 seed 集並按 seed 對齊。少量 seeds 得到的區間可以很不穩定，bootstrap 不會創造缺少的樣本；沒有 universal「5 個 seed 就夠」規則。此工具不是多任務 rliable 的等價實作。

## 搜尋與確認

keep／discard 和 `min-improvement` 只決定開發搜尋中保留哪個候選，不能支持統計顯著性。挑很多候選後最好的分數有選擇偏差；報告附同 session 的所有嘗試／失敗，仍不能自動知道你在其他 session 做過多少調參，需 study 級試驗帳。

確認前固定候選、baseline、預算、seeds、checkpoint 選取和分析。confirmation profile 只是事前配置，不能證明 seeds 未被讀過、未被調參使用。若根據確認結果繼續改方法，將它改列探索證據，另訂獨立確認。

## 公平比較檢查

- 相同完整環境與評測條件，避免不同 horizon、action rule、normalization 或 reward scale 混比。
- 所有預定 training seeds 都要有結果或失敗紀錄，不只選成功／最好者。
- 同 steps 是樣本預算公平的一部分；模型大小、調參次數、訓練時間和硬體也需交代。
- 只報 final checkpoint 不能宣稱樣本效率，除非另有預先定義且一致的學習曲線評測。
- OOM／timeout 不是零分，也不是可以靜默刪除的資料；重試規則要事前說明。
- 若不同實驗條件是研究自變量，本比較器可能拒絕；應設計對應分析而不是關掉完整性檢查。

`scientific_claim` 保持 NOT_ESTABLISHED，報告是證據輸入。結論可以是限定條件支持、反駁或證據不足；不要求每次研究都正向提升。

## 一手參考

[Empirical Design in Reinforcement Learning](https://www.jmlr.org/papers/v25/23-0183.html) 討論實驗設計、變異與實驗者偏差。[rliable](https://github.com/google-research/rliable) 提供跨任務可靠評估的參考；其封存狀態不影響方法可讀性，但本模板不把它列為強制依賴。多任務研究要明確 normalization 和 runs × tasks 資料，不混合不同尺度的 raw returns。
