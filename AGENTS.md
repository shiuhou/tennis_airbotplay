# AIRBOT Play 工程入口

先閱讀 [engineering.md](engineering.md)，再依序讀取 [SPEC](.agents/docs/SPEC.md)、[DESIGN](.agents/docs/DESIGN.md)、[ROADMAP](.agents/docs/ROADMAP.md) 和本地 `.agents/plans/PLAN.md` 的「§0 進度與交接」。SPEC 定義驗收，DESIGN 記錄設計，ROADMAP 記錄階段依賴。PLAN 是唯一活動計畫。運行入口見 [README](README.md)。

- 確認工作區與基線後再修改。保留既有使用者改動及上游原檔、授權與來源 commit。衍生資產與來源檔分開。
- 正常運動透過關節驅動，直接狀態寫入限初始化及重置。接入自製末端前先確認安裝變換，近似末端物理參數須由使用者確認。
- 以實際運行結果區分 PASS、FAIL、NOT_RUN、BLOCKED。測試放在 `tests/`，本地證據放在 `.agents/runs/foundation/`。
- 每次完成工作，交付前同步檢查並更新相關 Markdown：PLAN 記進度與下步、ROADMAP 記日期與驗證結果，SPEC／DESIGN／README 隨需求、設計及入口變更更新。特別維護 `doc/notes/`：每個里程碑都記錄做了什麼、為什麼、怎麼完成、程式資料流與關鍵代碼理解、驗證及限制。可按 `mx_notes.md` 或跨階段主題分檔，以目錄 README 索引；寫成可理解實作的研發筆記，不只列進度或簡短知識點，保留有效踩坑與負結果。
- `.agents/` 下僅追蹤 `docs/`。公開操作說明不依賴本地計畫。支援裸臂與已確認參數的固定球拍仿真，真機與發布須另有授權。不得擅自提交或重寫歷史。
