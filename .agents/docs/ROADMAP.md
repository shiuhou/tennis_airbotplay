# 路線圖與研發紀錄

日期採 `YYYY-MM-DD`（America/New_York）。「確認日期」表示已有結果可確認的日期，不代表工作起始日；未排定項目以「待定」標示。後續在本頁追加有實際結果的里程碑，記錄問題、處理方式與驗證，活動待辦仍由本地計畫管理。

## 階段進度

| 階段 | 確認日期 | 狀態 | 驗收或依賴 |
| --- | --- | --- | --- |
| 資產與環境 | 2026-11-16 | PASS | 固定來源與執行版本；核對六軸鏈、限制、裸臂碰撞及慣量；上游檔案校驗 10/10 |
| 固定底座 Foundation | 2026-11-16 | PASS | 完成 SPEC 的運動、FK／IK、軌跡、碰撞、Stop／Reset 與正常退出驗證 |
| 可重現交付 | 2026-11-16 | PASS | 固定 headless 入口，完成數值回歸及 Kit 渲染影片；專案已推送 GitHub |
| Fixer／球拍 CAD 配合 | 2026-11-16 | PASS（名義幾何） | 平底 fixer 三孔對準、安裝面貼合、球拍靠止擋；確認 fixer／link5 的全 J6 轉角淨空 |
| 自製末端物理接入 | 2026-11-16 | PASS（已確認近似） | 總載荷 247 g、均勻 CAD 分布；固定剛體、碰撞、慣量與 Racket TCP 已接入，工具及裸臂完整回歸通過 |
| 實物裝配與真機 | 待定 | NOT_RUN | 待確認螺紋、鎖固、公差與強度；真機操作須另有授權 |

## 研發紀錄

| 日期 | 項目／問題 | 處理與結果 | 驗證或成果 |
| --- | --- | --- | --- |
| 2026-11-16 | 裸臂模型與固定底座仿真 | 採官方六軸描述，保留來源、授權及版本；完成關節驅動、FK／IK、軌跡與碰撞監測 | `./verify.sh` PASS；數值測試 6/6；環境與來源見 [SOURCES](SOURCES.md) |
| 2026-11-16 | 靜止時 J2／J3 速度回饋偏差 | 調整 PD 增益未消除偏差；改用 PhysX 重力前饋後，靜止位置與速度回到數值零附近 | Foundation 驗證 PASS；正常運動直接狀態寫入 0，固定底座漂移 0 |
| 2026-11-16 | 啟動失敗可能被誤判為成功 | 以每輪 token、報告及進程退出碼共同判定，關閉前保存結果 | [啟動入口](../../run.sh)與 [Foundation 測試](../../tests/foundation.py) |
| 2026-11-16 | 舊 fixer 套筒干涉 link5 | 僅檢查 link6 貼合不足；完整手腕檢查確認 9 mm 深套筒干涉。最終移除安裝面後方整段套筒，採使用者新版 `fixer_airbot(2).STEP` | 全 J6 連續 360° 的 fixer／link5 名義軸向淨空至少 0.946981 mm；實物公差與強度未驗證 |
| 2026-11-16 | 新 fixer 與球拍裝配 | 保留三個 Ø4.20 mm 孔及 Ø24.30 mm 孔心圓；球拍柄尾置中靠止擋，先裝 fixer 再插入球拍 | CadQuery／OCP 回讀三個有效實體，兩兩交集體積 0；柄尾接觸 489.21 mm²；[裝配 STEP](../../assets/derived/fixer_fit/airbot_j6_fixer_paddle_fit.STEP)、[配合圖](../../assets/derived/fixer_fit/airbot_j6_fixer_paddle_fit.png) |
| 2026-11-16 | 首次 GitHub 交付 | 推送仿真專案、最新版 fixer 原檔、裝配與配合圖；本地計畫及執行紀錄不納入 Git | 提交 [`0d85854`](https://github.com/shiuhou/tennis_airbotplay/commit/0d858543fcebe70b07fae1a4bc04cdfd04c7f606)；推送前數值測試 6/6、上游校驗 10/10 PASS，未重跑完整 Isaac Sim |
| 2026-11-16 | 固定球拍由 CAD 接入物理 | 使用者確認固定件 65 g、8 粒螺絲共 12 g、球拍 170 g，以及均勻 CAD 分布近似與拍面 TCP；新增兩個固定剛體、碰撞、COM、慣量及工具接觸感測 | 工具數值測試 12/12、仿真驗收 10 項、固定件／球拍各自接觸與停止鎖定、14.6 s Kit 影片及解碼均 PASS；裸臂完整回歸 PASS。入口 `./verify.sh --tool-config configs/racket.json` |

| 2026-11-16 | 清理退役模型與重複產物 | 刪除舊套筒視覺預覽、中間模型與重複 CAD 交付，以及可重建 USD／Python 快取，共 51 檔、175.23 MiB；保留現用資產、上游來源／授權、PDF 與有效驗證證據 | 正式 CAD 交付保留於 `assets/derived/fixer_fit/`，仿真網格於 `assets/derived/racket/`；USD 啟動時重建 |

J6 的 Ø4.00 mm 孔徑與 4.00 mm 孔深為 STL 參考，螺紋尚未確認。CAD 配合通過不代表實物鎖固已驗證；工具物理驗證限定於已確認的質量與均勻 CAD 分布近似。

需求與驗收見 [SPEC](SPEC.md)，模型和控制方案見 [DESIGN](DESIGN.md)，使用方式與 CAD 成果入口見 [README](../../README.md)。
