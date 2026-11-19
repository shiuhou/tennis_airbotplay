# Gemini 335L：先用官方參數，再做獨立驗證

目前交接目標是確認這台相機的原廠參數、彩深對齊和深度量測是否適用。**不先重標內參，不做手眼、機械臂操作或 IMU 標定。** 不需要 ROS 2，也不需要啟動 Isaac Lab。

## 1. 哪些用官方，哪些是本工程補的

| 工作 | 現在使用 | 為什麼 |
| --- | --- | --- |
| 連線、裝置資訊 | 官方 `01_hello_camera.py` | 移除自寫裝置枚舉 |
| 讀原廠內外參 | 官方 `04_camera_calibration.py` | 直接執行原檔，不只引用連結 |
| 查看可用模式 | 官方 `04_enumerate.py` | 使用官方互動式列舉 |
| 預覽、錄影、回放 | 官方 `01_recorder.py`／`02_playback.py` | 移除自寫即時預覽、錄製及相機屬性控制 |
| 錄影轉資料集 | 自寫 `camera_recording.py`，重用官方回放與彩色解碼 | 官方沒有我們的 CSV、資料切分與驗證資料格式 |
| 幾何與深度驗證 | 自寫 `camera_calibration.py`＋OpenCV | 官方讀出參數不等於證明本台相機精度合格 |

舊 `camera_device.py`、`inspect`、`capture` 入口已移除，不保留兩套即時相機操作。舊資料不刪；已建立的舊 session 可繼續離線分析，新錄影請建立新 session。

官方來源固定於 [pyorbbecsdk 68cdfa18](https://github.com/orbbec/pyorbbecsdk/tree/68cdfa181e09f0ddd0f21b77dff29cb3e7a0f0d2)。[來源清單](../../configs/orbbec_examples.json)記錄每檔 SHA256；setup 下載未修改原檔與 Apache-2.0 LICENSE 到本地 runs，每次使用前核對。不同版本的範例不混用；發現本地原檔被改過會停止，不覆蓋修改。

## 2. 同學的實作順序

1. **接通、讀參數**：確認型號、序號、韌體及實際串流。官方輸出應真的有數值，不能只看退出碼。
2. **確認標定板**：量格長和 marker 長度，建立 session。
3. **官方錄影**：先短錄、回放，確認有需要的串流及匹配的 JSON sidecar，再錄獨立 validation。
4. **原廠驗證**：匯入後跑 `factory-check` 和 `alignment-check`；不必先產生自行估計的內參。
5. **獨立量距**：每個距離錄一段固定平面，跑 `depth-check`。
6. **有證據才重標**：先排除模式、同步、板子和畸變模型問題，再另錄 fit，與 validation 分開比較。

交回完整 session、原始 BAG＋sidecar 的儲存位置，以及「原廠內參／彩深／深度」各自的結果和最差案例。無法量測、缺串流或沒跑到的項目標 NOT_RUN，不只交 K 矩陣或成功截圖。大型資料不進 Git。

## 3. 安裝與官方入口

在本倉庫根目錄：

```bash
bash scripts/camera_calibration/setup.sh
bash scripts/camera_calibration/run.sh official info
bash scripts/camera_calibration/run.sh official parameters
bash scripts/camera_calibration/run.sh official profiles
```

第二條參數讀取命令實際執行的就是官方 [04_camera_calibration.py](https://github.com/orbbec/pyorbbecsdk/blob/68cdfa181e09f0ddd0f21b77dff29cb3e7a0f0d2/examples/beginner/04_camera_calibration.py)。它以預設 RGB／depth 串流讀參數、列印輸出；不是 JSON 匯出，也不保證預設模式就是正式模式。正式資料的參數另由錄影 profile 匯出。

- Linux／Python 3.12，固定 SDK `pyorbbecsdk2==2.1.3`、NumPy 2.2.6、OpenCV contrib 4.11.0.86。環境在 `.agents/runs/foundation/camera_calibration/venv/`。
- SDK 使用 `--no-deps`，所選官方範例只需已裝依賴；避免另一份 opencv-python 覆蓋 contrib 的 ChArUco。
- 只接一台待驗相機，關閉其他 Viewer，使用可靠 USB3 線材。官方範例預設第一台裝置，沒有本工程舊版的 `--serial`。
- Linux 權限問題按[官方環境說明](https://github.com/orbbec/pyorbbecsdk/tree/68cdfa181e09f0ddd0f21b77dff29cb3e7a0f0d2/scripts/env_setup)處理，不由本工具修改系統權限。
- 官方 parameters 無裝置時也可能退出 0：必須看到裝置與參數才算完成連線檢查。

## 4. 標定板：已確認與仍需量測的部分

預設設定是 [configs/camera_board.json](../../configs/camera_board.json)，根據使用者的 Calibration Target.pdf：

| 項目 | 設定 |
| --- | --- |
| 圖案 | ChArUco，PDF 橫放時 11 格寬 × 8 格高 |
| 格邊長 | 名義 15 mm |
| Marker 邊長 | 名義 11 mm，量含黑色邊框的完整標記寬度 |
| 棋盤區域 | 名義 165 × 120 mm，不含頁面留白 |
| PDF 頁面 | 約 200 × 150 mm |
| 內角點 | 10 × 7＝70 個 |
| Marker | 44 個，ID 0–43，按 OpenCV legacy board 的排列 |
| 相容字典 | `DICT_4X4_50` |
| OpenCV 排列 | **`legacy_pattern=true`** |

本輪把 PDF 渲染後實際偵測：44 個 marker、70 個 ChArUco 角點；非 legacy 排列未取得 ChArUco 角點。4×4 字典有共用的前綴圖碼，所以可以證明 DICT_4X4_50 與此圖相容，不能倒推原網站當時一定選了 `_50`。

**你仍要做的實物檢查：**

1. 100%／實際大小列印，不選「符合紙張」。貼在平整、硬質、霧面底板。
2. 量水平及垂直 10 格／多格總長，再除以格數。水平 10 格名義應是 150 mm；垂直不足 10 格就量可用格數。
3. 量 marker 外框邊長，確認整體等比例、紙張不翹曲。若 X/Y 縮放不同，重新印製。
4. `init` 輸入**實測值**。程式不把 PDF 名義尺寸冒充量測，`--confirm-measured` 是對該尺寸的明確聲明。

小板在一兩公尺距離可能角點太少。先看預覽能否穩定辨識；需要時等比例放大整張板，格長與 marker 長度一起換成新實測值。

可以另外檢查板圖或照片：

```bash
bash scripts/camera_calibration/run.sh board-check \
  --image /你的路徑/標定板照片.png \
  --output .agents/runs/foundation/camera_calibration/board_detected.png
```

輸出標註角點的單張預覽，只驗證圖案是否相容，不證明印刷尺度或板子平整度。


## 5. 建立資料集、錄影和回放

以下在同一終端操作；`SERIAL_SESSION` 換成實際序號和本次日期／用途：

```bash
data_root="$PWD/.agents/runs/foundation/camera_calibration"
session="$data_root/d335l_SERIAL_SESSION"
mkdir -p "$data_root/recordings"
# 下例15和11只是名義尺寸，務必改成實測值
bash scripts/camera_calibration/run.sh init --session "$session"   --square-mm 15 --marker-mm 11 --confirm-measured
bash scripts/camera_calibration/run.sh official record
```

錄影程式會要求輸入輸出檔名。輸入完整絕對路徑，例如上述 `recordings` 裡的 `validation.bag`；不要原樣輸入 shell 變數，官方 prompt 不會展開。先試錄約 10 秒，停止後確認同名 `.bag` 和 `.json` 都存在且非空。

- GUI：**S 暫停／繼續，Q 或 Esc 停止**。S 並非舊工具的「存一張」。
- SSH 無桌面：`bash scripts/camera_calibration/run.sh official record --no-gui`，Ctrl+C 正常結束並保存。不要直接 kill。
- 官方 recorder 自動選可用串流的預設模式，沒有 `--profile`；也可能錄到 IMU，本工程不分析它。查看並記錄實際設定。
- 若需特定模式，用官方 OrbbecViewer 配置／錄製，再確認 BAG 與 sidecar 能由此固定版本回放。**Viewer 產物相容性尚未實測**，不可假設只要副檔名相同就能匯入。
- 串流、韌體、SDK 或幾何處理不同，另開 session。原始幾何驗證用未額外對齊／矯正的輸出；不要用預覽偽彩圖當深度資料。
- 正式板子錄影約 50–60 秒，預留前 10 秒穩定曝光；每 2 秒換一個清晰姿態並稍停。覆蓋畫面中心、邊角、不同遠近與水平／垂直傾斜，避免過曝、模糊和只拍正面。
- validation 必須獨立拍攝；若之後需要 fit，另錄一段，不把同一錄影切成兩份充當獨立驗證。先短錄確認，避免累積大量 BAG。

```bash
bash scripts/camera_calibration/run.sh official playback
```

回放同樣提示輸入 BAG 絕對路徑。官方 GUI 可能套用建議深度濾波；下一步匯入保存原始回放 callback，不重複套預覽濾波，因此深度圖不一定與預覽完全相同。

## 6. 匯入及原廠驗證

```bash
bash scripts/camera_calibration/run.sh import-bag   --session "$session" --bag "$data_root/recordings/validation.bag"   --stream rgb --stream depth --split validation   --start-s 10 --interval 2 --count 20
bash scripts/camera_calibration/run.sh factory-check --session "$session" --stream rgb
bash scripts/camera_calibration/run.sh alignment-check --session "$session"
```

匯入前確認錄影確實含所選串流；需要 IR 時加 `--stream ir_left --stream ir_right`，同一 session 每次選擇一致。匯入會：

- 先驗證官方 sidecar 可載入，再用官方回放設定；序號、韌體、SDK、profile、可見矯正模式與出廠參數不一致會拒絕混入。
- 按第一組完整影格的裝置時間戳開始計時，跳過 `start-s` 後按間隔取樣。配對預設時間差不超過 10 ms；這只是資料篩選，不能證明硬體同步。
- 將 RGB 存 PNG、IR／depth 保留原始位元。深度尺度按每幀 SDK 的 mm/unit 轉成 m/unit。彩色重用官方 converter；目前接受 RGB、MJPG、YUYV、UYVY。上游 BGR 轉換方向有歧義，暫拒絕 BGR，不修改原檔。
- 將 BAG／sidecar SHA256、來源版本與設定寫進 manifest；逐幀資料寫 CSV。回放不是採集時刻，因此 host timestamp 留空。
- 成功數達標為 IMPORTED；錄影不足為 PARTIAL；丟幀／時間差拒絕為 REVIEW_REQUIRED；錯誤為 ERROR。這些都不等於標定精度 PASS。已匯入 BAG 不重複匯入，出錯保留檔案，先查 manifest，不能直接重跑覆蓋。
- 並非所有裝置處理都能從 sidecar 完整辨識。拍攝者仍需記錄 preset／對齊／矯正設定並保持一致；發現不明處理時另錄乾淨資料。

資料結構：

```text
d335l_SERIAL_SESSION/
├── manifest.json                 # 裝置、模式、錄影來源與批次狀態
├── board.json                    # 實測板尺寸與圖案
├── factory_calibration.json      # 錄影profile的出廠參數，非重新估計
├── frames.csv                    # 配對ID、時間戳、尺度、split
├── rgb/  ir_left/  ir_right/  depth/
└── results/
    ├── rgb_factory_validation.json
    ├── depth_rgb_factory_validation.json
    └── depth_validation.json
```

`factory-check` 固定原廠 K／畸變，每張只解板姿態，再量投影誤差。預設至少 5 張 validation，RMS ≤0.5 px，另有覆蓋／姿態跨度檢查；建議採 20 張。這是同張板影像的幾何一致性，不能單獨證明三維絕對精度。

`alignment-check` 用原廠 RGB 內參、depth 內參與 depth→RGB 外參，比較板平面和深度點；預設平面 P95 誤差 ≤10 mm。RGB／depth 必須配對，板在深度圖中需有足夠有效像素。詳細逐張失敗原因留在 results。

畸變按 SDK 的 model 解讀，不能一律抄成五個係數。支援 NONE、BROWN_CONRADY、BROWN_CONRADY_K6；inverse／modified 等未支援型號直接拒絕。K6 的 OpenCV 順序為 k1,k2,p1,p2,k3,k4,k5,k6。

## 7. 深度：用獨立量距驗，不拿自身預測當真值

找平整、霧面、正對 depth 光軸的靜止平面。獨立量得**從 depth 光學原點到平面的軸向 Z**；外殼前緣距離、斜距及 Viewer 顯示距離不能直接當真值。光學原點或量距不確定度無法確認，就記 NOT_RUN。

每個距離另錄一個 BAG，整段保持不動，取不含邊界／空洞的 ROI。建議涵蓋預計使用的近、中、遠距離。下面數值只示範參數，換成實測距離和有效 ROI；沿用與前段相同串流：

```bash
bash scripts/camera_calibration/run.sh import-bag   --session "$session" --bag "$data_root/recordings/depth_validation_080.bag"   --stream rgb --stream depth --split validation --start-s 10 --interval 1 --count 10   --reference-z-m 0.80 --depth-roi 300 200 100 100 --frontoparallel
bash scripts/camera_calibration/run.sh depth-check --session "$session"
```

報告包括 bias、有效像素比例、空間離散及多幀中位數的時間波動。預設 validation 每幀 |bias|≤10 mm、有效率≥90%；門檻是工程診斷值，不是官方規格或動態乒乓球保證。

只有額外提供至少三個不同距離的 fit 批次及獨立 validation，才會估計離線 scale/offset 候選；不寫回相機，也不自動套用。原始結果必須保留。

## 8. 原廠未達標才重新估計

先檢查正式解析度與模型是否匹配、板尺寸／平整度、曝光模糊、同步、D2C／矯正模式。若有持續系統性誤差，再另錄 fit BAG 並用相同模式匯入 `--split fit`：

```bash
bash scripts/camera_calibration/run.sh intrinsic --session "$session" --stream rgb
# 有同步IR資料、且兩個stream均已求過內參時才跑：
bash scripts/camera_calibration/run.sh intrinsic --session "$session" --stream ir_left
bash scripts/camera_calibration/run.sh stereo --session "$session" --first rgb --second ir_left
# 用重新估計的RGB內參比較，另存報告，不蓋原廠報告：
bash scripts/camera_calibration/run.sh alignment-check --session "$session" --model-source estimated
```

內參至少 15 張 fit＋5 張獨立 validation；OpenCV calibrateCamera 只擬合 fit，validation 解板姿態後評估。stereo 固定兩側內參，只求串流間 R／t，並在留出配對上測投影誤差。輸出 `p_to=R@p_from+t_m`，平移單位公尺。這是**相機內部串流外參**，不是相機到機械臂的手眼外參。

`board-pose --session ... --stream rgb --frame-id ...` 使用自行估計的內參輸出相機↔板變換；不是世界／球桌／機械臂外參。若只做原廠驗證，無需執行這一步。新增影像會改 CSV 雜湊，舊求解結果需重新產生。

## 9. 程式資料流與驗證範圍

`run.sh → cli.py → camera_official.run_example()` 只負責來源檢查與執行原檔。錄影交給官方；`camera_recording.import_bag()` 只開 PlaybackDevice，重用官方 sidecar／setup_camera／色彩轉換，再保存資料。沒有另一套 live Pipeline、曝光控制或雷射設定。

`camera_calibration.py` 保留必要離線演算法：偵測 ChArUco → 分離 fit／validation → 原廠一致性或可選重新估計 → 深度／彩深驗證 → JSON。BAG 雜湊串流計算，不一次載入整段影片。原檔、環境、大圖與影片留在被忽略的 runs，只提交程式、設定與說明。

2026-11-19 驗證：

- PASS：11 項離線測試，涵蓋已知 K／外參方向、留出驗證、模型拒絕、深度單位、官方 RGB 轉換及原始 uint16、模擬回放的 PNG／CSV 匯入與重複錄影拒絕。模擬 transport 不是實際 BAG 解碼測試。
- PASS：官方原檔與 LICENSE 的 SHA256、安裝與 CLI；實際啟動官方 info／parameters，收到「無裝置」。parameters 返回 0 仍不能算硬體通過。
- NOT_RUN：Gemini 335L 真實錄影／sidecar／BAG 匯入、Viewer 產物相容性、原廠與重標精度、彩深及量距。第一步應先以短錄完成這條鏈路，成功後才大量採集。
- 本輪測試曾因假影格使用 bytes、與 SDK 回傳 NumPy buffer 不一致而失敗；修正測試替身以符合 SDK 介面，官方原檔未改。

```bash
TMPDIR="$PWD/.agents/runs/foundation/camera_calibration/tmp" PYTHONPATH=src   .agents/runs/foundation/camera_calibration/venv/bin/python   -m unittest discover -s tests -p test_camera_calibration.py -v
```
