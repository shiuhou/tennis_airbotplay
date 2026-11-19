# Camera calibration｜Gemini 335L 操作與原理

更新：2026-11-19。型號依使用者更正為 **Orbbec Gemini 335L**。工具已建立，板圖與離線幾何測試已通過；本機 SDK 枚舉到 0 台 Orbbec，因此**實機採集、內參精度、串流外參精度及深度驗證仍為 NOT_RUN**。以下可在接好相機的主機依序執行；未執行的結果不預填成成功。

本工具的「外參」包括 RGB／左右 IR 等**相機串流之間**的變換、**相機相對標定板**的單張姿態，以及出廠 Depth→RGB 變換的核對。沒有相機到機械臂的手眼標定、機械臂控制或韌體標定參數寫入。M6 學習／訓練也不在本工具範圍。

**交接範圍再次確認：**直接使用Orbbec SDK v2與OpenCV，不需要ROS 2。先讀取這台相機、這個串流模式的出廠參數；官方讀參數範例不等於完整精度驗證。畸變模型與係數順序需核對，不能一律當成五係數Brown，也不能對已校正影像重複去畸變。IMU標定／相機–IMU同步暫不列入任務，只有決定使用IMU融合後再安排。相機安裝方式尚未確認，不預設Eye-in-Hand；裝在手臂上或固定在外部，後續的外參求解方式不同。

## 1. 同學接手先讀：先驗證原廠，必要時才重標

**這次任務的交付是「可追溯的出廠參數與實測驗證」，不是一定要重新算一組內參。**依序做A–D；原廠結果有明確問題才進E。相機／球桌／機械臂安裝外參與手眼留待另行安排。

| 階段 | 要做的事與入口 | 完成後交回 | 不通過時 |
| --- | --- | --- | --- |
| A 接通與選模式 | §2安裝、§4 `inspect`；確認335L、USB、序號與正式RGB／depth模式 | device_inventory.json、實際使用的profile | 沒裝置先處理連線／權限；不要開始求解 |
| B 原廠基線與試存 | §3量板、§5 `init`；§6 `capture`先存1組，split用validation；RGB與depth同時開 | manifest.json、board.json、factory_calibration.json、frames.csv及一組原始PNG | 核對尺寸、深度尺度、模型及外參方向；未知欄位保留未知 |
| C 原廠幾何驗證 | 再採至少15–20組不同位置／傾角的validation板圖；執行下方 `factory-check` 與§9 `alignment-check` | rgb_factory_validation.json、depth_rgb_factory_validation.json、失敗frame_id與原因 | 先排除板尺度／反光／模糊／profile錯配，不能直接判定出廠參數壞了 |
| D 深度獨立驗證 | §10量幾個工作距離、各約30張靜止平面圖；先用validation標記並執行 `depth-check` | depth_validation.json、參考距離量法與光學原點依據、各距離偏差／有效率 | 距離基準不明就標明無法驗證絕對精度，不用相機自己的讀值作真值 |
| E 條件式重標 | 只有C／D顯示可重現問題，且排除採集問題後，才另採fit資料並做§7內參、§8串流外參或深度修正候選 | 原廠／新結果的同條件比較、是否採用及理由 | 不因重標分數較低就覆寫出廠資料，也不寫入韌體 |

完成A–D後即可先交接結果；不必等到E，也不必實作手眼。診斷PASS只代表該測試範圍，不等於高速球定位已通過。

**C階段命令（不會重估內參）：**

```bash
bash scripts/camera_calibration/run.sh factory-check --session "$SESSION" --stream rgb
bash scripts/camera_calibration/run.sh alignment-check --session "$SESSION" --model-source factory
```

如果使用IR追球，對實際採集的IR串流也執行factory-check。原廠畸變模型若不在支援清單中，先回報UNSUPPORTED／錯誤；不要把係數清零或靠重新標定掩蓋模型轉換缺口。

**同學要交回的最小成果：**

- 相機序號、韌體／SDK、串流設定與採集日期；板子的實測尺寸及是否平整。
- 完整session資料的位置或團隊可取得的儲存位置，包含原始影像、CSV、出廠參數與results；大型資料不進Git。
- 一段結果說明：原廠內參／彩深／深度各為PASS、REVIEW_REQUIRED或NOT_RUN，最差案例在哪裡，是否建議重標及原因。沒有量距基準要明寫。
- 保留出廠檔案和失敗案例。傳給團隊時不要只交K矩陣或成功截圖。

下面§2–6是共用操作；§7–8是**需要時才做**的重標及姿態分析；§9–10可直接使用出廠參數。命令中的序號、profile、尺寸與距離範例都要換成實際值。

## 2. 安裝一次：獨立於 Isaac Lab

在倉庫根目錄執行：

```bash
cd /home/dsa-cli/ceo_pj/airbot_tt
bash scripts/camera_calibration/setup.sh
bash scripts/camera_calibration/run.sh --help
```

- 已在 Linux／Python 3.12 驗證安裝。環境放在本地 `.agents/runs/foundation/camera_calibration/venv/`。
- 固定 NumPy 2.2.6、OpenCV contrib 4.11.0.86、pyorbbecsdk2 2.1.3。採集不需要啟動 Isaac Sim 或機械臂。
- OpenCV contrib 提供 ChArUco。不要再在此環境安裝另一份 `opencv-python`，兩者會覆蓋相同的 `cv2`。安裝腳本對 SDK 使用 `--no-deps`，因為本工具不使用官方範例的 AV／pygame／pynput 等套件。
- 臨時檔、SDK log 和環境放本地 runs；本機 `/` 分割區曾滿，但本次 `/home` 有空間，安裝腳本將 TMPDIR 指向本地 runs。

相機接 USB3、確保供電與線材可靠，先關閉其他正在使用相機的 Viewer。桌面預覽要有 DISPLAY；SSH 無桌面可用 `--headless --count N`。Linux 若出現存取權限錯誤，按[官方 Linux udev 說明](https://github.com/orbbec/pyorbbecsdk/tree/68cdfa181e09f0ddd0f21b77dff29cb3e7a0f0d2/scripts/env_setup)由管理者設定後重插相機；工具不自動改系統權限。

## 3. 標定板：已確認與仍需量測的部分

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

## 4. 找到相機並選正式模式

### 官方最小範例：先確認能讀參數

官方已有[04_camera_calibration.py](https://github.com/orbbec/pyorbbecsdk/blob/68cdfa181e09f0ddd0f21b77dff29cb3e7a0f0d2/examples/beginner/04_camera_calibration.py)。如果先在官方pyorbbecsdk倉庫操作，可依其環境安裝說明執行：

```bash
# 在官方 pyorbbecsdk checkout 根目錄，不是在 airbot_tt 根目錄
python examples/beginner/04_camera_calibration.py
```

它啟動預設depth／color串流，收到影像後呼叫`pipeline.get_camera_param()`，印出內參、畸變資訊與Depth→Color外參。可用作最小硬體連線檢查；預設profile不一定是本專案正式profile，仍須確認尺寸與模式。這個範例路徑不在本倉庫，不能在airbot_tt直接照打該相對路徑。

**本專案使用狀態：**已閱讀固定來源範例並核對SDK接口，沒有直接執行官方範例、沒有複製整份範例或取得實機數值。本機capture改由實際收到影像的stream profile讀`get_intrinsic()`／`get_distortion()`／`get_extrinsic_to()`，以涵蓋RGB／左右IR／depth並保存對應模式。SDK匯入及無裝置枚舉已測，真機串流參數仍待同學確認。若官方範例能讀、我們工具不能讀，保留兩邊的模式與錯誤資訊，先修接口問題，不重新標定硬體。

### 本專案列舉入口

```bash
bash scripts/camera_calibration/run.sh inspect \
  --output .agents/runs/foundation/camera_calibration/device_inventory.json
```

輸出包含實際 `model`、`serial`、`firmware` 和各串流的 width／height／fps／format。若顯示 `NO_DEVICE`，先處理 USB／權限，不要繼續假裝已採集。請選正式使用的模式；下面範例中的解析度與格式**不是已驗證的硬體支援清單**，必須用 inspect 的結果替換。

串流名稱：`rgb`、`ir_left`、`ir_right`、`depth`。格式語法：

```text
rgb=1280x720@30:MJPG
ir_left=1280x800@30:Y8
depth=1280x800@30:Y16
```

保存支援 RGB、BGR、MJPG、YUYV、UYVY、Y8、Y16；深度只接受 Y16。串流個別支援不保證全部能同時啟用；若 USB 頻寬或裝置限制導致啟動失敗，先測一個／兩個串流，再選相容組合。不同組合建立不同 session。

工具不自動開啟對齊、去畸變或軟體深度濾波；會記錄可讀的硬體 IR rectify、畸變開關與 preset。這裡「原始」指未經本工具縮放／對齊的輸出，**不保證它是未經韌體校正的 sensor raw**。硬體屬性讀不到會保留 null，這個未知項需在結果解讀時保留。

## 5. 建立 session：不要直接使用空白範本求解

先設定你的序號及資料目錄：

```bash
CAMERA_SERIAL='替換為inspect列出的序號'
SESSION="$PWD/.agents/runs/foundation/camera_calibration/gemini335l_${CAMERA_SERIAL}_intrinsics_01"
```

只有實測確實為 15 mm／11 mm 時，才使用以下數字，否則換成你量到的值：

```bash
bash scripts/camera_calibration/run.sh init \
  --session "$SESSION" --square-mm 15 --marker-mm 11 --confirm-measured
```

已有 session 會拒絕覆蓋。此前的 `session_template/` 只是可讀範本，現在由 `init` 建立完整且匹配程式的 schema；不必手動複製範本、建立 CSV 或填空參數。

## 6. 採集內參／雙目外參影像

### 6.1 先採 fit，再另採 validation

以下先示範 RGB。把 profile 換成 inspect 列出的模式：

```bash
bash scripts/camera_calibration/run.sh capture \
  --session "$SESSION" --serial "$CAMERA_SERIAL" \
  --profile rgb=1280x720@30:MJPG --split fit
```

預覽中：

- **S**：保存目前一組影像，原始解析度不變，預覽上的角點與文字不燒進資料。
- **Q／Esc**：結束本次採集。已保存的影像保留。
- 畫面顯示串流、fit／validation、已存張數及同組時間戳差。
- 預設暖機讀流 10 秒；精密採集前讓裝置在正式環境中充分穩定，10 秒不代表已證明熱穩定。

建議先拍約 30–40 組 fit，再另拍 10–20 組 validation。至少需要 15 組有效 fit、5 組有效 validation 才會求解：

```bash
bash scripts/camera_calibration/run.sh capture \
  --session "$SESSION" --serial "$CAMERA_SERIAL" \
  --profile rgb=1280x720@30:MJPG --split validation
```

每次把板移到不同位置／角度，**停穩後按 S**。覆蓋中央、四邊、四角，改變距離與上下／左右傾角約 15–45°；不要一直正面居中，也不要用近乎相同的連拍湊張數。驗證集另拍新姿態，不能拿已用來調參的照片當獨立驗證。

### 6.2 若還要標定左右 IR／RGB↔IR

必須在**同一組採集**中同時開啟要配對的串流，並保持板子靜止。例如：

```bash
bash scripts/camera_calibration/run.sh capture \
  --session "$SESSION" --serial "$CAMERA_SERIAL" \
  --profile rgb=1280x720@30:MJPG \
  --profile ir_left=1280x800@30:Y8 \
  --profile ir_right=1280x800@30:Y8 \
  --split fit --laser off
```

這應在選定三串流的新 session 第一次採集就使用；不能把先前只拍 RGB 的 session 途中改成三串流。下一輪 validation 使用相同 profile 組合。

`--laser off` 是可選的投射器設定，供 IR 板圖被散斑干擾時使用；結束後恢復原設定。裝置若不支援此屬性會明確報錯，不假裝已關閉。深度測試用正式工作設定，例如 `--laser on` 或預設 `keep`。

相同 frameset 不代表曝光時刻完全一致。工具保存各串流裝置時間戳，預設差距超過 10 ms 就拒絕保存；這只是在靜態板下的配對檢查，並非高速球同步驗收。若無法穩定達到，先檢查模式與同步設定，不為了通過而任意放寬門檻。

### 6.3 無桌面模式

```bash
bash scripts/camera_calibration/run.sh capture \
  --session "$SESSION" --serial "$CAMERA_SERIAL" \
  --profile rgb=1280x720@30:MJPG --split fit \
  --headless --count 30 --interval 3
```

這會每 3 秒存一組，共 30 組。需要安排板子在存圖時靜止；內參採集優先使用按鍵模式。每次最多 500 組，不無限錄影。

## 7. 必要時才重新求內參與畸變

完成 fit 和 validation 採集後：

```bash
bash scripts/camera_calibration/run.sh intrinsic --session "$SESSION" --stream rgb
# 有採集左右IR時再執行：
bash scripts/camera_calibration/run.sh intrinsic --session "$SESSION" --stream ir_left
bash scripts/camera_calibration/run.sh intrinsic --session "$SESSION" --stream ir_right
```

輸出以串流分檔，例如：

```text
results/rgb_calibration.json
results/rgb_validation.json
```

核心流程：偵測 ChArUco 角點 → 依 ID 取得板上 `(X,Y,0)` → `cv2.calibrateCamera` 共同估計 K、5係數畸變與各張板姿態。模型是 OpenCV forward Brown：`[k1,k2,p1,p2,k3]`。固定這組 K、畸變後，在 validation 照片只求板姿態，再量投影誤差，沒有重新用 validation 擬合內參。

報告包括：

- K 中的 fx／fy／cx／cy、畸變係數及影像尺寸。
- fit RMS、validation RMS、角點 P95 誤差與逐張結果。
- 畫面覆蓋比例、板面法向角度跨度、偵測失敗與原因。
- 出廠參數在相同 validation 上的比較。僅明確支援 NONE、BROWN_CONRADY、BROWN_CONRADY_K6；inverse／modified／其他模型標 UNSUPPORTED，不錯套 OpenCV 公式。

預設診斷門檻：validation RMS ≤0.5 px、fit 角點水平與垂直覆蓋各≥60%、板法向跨度≥15°，且沒有偵測剔除。通過稱為 **PASS（此數值／覆蓋檢查）**；否則 **REVIEW_REQUIRED**，仍保留結果供分析。這不是 Orbbec 官方精度認證，也不表示已達到乒乓球三維定位要求。

如果漏偵測，先看報告的 frame_id，檢查模糊、反光、距離及圖案設定；不要只為降低平均誤差刪掉邊緣影像。原始影像都保留，工具不自動淘汰難例或反覆調參。相同影像若同時出現在 fit 與 validation，會拒絕計算。

## 8. 必要時才重新求相機內部串流外參

先完成兩個串流各自的內參，再執行：

```bash
bash scripts/camera_calibration/run.sh stereo \
  --session "$SESSION" --first ir_left --second ir_right

bash scripts/camera_calibration/run.sh stereo \
  --session "$SESSION" --first ir_left --second rgb
```

輸出 `results/ir_left_to_ir_right.json` 或 `results/ir_left_to_rgb.json`。定義固定為：

```text
p_to = R @ p_from + t_m
```

例如 `ir_left_to_rgb` 表示把左 IR 座標中的三維點轉到 RGB；平移以公尺保存，SDK 的毫米平移在匯出時換成公尺。逆變換是 `R.T` 與 `-R.T @ t`，不是只把 t 改負號。

程式以 frame_id 配對、檢查時間戳與共同角點，使用 `cv2.stereoCalibrate(..., CALIB_FIX_INTRINSIC)`。在 validation 中只由 first 相機求板姿態，套用外參後投影到 second，避免在第二張影像重新求姿態而掩蓋錯誤外參。可取得出廠外參時，使用同一組影像與內參另列對照。

預設至少15組fit／5組validation，共同角點每組至少8個且不能共線。跨相機驗證 RMS≤1 px、無剔除且兩組內參檢查均PASS才標PASS。不同解析度允許，但各串流自己的尺寸必須固定。

**深度圖不是棋盤強度影像，不能拿它當 RGB／IR 圖直接跑 stereo。**本工具讀取 Depth↔各串流的出廠變換，做下一節的一致性檢查；不重做 Orbbec 韌體內部完整深度標定。

### 8.1 相機相對標定板的外參

從 frames.csv 選一個已保存的 frame_id（例如採集畫面顯示的 `批次ID_0000`）：

```bash
bash scripts/camera_calibration/run.sh board-pose \
  --session "$SESSION" --stream rgb --frame-id 替換為實際frame_id
```

輸出 `results/rgb_board_pose_<影像檔名>.json`，包含 `T_camera_from_board` 與 `T_board_from_camera` 兩個4×4矩陣、角點數與重投影誤差。板座標原點是OpenCV板的左上角，X向右、Y向下、Z按右手系，單位公尺；相機光學座標X向右、Y向下、Z向前。

這是已校正影像上的平面PnP姿態，仍可能受小板／正視姿態歧義與角點誤差影響，需多姿態或已知幾何交叉檢查。只有當板子在世界／球桌中的安裝變換也已知且固定，才可進一步求相機到世界／球桌的外參；目前不替你假定這個變換，也不操作機械臂。

## 9. 檢查 Depth→RGB 對齊的一致性

建立一個固定 **RGB＋depth** profile 的 session，在 fit／validation 採集時同時存兩個串流：

```bash
bash scripts/camera_calibration/run.sh capture \
  --session "$SESSION" --serial "$CAMERA_SERIAL" \
  --profile rgb=1280x720@30:MJPG --profile depth=1280x800@30:Y16 \
  --split validation --laser on
```

第一輪直接採 validation，使用出廠 RGB 內參，無須先執行 intrinsic：

```bash
bash scripts/camera_calibration/run.sh factory-check --session "$SESSION" --stream rgb
bash scripts/camera_calibration/run.sh alignment-check --session "$SESSION" --model-source factory
```

輸出 `results/depth_rgb_factory_validation.json`。若後續完成重標，可指定 `--model-source estimated` 另存 `depth_rgb_estimated_validation.json` 作對照，不覆蓋原廠基線。它由 RGB 角點估計板平面，將深度像素反投影成射線、使用出廠外參變換到 RGB，再檢查板內有效深度點到該平面的距離。ROI 由射線與預測平面的交點決定，不以量得的深度是否接近預期值來挑好看的點；板邊內縮10 mm以減少邊緣混合像素。

預設至少5個有效 validation 板姿態，各組平面絕對誤差 P95≤10 mm、有效比例≥90%，且所選RGB模型的驗證PASS／無剔除才通過。拍攝多種傾角，避免只用正對相機的同一平面。這是**聯合一致性診斷**：錯誤可能來自板尺寸、RGB姿態、深度或外參；平面檢查不能單獨證明所有外參自由度精確，也不是獨立的絕對距離量測。

如果 SDK 的深度畸變模型不受支援，會拒絕套用不明公式；需另核對 SDK 投影接口後擴充，不自動把畸變清零。

## 10. 深度距離驗證與離線修正候選

### 10.1 準備獨立距離參考

用平整、霧面、尺寸足夠的平面，盡量與深度相機成像平面平行，固定不動。在深度畫面選一個完全落在平面內的矩形 ROI，避開邊緣、反光和空洞。

`reference_z_m` 是**從深度光學原點沿相機 Z 軸到平面的距離**。它不是斜距、不是從外殼前緣量起的距離，也不能拿待驗證的深度值本身當真值。若無法確定光學原點或獨立量距，只能先做重複性／平面一致性，不能把此結果宣稱為絕對精度標定。

建立 depth-only session；板尺寸仍需填寫以統一session資料，但深度距離檢查本身不使用 ChArUco 圖案。下面 0.5 m 與 ROI 都只是示例，必須替換成實際量測與畫面座標：

```bash
DEPTH_SESSION="$PWD/.agents/runs/foundation/camera_calibration/gemini335l_${CAMERA_SERIAL}_depth_01"
bash scripts/camera_calibration/run.sh init \
  --session "$DEPTH_SESSION" --square-mm 15 --marker-mm 11 --confirm-measured

bash scripts/camera_calibration/run.sh capture \
  --session "$DEPTH_SESSION" --serial "$CAMERA_SERIAL" \
  --profile depth=1280x800@30:Y16 --split validation --laser on \
  --reference-z-m 0.5 --depth-roi 500 300 200 150 --frontoparallel \
  --count 30 --interval 0.2
```

ROI 是**原始深度解析度**下的 `x y width height`，不是縮小預覽的座標。`--frontoparallel` 是你確認平面平行的聲明，程式不會由這個旗標自動證明平面已對正。

### 10.2 多距離採集，另留驗證距離

在同一 depth session、相同 profile 下，移動平面到數個覆蓋工作範圍的距離，每個距離重新執行 capture，填入新的實測 reference。每次 capture 都是獨立 batch；同一 batch 中平面不能移動。

- **fit**：僅在需要擬合修正候選時才採集至少3個不同參考距離，例如實際可量測的近、中、遠位置。
- **validation**：另量一個或多個未用來擬合的距離，把 `--split` 改為 `validation`。
- 每個距離約30張可先看重複性，保持相機、平面與工作模式穩定。

```bash
bash scripts/camera_calibration/run.sh depth-check --session "$DEPTH_SESSION"
```

`results/depth_validation.json` 會列出：

- 每張 ROI 的中位深度、對參考 Z 的偏差、有效像素比例。
- ROI 空間標準差／MAD／P95絕對誤差；這些會受平面傾斜或材質影響，不全等於感測器雜訊。
- 每個 batch 的 ROI 中位深度時間標準差；它是區域中位數的穩定性，不是逐像素時間雜訊。
- 至少3個不同 fit 距離、量得深度跨度>5 cm且有validation時，估計 `z_corrected = scale*z_raw + offset`，並另列驗證距離修正前後的偏差。

預設只在**原始未修正的 validation** 偏差≤10 mm、有效比例≥90%且無資料拒絕時標PASS。線性修正永遠標 `CANDIDATE_ONLY`，不自動套用、不寫相機、不外推到未採集的距離，也不以修正後通過掩蓋原始失敗。光學原點偏移、板面傾斜也可能被擬合成 offset，必須先排除量測方法錯誤。

## 11. 資料、結果與重跑規則

```text
.agents/runs/foundation/camera_calibration/
├── venv/                             # 本地工具環境
├── session_template/                 # 可讀空白範本；正式採集使用 init
├── sdk_logs/
└── gemini335l_<serial>_<session>/
    ├── manifest.json                 # 裝置、profile、硬體設定、各次採集狀態
    ├── board.json                    # 實測尺寸、字典、IDs與legacy排列
    ├── factory_calibration.json       # 實際串流出廠內外參，READ_NOT_VALIDATED
    ├── frames.csv                    # 每個stream/frame一列，含batch與split
    ├── rgb/
    ├── ir_left/
    ├── ir_right/
    ├── depth/                        # 原始uint16，尺度逐張記錄
    └── results/
        ├── rgb_calibration.json       # 其他串流同樣命名
        ├── rgb_factory_validation.json # 第一輪原廠檢查
        ├── rgb_validation.json
        ├── ir_left_to_ir_right.json
        ├── depth_rgb_factory_validation.json
        └── depth_validation.json
```

- 所有長度／平移結果用公尺；像素誤差用 px，角度跨度用 degree。
- 裝置時間戳是 µs，主機接收時間用 monotonic ns；兩者不同時鐘，不直接相減當延遲。
- 曝光／gain metadata 按 SDK 原始值保存，未確認單位時不冒稱 µs／dB；讀不到留空。
- MJPG 來源本身有壓縮，另存PNG只避免再次壓縮，不會恢復已失去的資訊。16-bit IR原樣存，只有角點偵測／預覽使用亮度正規化。
- 序號、韌體、SDK、profile或幾何相關硬體設定改變，建立新session。工具會拒絕可檢出的混用。
- 補拍後重新執行所有相關求解。結果記錄 board／frames 的 SHA256，stereo及alignment拒絕使用這兩者已改變的舊內參結果。
- 結果檔使用固定名稱重算更新，原始影像不覆蓋。不要讓多個程序同時採集或求解同一session。
- 失敗返回非零退出碼，並保留已保存影像與錯誤狀態。若磁碟寫入中斷，先查看 manifest 的 partial_files 與CSV，不能直接認為最後一組完整。
- 原始資料、環境和log不進Git。程式、固定依賴、板圖設定、測試及操作文件可追蹤。

## 12. 程式資料流與驗證證據

- [cli.py](../../scripts/camera_calibration/cli.py)：命令、參數、狀態與退出碼。
- [camera_device.py](../../src/airbot_tt/camera_device.py)：Orbbec枚舉、profile選擇、原始幀解碼、出廠參數匯出、預覽與CSV採集。不匯入機械臂控制程式。
- [camera_calibration.py](../../src/airbot_tt/camera_calibration.py)：板設定、角點、內參、固定內參雙目外參、跨相機獨立驗證、深度平面一致性與距離誤差。
- [test_camera_calibration.py](../../tests/test_camera_calibration.py)：已知幾何生成影像，經真正角點偵測和求解驗證；另測深度尺度、變換方向、模型係數順序、時間戳拒絕及原始影像解碼。

重跑離線測試：

```bash
TMPDIR="$PWD/.agents/runs/foundation/camera_calibration/tmp" \
PYTHONPATH=src .agents/runs/foundation/camera_calibration/venv/bin/python \
  -m unittest discover -s tests -p test_camera_calibration.py -v
```

離線測試與PDF圖案核對是PASS；實機為NOT_RUN。相機SDK API依固定官方來源核對，當前僅成功驗證匯入及無裝置枚舉，不能把這當作真機串流採集通過。連上Gemini 335L後，先做一張試存並核對PNG尺寸／深度尺度，再開始正式多姿態採集。

來源：[Orbbec SDK Python 固定來源](https://github.com/orbbec/pyorbbecsdk/tree/68cdfa181e09f0ddd0f21b77dff29cb3e7a0f0d2)、[OpenCV ChArUco calibration](https://docs.opencv.org/4.11.0/da/d13/tutorial_aruco_calibration.html)。官方範例用於核對接口；本機測試範圍與來源閱讀範圍分開記錄。
