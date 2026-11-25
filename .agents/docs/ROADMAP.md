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
| M0 工程基線重驗 | 2026-11-16 | PASS | 裸臂與固定球拍完整回歸通過；本次遠端核對確認個人及組織 main 均為6809f7c，原始歷史副本已存在 |
| M1 PACE 原始實作核查 | 2026-11-16 | PASS（來源核查） | 核對論文、程式版本與授權，整理阻力、預測、獎勵和訓練架構；尚未重現 PACE 訓練結果 |
| M2 獨立球物理參考 | 2026-11-16 | PASS（未標定模型） | 40 mm／2.7 g 球、阻力、桌／網／固定 CAD 拍面；三種步長的速度／接觸時間／落點收斂通過；尚無可動手臂擊球 |
| M3 單拍任務 | 2026-11-16 | PASS（限定預先就位接球） | 同場景手臂／工具／球，四組重現發球、事件／失敗／重置及限時可達性；三組可接觸、一組拒絕；尚無主動回球 |
| M4 球軌跡預測 | 2026-11-16 | PASS（零旋／限定接觸） | 480Hz下24例獨立PhysX事件比對；各類P95位置≤6.12mm、時間≤5.64ms；M3同場景碰拍前／實測碰拍後預測通過 |
| 實物鎖固 | 2026-11-16 | 使用者確認 | 真機已鎖固，仿真採理想固定安裝；不等於強度或真機擊球測試通過 |
| M5 規劃擊球 | 2026-11-16 | PASS（限定scripted基線） | 960Hz、三條名義預規劃軌跡；獨立驗證36/36合法且命中，各目標P95落點誤差24.30–25.56mm；空揮拍、移動拍收斂及原功能回歸PASS。在線重規劃未達時限 |
| M6 物理引導與predictor | 2026-11-19 | PASS（離線／窄分布） | 官方MLP；128訓練／32驗證／64新held-out發球，2,341窗口3D RMSE11.6mm；教師P95 5.6mm／2.2ms，時序與checkpoint通過；尚無PPO或可動臂RL |
| M7.1–M7.3 初始整合 | 2026-11-19 | PASS（限定工程鏈） | 16-env 生命週期／局部reset／terminal obs，48Hz動作與M5中央軌跡、官方PPO更新及checkpoint重載通過；不是已學會回球 |
| M7 A／B 小試 | 2026-11-19 | 完成；INCONCLUSIVE | 各3 seeds×6,144 transitions；A碰拍100%／100%／0%，B全0%，兩組合法回球及目標命中全0%；正式五組未完成 |
| Gemini 335L相機工具 | 2026-11-19 | PASS（離線工具）；實機NOT_RUN | 採集／出廠參數、ChArUco內參、串流外參、相機↔板姿態、彩深平面及深度距離檢查；9項離線測試與PDF70角點核對通過。SDK未偵測到裝置，不含手眼 |
| H0–H4 真機控制準備 | 2026-11-19 | 規劃；真機NOT_RUN | 官網版本／API核對完成；實機版本與拓撲未知，先主機盤點，再只讀、低速停止、幾何與時序驗證；獨立於M6–M9 |
| 真機擊球驗證 | 待定 | NOT_RUN | 螺紋規格、公差與強度未獨立驗證；真機操作須另有授權 |

## 研發紀錄

2026-11-19：依使用者閱讀回饋，重寫M6筆記為從零理解的流程：M5與M6關係→資料→物理教師→predictor監督學習→獎勵→接口→結果與限制。程式／參數／負結果移至可展開細節，保留既有證據與圖；notes索引補寫作偏好。僅文件修改及連結檢查，未改程式、重跑實驗或開始M7。

2026-11-19：完成[M6離線適配](../../doc/notes/m6_notes.md)。直接使用官方MLP，新增教師／引導reward、整serve窗口、same-step hook／actor frame、checkpoint及可重跑入口。首次batch128／20epoch誤差約54.5mm FAIL，保留證據；改batch32後validation10.5mm，另取新seed6104的64球驗收11.6mm PASS，沒有降低30mm門檻。平均點基線13.0mm；非跨反彈窗口物理估速法更準，不能宣稱MLP泛化優勢。64/64最高點距base 0.9819–1.0039m，超出TCP保守距離上界0.9443m；保留apex教師，不擅改可達點，列M7引導有效性風險。教師P95位置5.6mm、時間2.2ms；8項M6契約測試、Isaac環境48項（含M6與原碼查證）、相機獨立venv11項PASS。起初根分割區滿導致TMPDIR錯誤，改用runs暫存；容器缺Orbbec的2項測試改在既有相機環境驗證。未重跑完整手臂／M5仿真，未開始M7或操作真機；資料與模型留本地。

2026-11-19：整合既有M6–M9規劃與PACE原碼深讀，補齊本地詳細遷移計畫：模組映射、觀測／動作草案、教師與獎勵、資料隔離、五組消融、reset／timeout修正及逐工作包驗收。960Hz物理／48Hz策略為待測候選，未定案；[資料流筆記](../../doc/notes/pace_pipeline.md)補設計理解。僅文件檢查，沒有實作、訓練或真機操作，M6–M9仍NOT_RUN；完整活動計畫不發布。

2026-11-19：依使用者授權發布PACE深讀、原函式查證測試及H0–H4真機準備規劃；補PDF的USB-2控制與can0接口區分。沿用9項已通過的原函式隔離測試，發布前文件連結／來源雜湊／diff檢查PASS；未重跑完整仿真、未操作真機，M6移植仍未開始。

2026-11-19：將使用者Hardware Bring-up規劃納入既有PLAN與[notes](../../doc/notes/hardware_bringup.md)，保留H0–H4獨立狀態。核對官網V5.2版本、電流單位與急停權限；發現官方讀狀態範例含控制權與重力補償，H1不可原樣執行。兩個英文連結404改用有效中文來源。使用者回覆實際版本／拓撲未知；提供不連設備的主機盤點步驟，未寫硬體控制程式、未安裝／升級／運動。文件檢查PASS，硬體NOT_RUN。

2026-11-19：完成[PACE控制／學習資料流深讀](../../doc/notes/pace_pipeline.md)，固定fdb192f8與論文v5，追完teacher／predictor、405／490維觀測、21維動作、延遲PD、reward、GAE／PPO、reset與checkpoint。9項官方原函式隔離查證PASS（包含重現跨reset資料、no-noise初始化與未訓練checkpoint旗標問題）；原版完整仿真／訓練及AIRBOT移植仍NOT_RUN，未改上游或啟動M6實作。

2026-11-19：相機入口改為直接執行固定版本官方資訊／參數／模式／錄影／回放原檔；刪除自寫camera_device.py與inspect／capture，僅保留離線BAG匯入與獨立驗證。11項離線測試、來源與授權雜湊、安裝／CLI檢查PASS；官方info／parameters實際執行回報無裝置，真實BAG整合及硬體精度NOT_RUN。公開筆記重寫交接步驟與來源分工；以下採集工具紀錄為歷史版本。

2026-11-19：相機交接發布整理，明確SDK v2＋OpenCV不需ROS 2、出廠畸變模型不可一律套五係數、IMU與手眼不列本次範圍且不預設Eye-in-Hand。同步操作筆記與SPEC，按使用者授權準備推送雙遠端；實機驗證仍NOT_RUN。

2026-11-19：相機交接流程改為A接通／B出廠基線／C原廠幾何驗證／D深度獨立驗證／E有證據才重標，列出每階段交回資料與失敗處理。新增factory-check，alignment-check預設使用出廠RGB內參，原廠與重標結果分檔；10項離線測試PASS，實機仍NOT_RUN。相機工具與文件尚未提交／推送。

2026-11-19：依使用者授權完成相機可執行工具與詳細[操作筆記](../../doc/notes/camera_calibration.md)，型號更正為Gemini 335L。SDK2.1.3／OpenCV4.11.0隔離環境、9項已知幾何／影像測試、CLI與語法／連結檢查PASS；PDF44標記ID0–43、legacy排列70角點核對成功。套件預設會拉入另一份opencv-python，安裝入口改以contrib與SDK --no-deps避免覆蓋；無關範例依賴移除。SDK枚舉0台，真機採集、控制屬性設定與各項精度NOT_RUN；沒有機械臂操作、手眼、M6訓練或發布。

2026-11-19：補齊相機標定範本board.json的PDF名義尺寸、11×8格、70內角點、44標記與印刷檢查；數量及JSON格式核對通過。完整字典、ID排列與實測尺寸待確認，未執行標定。

2026-11-19：依使用者指定資料樹建立本地camera_calibration/session_template，含5份JSON、CSV表頭、影像與results目錄及填寫說明。JSON解析與Git忽略檢查通過；未填造裝置或標定數值，實際採集與標定仍NOT_RUN。

2026-11-19：新增[相機標定筆記](../../doc/notes/camera_calibration.md)，記錄Gemini D335L出廠參數核對、使用者ChArUco板資訊、內參採集及資料規格。字典、實測板尺寸、串流與固定方式待確認；僅文件整理，內參／深度／手眼驗證均NOT_RUN，未操作真機。

2026-11-16：整理M5可視化、PACE研究定位及獎勵學習筆記供雙遠端發布。發布前31項數值測試PASS、Python語法及差異檢查通過；繪圖需NumPy／Matplotlib與Noto CJK字型，主機預設Python缺Matplotlib，使用既有Isaac環境及主機字型重建檢查。沿用三條實測影片證據，未重跑完整仿真；影片／計畫留本地。使用者先完成學習，M6實作與訓練仍NOT_RUN。

2026-11-16：依使用者提供的M6學習筆記補充 [PACE獎勵解讀](../../doc/notes/pace_rewards.md)：新增AIRBOT接口對照、局部高分但整拍失敗的反例、策略步長與事件獎勵尺度，以及候選驗證目的。保留原碼細節與既有公式；只更新文件，未執行候選檢查或啟動M6。

2026-11-16：補充 [PACE獎勵解讀](../../doc/notes/pace_rewards.md)。四個固定版本來源檔校驗一致，核對T1實際參數、15 cm擊球引導平台、事件式回球評分與時間縮放，以數學例子解釋排序及限制。僅靜態閱讀／算例，非原版訓練驗證；依使用者要求先理解，不啟動M6實作，M6–M9仍NOT_RUN。

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
| 2026-11-16 | 球物理參考與 PACE 來源核對 | 固定 PACE 原始版本，釐清仿真／預測器質量與係數差異；本地實作單一球參數、世界阻力及靜態 CAD 拍面測試 | 數值14項 PASS；240／480／960 Hz 各自 PhysX 程序與收斂比較 PASS，入口 `./verify_ball.sh` |
| 2026-11-16 | 球拍凸包不等於平坦擊球面 | 整拍單一凸包連接厚柄與薄拍頭；−3 m/s 正面來球產生約−0.0233 m/s 側速，原 CAD 靜態三角面側速為0 | M3 在 link6 Z=205 mm 分成拍柄／拍頭凸包；正面出球側速0，速度與落點和原CAD參考一致，裸臂／工具回歸通過 |

| 2026-11-16 | M3 單拍任務與事件判定 | 凍結四組低旋來球，各重複兩次；三組碰到預先就位拍面後落回己方，一組偏球漏接，合法回球0/4；撞網、碰固定件及故障後重置通過 | `--mode shot` 8項 PASS；數值19項 PASS；速度／時限／碰撞納入姿態可達性，尚未實作主動擊球 |

| 2026-11-16 | 底座改為桌面直接固定 | 依使用者要求，底座移至己方桌端中軸線，底面貼760mm桌頂，後緣距桌端10.77mm；重求接球初始姿態，保留原四組來球；只排除底座／桌面安裝接觸 | 數值20項、單拍9項 PASS；三組接觸後己方落桌、一組miss，0/4合法回球；重置軌跡完全重現，仍非主動回球 |

| 2026-11-16 | M4 連續預測與步長選擇 | 連續積分重力／三維阻力，桌面回彈後繼續預測；240Hz落點時間P95=11.883ms FAIL，480Hz=5.633ms PASS，未改門檻或係數；單拍改480Hz，Foundation保留240Hz | 23項數值PASS；24例事件分類一致，預測25項檢查PASS；同場景最大位置6.81mm、時間2.95ms；裸臂／工具運動回歸PASS。桌緣／網邊擦碰仍unsupported，尚無主動回球 |

| 2026-11-16 | 統一研究參考入口 | SOURCES新增論文／研究程式索引，區分待讀、已核查及已採用；收錄使用者新提供的物理模型論文及團隊參考 | README與notes已連結；僅整理來源，未宣稱閱讀新論文或變更物理模型 |

| 2026-11-16 | 文獻索引審查 | 補Gate／證據範圍、Sony Ace固定版本與資料／示意程式入口、Google資料授權、Ruckig限制及研究候選 | Sony Physics v2版本／範圍已核對；Isaac Sim4.5文件404標BLOCKED；未採用新演算法或運行上游訓練 |

| 2026-11-16 | 研發筆記按里程碑整理 | doc/notes/取代單一notes.md，補M0–M4做法、代碼理解、驗證與限制；另分裝配與控制準備主題 | 文件及相對連結檢查，沒有新增仿真結果；[筆記索引](../../doc/notes/README.md) |

| 2026-11-16 | 補齊M0–M3實作筆記 | 補關鍵方程、代碼流程、原碼定位、測試含義與負結果，區分歷史階段和M4後現行行為 | Markdown連結與差異檢查；未重跑仿真或新增PASS宣稱 |

J6 的 Ø4.00 mm 孔徑與 4.00 mm 孔深為 STL 參考，螺紋尚未確認。使用者已確認實物鎖固，仿真採理想固定安裝；工具物理驗證限定於已確認的質量與均勻 CAD 分布近似。

需求與驗收見 [SPEC](SPEC.md)，模型和控制方案見 [DESIGN](DESIGN.md)，使用方式與 CAD 成果入口見 [README](../../README.md)。

| 2026-11-16 | M5有速度擊球與三目標scripted基線 | 兩段C2五次插值、解析極值限位、曲線碰撞；DOP853逆向落點求解、無摩擦相對碰撞法線、IK及最小範數法向Jacobian解。完整規劃約1.78–1.99s，改為發球前名義三條script，發球後50ms選用；驗證來球不逐球重算 | 開發9/9、驗證36/36合法且命中；每目標12球，平均落點誤差18.21–19.00mm；[M5實作筆記](../../doc/notes/m5.md)含資料流與限制 |
| 2026-11-16 | 可動拍較高速度的步長失敗 | 低速240Hz不收斂；實際script480→960Hz落點差10.2–10.7mm也超10mm門檻。追加1920Hz對照，960→1920Hz落點≤3.35mm、出球向量≤0.57%、時間≤1.042ms，採M5專用960Hz，未放寬門檻或擬合係數 | 四個獨立PhysX程序PASS；收斂接受960Hz，粗步長FAIL保留；相對法向能量差≤1.33%。M3／M4仍480Hz、Foundation仍240Hz |
| 2026-11-16 | M5交付回歸與範圍 | 31項數值、六組空揮拍、裸臂9項、工具10項、原單拍9項（含M4同場景預測）均PASS及退出0。實際script最大TCP位置0.579mm、線速度29.22mm/s、擊球時間0.577ms；來源雜湊核對一致 | 未重錄影片；幾何、質量、安裝和物理係數不變。結果只覆蓋合成高反彈小擾動分布，未做在線重規劃、PPO或真機。工作區未提交／推送 |

| 2026-11-16 | M5發布受網路阻塞 | 依使用者push授權建立本地實作提交c794c07；個人實驗分支與org/main的fetch／push均遇TLS握手中斷，HTTP/1.1重試及SSH亦無法連通 | BLOCKED：未確認任何遠端更新；本地計畫及原始報告未納入Git，恢復連線後以普通push續推 |

| 2026-11-16 | M5發布恢復 | 重新fetch確認兩端皆可快進，普通push個人exp/ball-physics與團隊main；遠端實讀均確認收到a62b51c（含M5實作c794c07及筆記） | PASS：先前網路阻塞解除；個人main仍6809f7c，origin URL不變，未force或改寫歷史；本地PLAN／runs未發布 |

| 2026-11-16 | M5實測視覺化 | 新增`--mode return-demo`，960Hz物理／60Hz取畫面／四分之一速度，錄製同一驗證來球對三個目標；球心標記及目標色框僅供觀看。以原36次驗證報告繪製落點放大圖，另疊加本次實測回球路線 | 三次合法回球與原驗證事件順序、位置、時間完全一致；17.2秒／258幀影片完整解碼PASS。圖已嵌入[M5筆記](../../doc/notes/m5.md)，影片與逐幀資料留本地；未重跑整組36球 |

| 2026-11-16 | M5後研究主線與筆記形式確認 | 維持PACE→AIRBOT遷移及邊實作邊學習；M5凍結作比較基線，即時規劃加速不阻塞M6。M6重點是reward／predictor、M7保留五組PPO消融並補同條件對照及960Hz吞吐檢查，M8再做goal擴展 | 僅方向與文件更新，M6–M9仍NOT_RUN、預算不變。notes可用圖／影片／HTML；本輪清除3張冗餘本地預覽並停止自動輸出，保留代表圖、影片、原始證據及生成入口；未重錄仿真 |

## 2026-11-19：採納通用研究工程流程

- 保留 PACE／Isaac 主線；選入三份原始規約、LICENSE 及來源雜湊，未移入 CartPole trainer 或任務管理系統。
- 新增環境分組驗證、不可重用的執行目錄、來源／結果雜湊、共同條件與方法變因分開的描述比較。10 項研究工具、31 項 core、17 項 PACE、11 項相機，共 69 項 PASS；主機缺依賴正確 BLOCKED，skip 不算 PASS。
- M6 train／evaluate 分離回歸 PASS；曲線、逐球／分段誤差與既有結果一致，RMSE 11.5803 mm。禁止 Torch 匯入後的舊報告繪圖 PASS。原資料／報告保留，新回歸證據在本地 `m6/workflow_check/`，不是新獨立研究樣本。
- [研究流程筆記](../../doc/notes/research_workflow.md) 說明用途、程式流程及後續 M7–sim-to-real 使用方式。PPO 診斷、推論統計及完整 resume 尚未實作；沒有開始 M7、硬體操作或推送。

## 2026-11-19：M7 初始整合與 A／B 診斷

- 採納審查：保留 apex 教師；記錄平台／有效引導時間，區分 A／B reward 與 D／E predictor 問題；先驗 action 能力，不用短 smoke 判方法有效性。PLAN 前面新增唯一現行摘要，舊狀態明列歷史。
- 直接重用固定 PACE ActorCritic／PPO／Storage，以及 Lab 2.1 InteractiveScene。共用資產建構、局部座標及 tensor 事件；16-env reset 隔離、重置 TCP、末態與 timeout PASS；固定姿態球路與舊 SingleShot 同結果。
- 初版逐目標煞停造成 48Hz 約38.7mm位置誤差；提高頻率／補償及局部擬合未解決原48Hz驗收。退役該實作，改有界三次插值後：0.91mm、24.8mm/s、0.58ms，沿用原門檻PASS。物理960Hz／策略48Hz不變。
- 首次 smoke 因 InferenceMode 跨評測更新 Isaac buffer 失敗；改 no_grad，補官方 reset forward，重驗 gate 與 smoke PASS。成功 smoke 1,536 transitions／約33.1s，actor和critic均實際更新且重載一致。
- A／B 固定每seed6,144 transitions，共36,864；6次訓練及評測完整，流程約522s。每seed每組16顆固定評測球全計入：A碰拍16／16、16／16、0／16，B皆0／16；全部合法回球與目標命中0。零動作同球基線16／16碰拍、8／16合法、0／16目標。這是短跑負結果，不判定apex必然無效；全部評測平台停留0秒。
- 73項分組單元PASS；本輪沒有C／D／E、在線MLP或真機操作，未commit／push。[M7筆記](../../doc/notes/m7_notes.md)包含程式流程、公式、負結果、代表圖與重跑入口。圖以既有JSON重建，checkpoint／完整報告留本地。

- 收尾回歸：原 M5 development 的三條中央／左右目標空揮拍與9/9合法回球 PASS；單拍場景、固定工具與驅動未退化。未重跑原36球獨立驗證，不把development回歸冒稱新的held-out研究。

- M5首次收尾數值完成但外層退出143，保存INCOMPLETE（原因未定位）；重跑同入口數值與退出0確認PASS，未覆寫首次失敗紀錄。

## 2026-11-19：M6／M7 與研究流程發布

使用者授權推送個人 `exp/ball-physics` 與團隊 `main`。發布內容包含 M6 離線實作、研究流程工具、M7 初始整合與 A／B 負結果、筆記及兩張代表圖。沿用已完成的73項單元、M7 gate／smoke／A-B與M5開發回歸；發布前核對來源雜湊與正常退出，不重跑訓練。本地 PLAN、資料集、checkpoint及完整執行紀錄不納入Git。

## 2026-11-19：M7 負結果原因分析

只讀六組訓練日誌並重播原 checkpoint／零動作，112 顆原評測球的碰拍與結果分類全部重現，退出0；不是新增獨立樣本。代表案例確認 A 的兩組碰拍後先落回己方，另四組漏球出界。首次 PPO 更新 KL 10.62–25.57（目標0.01）、學習率降至下限，是優先定位的更新穩定性問題，底層原因尚未證實。三組訓練曾有合法回球，不能把最終均值評測0當成從未探索成功。已在[M7筆記第9節](../../doc/notes/m7_notes.md)補物理案例、reward／動作限制與排查順序；沒有重訓、調參、改教師、commit或push。

## 2026-11-20：定位並修正 M7 首次 PPO 更新過大

- 六組首次rollout的mu／sigma／log-prob一致，最大log-prob誤差9.54e-7。用相同資料／權重／RNG重現原KL10.62–25.57；只降低初始learning rate，首次完整更新KL降至0.00487–0.01349。原因是原初始更新幅度對本地小批次設定過大；保留官方PPO adaptive，不改原碼。
- 預先選定1e-5，再做同樣三seeds×A/B×6,144 transitions。A合法16／16／3、B合法16／16／4（每seed16顆），全部碰拍；第三seed其餘主要撞網，另B有1顆己方落桌，目標框全部0。原零動作8/16；未訓練均值策略11／3／0。改善不等於已解決seed穩定性、泛化或證明B較好。
- 新預設初始率1e-5，與候選protocol的完整cfg一致。22項PACE單元、首次更新診斷、六組短對照、初始策略評測、新設定lifecycle／control／parity皆PASS且執行程序退出0；控制0.91mm／24.63mm/s／0.584ms，門檻未改。沒有重跑完整M5或正式五組研究。
- 新增可重用首次更新診斷、顯式學習率實驗參數與一張小型比較圖；更新[M7筆記第10節](../../doc/notes/m7_notes.md)、README／SPEC／DESIGN／索引／本地PLAN。診斷約38.4s、短對照528.7s（不含其他驗證啟動耗時）；checkpoint、rollout、原gate及失敗證據留本地。未commit／push，未操作真機。

## 2026-11-20：第三seed拍面／拍速與網頂診斷

A/B seed7103與成功seed7101共64顆原評測球全部重播一致，PASS、退出0。第三seed向前TCP速度與出球速度較低，B另有明顯側偏；第三seed合法球實測球底淨空僅26–41mm，成功對照241–598mm。另6顆球心越過x=0時雖有4–20mm淨空，下降中的球體隨後擦網；不能用單截面高度代替整球通過。更正舊文案：B7103是11net+1wrong_bounce，不是12net，原報告保留。新增[M7筆記第11節](../../doc/notes/m7_notes.md)及一張引用圖，原始trace留本地；無重訓／調參／物理或介面變更，未commit/push。


使用者隨後校正回合規則：擦網後合法落到對方桌面應算有效回球。現行一碰網即終止過於嚴格；上述net數量是舊實作分類，並非已確認的真實違規。已在M7筆記註明，後續優先修正事件判定並重評原checkpoint；本輪尚未修改事件機或追蹤擦網後落點。

## 2026-11-19（America/New_York）：修正擦網判定並重評

- 兩套事件機採`rally_net_contact_v2`：net只記接觸，取消高度捷徑終止，同子步先跨網再落桌；實際網碰撞與物理參數不變。增加只重評的`pace-eval`，校驗原checkpoint及完整cfg，標記新舊分類／規則版本。
- 原lr1e-5六個checkpoint＋zero、同16球共112次評測PASS／退出0，未重訓。A7103由3/16到6/16、B7103由4/16到8/16；新增3／4顆為擦網後合法，其他四組仍16/16。零動作由8/16到16/16，目標框全0；現有策略尚未優於新規則的零動作基線。
- core31＋PACE23單元、新規則reference／parity／lifecycle／control、獨立實體擦網後成功／被擋回失敗案例均PASS。完整M3單拍回歸及重跑則在left碰撞可達性HiGHS status15失敗，未定位，不冒稱全回歸PASS。另一次gate因執行中註解改動觸發來源檢查失敗，固定來源後已重跑通過；失敗證據保留。
- [M7筆記第12節](../../doc/notes/m7_notes.md)、README／SPEC／DESIGN／本地PLAN及索引已同步。新報告m7/net_rules_v2*、原gate與歷史評測留本地；無commit／push、真機或新增訓練。


## 2026-11-19：碰撞求解修正與來球範圍單變因對照

- 保存M3出錯姿態及link3/link5凸包，重現HiGHS Unknown；有限輸入下的原零目標可行性形式數值求解失敗，未定位上游內部缺陷。改用同幾何／同裕度的最大共同半空間餘量LP，仍用SciPy HiGHS，非最優仍拋錯。centering與IPM-only候選因回歸失敗撤除。
- core33、PACE23、完整M3九項及當前lifecycle／control／parity均PASS。解除前輪M3阻塞，未放寬1mm／0.5mm碰撞間距。
- 凍結新效用門檻及篩查規則；dev zero在±12cm為11/16，選該寬度。僅改訓練來球Y範圍：±1.5／±12cm，固定A、三個相同seed、各6144 transitions，共36864；獎勵、控制、教師和M5設定不變。
- seed7411共同評測448球完整PASS/exit0/hash一致；zero窄／挑戰32/32、13/32。窄訓練三seed窄32/2/32、挑戰22/0/21；寬訓練窄32/32/26、挑戰17/18/12（均分母32）。兩條件整體效用FAIL，沒有挑好seed或追加預算。窄7102大量非預期球接觸需後續逐部件重播。
- [M7筆記第13節](../../doc/notes/m7_notes.md)補公式、失敗候選、資料流、結果、限制與一張成功率／落點圖；SPEC／DESIGN／README／SOURCES／索引及本地PLAN同步。正式五組消融與真機未做，未commit／push。

## 2026-11-20：M7.4首段A/B開始，區分接到與回好

- 核對原計畫後保留六維關節residual、官方PPO、原apex及15cm平台，按A→B→C→D→E進行；A未過效用不阻塞測B。同步已驗證的lr1e-5與net v2，不改教師或換末端控制。
- 重播7102前16球，三個策略的結果均與保存資料一致。10次非預期接觸均是paddle接觸、球心被現行規則分在柄區，未量測shape級實際接觸點。±12cm的M5定目標揮拍與平移見證未通過；保留失敗，不把搜索失敗當不可達證明。
- 新合成開發題目：初始Z中心2.3m、Y±.015m。dev7510的16球zero1/16、經原關節介面的可行參考動作16/16；動作最大0.6297、最小時間餘量11.60ms，保持原碰撞／速度限制。新增`--pace-serve-z`保存有效高度，M5原檔不改。
- 23項PACE單元及lifecycle／control／parity PASS。A/B各三seed×6144 transitions，共36864，約533.5s；新seed7511的448球主評測完整PASS／exit0／來源一致，約348.6s。
- 新題目六個模型有效碰拍均32/32；合法回球A為32/3/32，B為0/0/32。舊題目合法A為32/0/11，B為0/0/32。zero新題目1/32、舊題目32/32；目標框命中全部0。兩組跨seed綜合效用FAIL，但不能將碰拍進展與回球不穩定混成完全沒學會。
- [M7筆記第14節](../../doc/notes/m7_notes.md)記錄開發負結果、同介面可達見證、稠密引導與成功參考路徑不同向的診斷、代碼資料流、A/B主結果及限制；只新增一張被引用總圖。C/D/E、在線predictor及真機尚未完成，沒有commit／push。

- 主評測後追加192球初始化對照（明確post-hoc、同一已用評測集，非新獨立驗證），約134.2s、PASS。新題目未訓練有效碰拍1/4/31、合法1/1/29（分母32）；A／7101提高至32/32較能顯示更新後進展，7103初始就有29/32不可全歸功於訓練。後續正式實驗預先保存未訓練Actor對照。下一工作包為C的離拍／return guidance接合，尚未執行。

## 2026-11-20：M7階段成果發布整理

使用者授權將累積的學習率診斷、擦網判定、凸包碰撞修正、來球覆寫入口及M7.4首段A/B研究紀錄同步至個人`exp/ball-physics`與團隊`main`。發布前fetch確認兩端與本地基線無分歧，核對M7.4開發／訓練／主評測／初始化對照的來源雜湊與退出碼，沿用已有驗證、不重跑訓練。Git只收必要程式、測試、文件與被引用代表圖；本地PLAN、模型、逐球資料及完整執行證據保持忽略。上述「未提交／推送」描述保留為各輪當時狀態。

## 2026-11-20：確認後續研究順序，C優先

採用使用者路線：先驗證離拍／預測落點／過網高度與一次性return reward，再做與A/B同題目、控制、seed及樣本預算的C短對照；接著D檢查兩種引導互補、E檢查learned predictor的額外價值。E必須先驗新Z2.3分布，不能外推M6精度。

若首輪仍不穩定，才獨立替換reward引導目標為經動態可達性驗證的擊球點；若探索仍不足，再做BC獨立基線及BC+PPO／從零PPO對照。保持action、reward、teacher等因素的歸因界線。M8目標XY與Sim-to-Real建立在可重現策略及H0–H4證據上，不因列入路線就啟動真機。此輪發布偏好另已記錄：往後個人／團隊都使用各自main，沒有自動推送。


## 2026-11-20：C回球引導接合與短對照完成

- 向量環境接入有效接觸後的離拍鎖存、CPU參考回球預測及一次性事件獎勵。25項PACE單元、lifecycle／control／parity及96球固定模型實體重播PASS／exit0；41個無干擾有效預測的落點／網高P95誤差3.247／4.414mm，時間0.846／0.173ms，A/B行為一致。
- 同題目／控制／預算完成C三seed各6144 transitions（約282.2s）。訓練前保存初始Actor，固定最後checkpoint；新seed7521對zero、initial3、A/B6、C3各新舊32球，共832球PASS／exit0（約653.1s），來源及權重驗證一致。
- 新題目合法回球A32/4/32、B0/0/32、C0/24/3，initial1/0/22，zero4（均分母32）。C舊題目0/32/32，整組效用FAIL，全部目標框命中0。C7102局部改善不能代表跨seed增益；7101／7103有效碰拍退步。
- C7102另有17顆實際合法、未碰網的回球被參考模型標unsupported_table_edge而無引導分數，保留覆蓋限制，不據此推定全部失敗根因或改模型。完整資料流、結果及單張圖見[M7筆記第15節](../../doc/notes/m7_notes.md)。下一步D只增加兩引導組合；E前先驗Z2.3 predictor。未啟動D/E、BC或真機，未commit／push。


## 2026-11-20：D擊球＋回球引導短對照完成

- D共用B的hit增量和C的一次性return增量，基本獎勵不重複，action／觀測／控制／教師／桌邊遮罩未變。96球固定模型重播PASS/exit0：D−C−hit最大1.058e-6分，8次全reset與15次部分reset隔離PASS，自由飛行精度與C一致。25項單元及lifecycle／control／parity亦PASS。
- 訓練前凍結7531評測與既有初始化SHA，D三seed各6144（總18432，291.6s）；zero+initial3+A/B/C/D各3共1024球共同評測PASS/exit0（750.5s），來源／執行程式／權重驗證一致，固定最後模型，未重訓A/B/C。
- 新題目合法A32/4/32、B0/0/32、C0/23/1、D0/7/0，initial1/0/27、zero2（每格32球）。D有效碰拍32/7/32，第一／三seed全wrong_bounce，第二25顆unexpected_contact；舊題目D0/32/0，目標框全部0。D三seed效用皆FAIL，未呈現互補增益；接合正確不代表學習成功。
- D7101有5個有效離拍預測最終失敗且均有網接觸，保留預測範圍限制，不把它或獎勵尺度當成已證實根因。詳細方法、結果與一張圖見[M7筆記第16節](../../doc/notes/m7_notes.md)。下一步E先收集／切分／驗新Z2.3 predictor；尚未開始E、可達點／BC或真機，未commit／push。


## 2026-11-20：E新分布資料、在線接口與覆蓋診斷

- Z2.3重新收集128/32/64整serve後開窗，224條皆有apex，獨立資料完整性審核PASS/exit0。收集啟動程序退出143，雖資料已完整落盤，仍保留異常退出，不列程序PASS；最初CLI高度限制及系統/tmp無餘量亦已定位處理。
- 原MLP／20epochs／原門檻不改，test2,428窗3D RMSE10.88mm、固定train平均點12.85mm；物理／等速共同2,424窗342.85／1832.27mm。物理基準反彈前／後5.89／2.34mm更準，跨反彈估速拖累整體；有效窗口覆蓋原始89.40%，共同覆蓋99.84%。離線原30mm／優於等速及教師門檻PASS，非沿用M6舊結果。
- E增加3D+valid至actor135維，critic200維不變，獨立在線MSE近期buffer／batch1024／恰好前20次更新資格，未載離線模型。26單元、真向量same-step／reset／無效／checkpoint、lifecycle/control/parity/combined皆PASS/exit0；實測巢狀history9時刻、166.7ms。
- E四輪smoke（每次1536 transitions）首次使用提示16回合在0.5s，後續18回合83.3ms；擊球前699/1218步用到提示，12次離拍中6次有效return。4次MSE epoch實際僅5次optimizer更新，使用中提示對教師RMSE2.42→2.37m；不能套用離線10.88mm或宣稱在線提示有效。E自己的initial已保存，同16顆開發球合法由0到4，但非D/E公平比較。
- 獨立重載初次CPU/GPU張量比對FAIL，修正後16球一致、PASS/exit0；為驗證目前runner來源另重跑相同smoke，結果一致，兩次共3072個工程驗證transitions，沒有選模型。新增一張引用圖及[E專題筆記](../../doc/notes/m7_predictor.md)，M7第17節／索引／README／SPEC／DESIGN／本地PLAN同步。E三seed效用NOT_RUN；下步先明確在線更新量／收斂診斷，不默默改為離線預訓練，未commit/push或操作真機。


## 2026-11-20：在線predictor原20次資格收斂診斷

固定D7101/7102/7103權重、共同serve7540與原OnlinePredictor配置，各16env×24×20=7680 transitions，總23040，PPO更新0。三條流完整PASS/exit0，320.8s，policy及來源不變；第21次cutoff且模型/RNG/optimizer計數不變。實際optimizer步20/37/36，固定validation RMSE由2.409m降至1.878/1.282/1.325m，三條皆未達30mm參考；固定train均值validation12.08mm。提示可提供約78.06%/80.87%/79.63%，D不讀提示，不冒稱實際使用或E聯合訓練效果。保留全20點與最後模型，沒有重開test、選checkpoint或加預算。[E筆記第7節](../../doc/notes/m7_predictor.md)附一張曲線；下步候選僅batch1024對32的同資料／同初始化單因素診斷，另報增加的optimizer工作量。未執行該對照，未commit/push或真機。


## 2026-11-20：batch1024／32配對診斷完成

同三個固定D、同初始化6100、同資料buffer／shuffle及20次資格；每流7,680 transitions，共23,040，PPO更新0。工程PASS／exit0，325.5s，來源／固定權重／配對／cutoff驗證通過，1024逐點重現舊結果。validation RMSE（mm）1024為1878.2／1282.4／1325.2，32為685.4／713.7／795.5，下降40–64%，但兩組三條皆未達30mm。optimizer20／37／36→501／809／719；非等計算量，不能將改善唯一歸因為更新次數。未開test或選checkpoint，正式配置仍1024，E/D效用NOT_RUN。[E筆記第8節](../../doc/notes/m7_predictor.md)保存方法、曲線及成本。下一步建議查在線資料與validation的覆蓋／標籤差異，尚未執行；未commit/push或真機操作。


## 2026-11-20：predictor資料與擬合診斷完成

既有模型純推論交叉評測PASS／exit0，未開test或訓練。三流在線32對自己buffer教師RMSE113.1／74.8／56.2mm，validation685.1／713.4／795.1mm；離線MLP在buffer10.3–10.5mm、validation9.91mm，固定平均約12mm。validation反彈後1.06–1.24m，貢獻>99%平方誤差；buffer較晚飛行覆蓋不足，與自身擬合不足並存，未唯一歸因。

固定D7102、serve7540、16env×72步=1152 transitions補採PASS／exit0（27.3s），MSE/PPO0。879窗座標／時序／快照／reset／hit遮罩核對通過；871窗固定與重算教師RMSE1.49mm，8窗重算below_surface保留且未從模型誤差排除。官方4096env滿buffer每epoch492 optimizer步、cutoff20 fresh-run21資格是原碼推導；本地恰好20保持。詳細交叉表／XYZ／phase／env／圖見[E筆記第9節](../../doc/notes/m7_predictor.md)。下一優先提案E-frozen隔離提示價值，未實作；E-online保留，無參數掃描／策略對照／commit/push或真機操作。


## 2026-11-20：同球四種動作可視化

使用原M5 validation_0，比較原中央script、zero、既有A7101／B7101；兩段真PhysX重播PASS／exit0，無新訓練或state write。M5／zero／A合法，B碰拍後wrong_bounce。碰拍前TCP速率.871／約0／.543／.730m/s，X分量+.817／約0／+.456／−.666；落桌X/Y為(.7445,−.0037)／(.1183,−.0039)／(.1730,.2475)／(−.8822,−.0431)m。M5命中中央，A只是合法；單球不得代表策略效用。新增[動作筆記](../../doc/notes/motion_comparison.md)及一張引用圖，6.4秒0.25倍速影片／trace留本地。E-frozen暫未開始，沒有commit/push或真機操作。


## 2026-11-21：DeepMind Dataset-to-AIRBOT Feasibility Audit

固定官方54cf18e4，15,792筆（2704 serve／13088 rally）來源／授權／九維狀態／proper座標＋桌高已核對；全資料分布與零旋預測完成。速度中位5.49m/s、P5–95 3.90–7.68，spin中位48.7rad/s；無完全零旋紀錄。保守≤10rad/s得814，現任務階段相容343（342 rally／1 serve），沒有把其餘資料當策略失敗。

固定面96筆0候選；位置診斷96+247全343得8命令候選，補驗全動作碰撞；原M5中央有限規劃343得0候選。真PhysX共102個rawrow、176cases（96zero+80cross）PASS／exit0：zero0合法，8專用全碰拍但7wrong_bounce/1out，矩陣15碰拍/80、0合法，單固定腳本最多碰3/8。零旋反彈位置／時間P95 4.78mm/1.09ms，非真spin物理驗證。343中335是未找到見證，不是不可達證明。

[審查筆記](../../doc/notes/deepmind_feasibility.md)與一張分布／矩陣圖、來源與重跑入口已完成；8接觸案例另存、合法回球清單空且NOT_READY，未接入訓練。下一步需先取得少量合法回球見證，再建立不同動作與獨立測試分布；不改PPO／預算／碰撞或ready。未commit/push或真機操作。


## 2026-11-23：DeepMind小型完整回球見證搜尋

依使用者授權，保留原控制／物理／ready／PPO，前三個train接觸案例9378、10818、5255各27個擊球時間／中央落點／飛行時間組合，以面內轉角自由的接觸位置＋法向IK診斷。81組／162次局部求解均未過姿態條件，未進拍速／軌跡與PhysX，未找到新合法回球見證；不構成不可達證明。CPU5.25s、exit0、source SHA不變；流程PASS、見證未取得、H6及E-frozen NOT_RUN。詳見[審查筆記§10](../../doc/notes/deepmind_feasibility.md#10-從碰得到補查打得回去2026-11-23)。原summary與176次重播不覆寫，沒有commit/push。


## 2026-11-23：目前安裝的球拍工作空間採樣

兩範圍各4096 Sobol姿態＋ready，現在±0.6rad全4097靜態清晰；完整URDF範圍2953清晰／1144碰撞，納入現在集合後7049姿態。保留拍心、法向及正拍面球心潛在接觸點，球桌／網／地面球體穿透另篩；一張三視／3D圖與[筆記](../../doc/notes/racket_workspace.md)。目前球心Y採樣極值−.355..+.358m，非每個高度都可達的方盒。CPU流程PASS／exit0、無PPO／PhysX／真機；動態路徑、時限、合法回球NOT_RUN。支援另存底座平移診斷，非安裝支撐認證。

追加九組對照：±.3/.6/1.0rad × 後退0/.15/.30m，各4097姿態，原位.6復用，新增8組32776次靜態姿態檢查。原位最前球心X−.876/−.779/−.677m；後退.3且±1.0到−.977m。±1.0碰撞79/79/76，.3/.6全0；退後僅多3個靜態姿態，未證明更易回球。comparison.json／verification與第二張九宮格圖完成，支架未建模，維持原配置。


## 2026-11-23：±0.6／±1.0rad回球姿態配對診斷

三個固定train球，每組81候選／162次局部IK；±0.6逐候選重現舊誤差，兩組候選輸入相同。兩組姿態合格均0，±1.0最佳誤差改善但仍不達標。81候選中80個較寬解貼到J5 URDF上限1.7628rad；ready1.1使正方向最多+.6628rad，J3亦受URDF下限限制。原始官方URDF值核對一致；不作實機限位確認或全域無解聲明。CPU5.28/5.43s、exit0，配對工程PASS，回球見證未取得，軌跡／PhysX NOT_RUN，正式PPO／物理／安裝均未改。詳[筆記§11](../../doc/notes/deepmind_feasibility.md)。


## 2026-11-23：固定五顆DeepMind train球A組PPO完成

新bank接口與34項PACE單元PASS；當前source lifecycle/parity/control/bank四個PhysX gate全部PASS/exit0，五舊腳本重現5/5碰拍（4wrong_bounce/1out）。三seed7101/7102/7103各6144，共18432transitions，actor/critic更新且loss有限，固定最後模型。主要評測zero5+initial15+final15全0有效碰拍／合法；重載15回合逐球一致。訓練完成回合116/112/118，有效碰拍5/1/3，全部0合法；合計9/346接觸，8wrong_bounce/1out。這次A短配方未學成，非不可回證明、非泛化測試。

主流程225.22s wall time、PASS/exit0，沒有加預算、改ready／action／物理。小manifest、權重、逐球motion與發球序列保留m7/deepmind_five；[筆記](../../doc/notes/deepmind_ppo.md)及單張矩陣圖已完成。下一步先選接觸探索的單變因診斷；未啟動BC／±1.0／新reward或真機。


### 2026-11-23：五球接觸BC初始化的固定配方完成，閉環門檻FAIL

新增獨立pace-bc及Actor-only初始化接口；核對固定PACE上游Distillation後，以原Actor子網加PyTorch Adam/MSE接合，不改原PPO／環境／控制。原35球zero/initial/final與五接觸腳本重現；收237筆首episode快照、三seed各2000full-batch更新，Critic/std逐值不變，最後模型及重載皆0/5碰拍、0/5合法。離線MSE4.7394e-7／5.1551e-7／5.9745e-7；44單元PASS、主流程PASS/exit0、127.23s walltime（fit11.46s），BC接觸gate FAIL，噪聲及BC＋PPO NOT_RUN，無新PPO transitions。

另作15次只讀閉環定位，保持所有權重／環境：15案例action誤差於.146–.271s超.05、q誤差於.125–.208s超.01rad，均早於腳本.579–.959s接觸前時刻；保存示範觀測的逐步誤差仍≤.002121。這是偏離累積的直接量測，不獨斷前動作／qd／控制器的因果占比，也不以小loss宣稱模仿成功。單張圖與資料流／負結果／入口在deepmind_ppo筆記，候選下步為先驗證偏離狀態的有效修正示範；未開始新資料聚合或訓練。未commit/push／真機。

補充重播的Kit退出問題已修：沿既有clear_instance/post_quit/fast shutdown生命週期，最終PASS/exit0。兩次失敗發生在量測結束後，三次共45episode量測完全一致，仍只計15案例；沒有重做BC，保留失敗metadata與最終逐步資料。


### 2026-11-23：同球BC三方式實際物理重播與影片

依使用者要求先看影片，固定第一ID9378／seed7101，不訓練；原五球每方式一次，共15個計分episode。原script5/5hit、BC讀保存示範觀測預先算action後播放5/5hit、BC自主讀即時obs0/5hit；全部0legal。新預算BC與script逐球同4wrong_bounce/1out，自主BC重現歷史結果。擷取session63595 PASS/exit0，原source／script／資料／權重hash及權重不變核對通過。這是實際PhysX對照，支持自主回饋後偏離的方向，不是全三seed結論或可部署反應式政策。

`tests/pace_bc_rollout.py --capture-comparison`復用原bank與motion量測，120Hz記姿態／960Hz事件；`scripts/plot_motion_comparison.py --bc-report ... --video ... --poster ...`重繪同步三欄0.25x，簡化外形明標。影片／poster／trace只留本地bc/replay_comparison，公開五球筆記記錄解讀與生成入口；未改正式PPO／觀測／控制，未commit/push或真機。

影片生成曾遇metadata與輸入JSON同名，已將輸出改成獨立_visuals.json，並在寫入前拒絕輸入／輸出路徑重合；新舊CLI拒絕案例驗證通過。以相同條件無訓練重播恢復完整擷取，SHA256與首次原檔逐byte一致，無結果漂移；實際共兩次×15episode，不當額外獨立證據。

### 2026-11-24：BC前動作／速度回饋四組介入完成

固定seed7101、原五球及權重，無訓練；自主／只替換五幀前action／只替換qd／共同介入，20計分episode有效碰拍0／1／4／4，全部0合法。前50ms實際q／qd無偏差，前action已影響輸出；187.5ms速度介入的五球action RMSE中位數.094374→.001223，實際q .033620→.000804rad。前action介入仍有單球退化，10818四組皆漏接，不能當成唯一根因或特徵刪除方案。

`tests/pace_bc_rollout.py --feedback-audit`只改Actor輸入副本，226次推論核對原觀測／歷史／真實前action／控制記憶未變；基線逐步action／q與舊證據重現。session85591 PASS/exit0、35.37s wall time，47單元PASS，完整20軌跡獨立只讀重算PASS。事前protocol、結果與NPZ留本地bc/feedback*；沿既有繪圖入口新增一張引用圖，詳細解讀與命令見[五球筆記](../../doc/notes/deepmind_ppo.md#bc前動作與速度回饋診斷)。原自主BC gate仍FAIL，noise／BC+PPO NOT_RUN；下一候選先驗偏離狀態恢復動作，未新增訓練、改物理或發布。

### 2026-11-24：五球接管可恢復，修正BC兩組達標

固定舊7101在6/9/12 tick接管，15/15實際有效碰拍、0合法（13錯誤落桌／2出界），10案達.01rad偏離並涵蓋5球，覆蓋PASS。15條接管軌跡提供452筆，加原237筆，各seed相同689筆做2000次更新。修正BC自主／重載均2/5、4/5、4/5，舊BC均0/5；新自主0合法，三seed共同gate FAIL，noise與PPO NOT_RUN。收集／擬合／獨立重算皆PASS/exit0，50項PACE單元PASS。重算720個控制步、452筆標籤與三組損失；兩次Adam建立前啟動失敗，修正匯入順序後完成三次有效擬合。詳見[五球接管與修正示範](../../doc/notes/deepmind_ppo.md#偏離後接管與修正示範-bc)。

2026-11-24：完成[新7101／9378的250ms接管診斷](../../doc/notes/deepmind_ppo.md#新7101的9378250ms接管診斷)，固定模型兩個PhysX回合，session59921 PASS/exit0、20.85s。自主漏接數值重現；接管0.932288s有效碰拍，1.092702s己方落桌，0/2合法。切換前obs/q/qd/前action/控制記憶逐值一致，正常運動state write0；舊三參考於250ms的q差.07003/.11620/.13976rad。新Actor在舊250ms接管樣本仍有.09254動作RMSE，不能把失敗全歸因資料覆蓋。獨立CPU驗112步／hash鏈／6組統計與Actor重推PASS/exit0；50PACE單元PASS，單圖及原圖像素回歸PASS。下一候選為其餘四個漏接(seed,ID)同時刻接管，不自動重訓；原三seed2/4/4 gate仍FAIL，噪聲/PPO NOT_RUN。未commit/push／真機。


### 2026-11-24：BC三seed初始化門檻通過，短PPO接觸退化

四個剩餘漏接案固定250ms接管，3/4恢復有效碰拍，全部0合法；7102/14852未救回，完整保留。成功後綴新增70筆，加前次9378的33筆，總資料792筆。候選1只加資料，三seed碰拍4/2/1，較原2/4/4有退化；候選2資料不變，只改官方cosine學習率衰減，固定2000步最後模型，自主與重載3/4/3，共同接觸門檻PASS。兩候選均0合法，候選3未執行。

依授權直接完成300次配對噪聲：原initial0/0/0，候選2為6/7/8（各50球），0合法。再各6144 transitions短PPO，共18432；最後自主與重載碰拍1/0/1（各5球），0合法、0目標。訓練332完成回合、42次碰拍、0合法。工程與來源／預算／初始化／重載獨立核對皆PASS/exit0，54PACE單元PASS；研究上沒有合法回球收益，不能把BC門檻當穩定回球能力。

[研發筆記](../../doc/notes/deepmind_ppo.md#短ppo完成初始化已可用但原配方未保住碰拍)附完整圖、資料流、歷史來源唯讀重算命令、負結果及成本。下一候選為固定BC後隔離探索噪聲，尚未開始；本輪沒有延長訓練、放寬保護、合併E-frozen、commit/push或真機操作。

### 2026-11-24：E-frozen獨立接口與配對研究準備

在隔離工作區把已驗Z2.3離線MLP接為顯式E-frozen，原E／E-online仍在線從零；D115維、E135維、critic200及D原reward／控制不變。來源鏈唯讀核對PASS，模型SHA為7e0f666b…，原validation選定epoch20、RMSE10.389mm；保留資料launcher exit143與後續持久資料完整性PASS，不重開raw held-out或選模。額外預訓練128train／32validation、3060 optimizer步、原2.641s另列。

CPU統一選組27/27 PASS、零skip／error／failure，session32686 exit0、執行1.088s，receipt為`m7/e_frozen/cpu.json`。12項模型／教師／窗口、10項環境張量／事件／控制、5項runner官方PPO合成更新；既有validation1223窗推論逐值一致、RMSE10.389121mm，checkpoint／storage不變、MLP無optimizer／gradient、initial RNG及online alias通過。修正NaN時間清history與凍結建構器只設CPU RNG；後者用CPU mock檢查CUDA seeding不被呼叫，不冒稱已測GPU RNG。公開入口`./run.sh --unit frozen`。

以上準備輪交付時，GPU凍結／控制／重載與三seed策略比較均NOT_RUN，當時沒有GPU或策略實驗；後續GPU工程狀態見下一條。配對規格固定7101–7103各6144 transitions、最後checkpoint，既有D保持凍結並在當前來源重評；Z2.3的已讀7531共32球，zero1＋D3＋E-own-initial3／final3，共320計分episode。這是配對開發診斷，不是新held-out。E九時刻166.7ms對D83.3ms及不同初始Actor限制均寫入[SPEC](SPEC.md)與[E筆記第10節](../../doc/notes/m7_predictor.md#10-e-frozen固定準確提示再問策略能否使用)，策略效用尚無結論。沒有回寫原工作區、commit／push或真機操作。

### 2026-11-24：E-frozen追加GPU工程關卡，禁止啟動訓練

在BC研究結束後，依使用者本次限定授權於隔離工作區順序執行五個GPU工程關卡。frozen2／lifecycle16／parity16／control2／combined16全部PASS／exit0；smoke／PPO／策略效用NOT_RUN，新增PPO與MSE更新0。本輪已在五個關卡後停止，下一個會更新PPO的smoke未執行。

frozen驗真135／200維觀測、九時刻same-step提示、保存觀測不變、正常步進無狀態寫入、局部reset／invalid清history及磁碟重載，實測首次prediction83.3ms、凍結權重與無梯度契約通過。首次空快取程序138.305s含場景／資產準備，不作推論效能指標。parity16僅正式比對env0，重現legal_return、最大位置差0.024512mm；control2正式TCP／時刻量測亦取env0，位置0.906785mm、方向0.002197rad、線速度0.024794m/s、角速度0.059565rad/s、時間0.583768ms，全在原門檻內。

combined程序62.666s，固定六個A/B模型96球全部重現，D=C+hit最大誤差1.05798×10⁻⁶、return不變；全量reset8次／局部15次隔離通過。49次有效預測中8次後續接觸另列，41個自由飛行樣本落點／網高P95誤差3.247415／4.413510mm、落桌／過網時間P95誤差0.845743／0.172956ms，全在20mm／10ms原門檻內；96球全部失敗仍保留，不把精度樣本當全任務成功率。

各JSON／exit code與54份入口／gate／設定／資產啟動前hash留於`m7/e_frozen/gates`，歷史CPU27證據不改寫。沒有新held-out、選模或資料擬合；真CUDA RNG尚未逐值驗證，完整runner仍NOT_RUN。詳細資料流、量測範圍與後續需授權入口見[E筆記§10.6](../../doc/notes/m7_predictor.md#106-實際驗證狀態與入口)；未回寫主工作區、發布或操作真機，無新增圖／影片。


### 2026-11-24：E-frozen smoke、三seed短比較與BC衝突檢查完成

使用者追加授權後，隔離工作區smoke PASS／exit0（session71243，1536 transitions／29.402s）；三seed7101–7103各16輪／6144、共18432 transitions PASS／exit0（session45614，187.301s）；共同7531／Z2.3、32球×10策略=320episode PASS／exit0（session65234，152.094s）。秒數包含各程序內建評估；smoke與正式runner合計144個7201工程episode另外列帳。四次訓練的Actor／Critic確實更新，52輪記錄predictor MSE0、凍結權重不變；固定最後checkpoint，沒有新增predictor擬合、選模或held-out。歷史CPU27與GPU五關證據保留，預訓練3060步／2.641s另計。

共同題目合法回球D0／7／0、E-own-initial31／0／0、E-final32／3／0；碰拍分別32／7／32、32／31／1、32／3／15，各分母32，目標命中全0。7101初始已接近上限，PPO只多1合法；7102對D少4合法且自身碰拍減28；7103碰拍多14仍0合法。描述性E35/96對D7/96及自身initial31/96不能當96次獨立訓練，**工程PASS，跨seed穩定效用尚未證實**。所有失敗與合法球落點樣本數分開報，三seed訓練19個arm_fault及1個illegal_hit保留。

E-final提示輸入覆蓋2399/2783=86.20194%、對教師RMSE9.525685mm，首次提示83.333ms；這是送進Actor的可用欄位，未證明Actor因果依賴提示。對D比較仍包含135維輸入、較長歷史及額外預訓練。獨立只讀權重／指標／來源重算PASS／exit0，D逐球離散結果重現、落點分量最大差2.861μm。證據為本地e_frozen的training、paired_evaluation、verification及experiment_execution；詳[E筆記§10.7](../../doc/notes/m7_predictor.md#107-三seed短比較提示已準確策略效用仍不穩定)。

隨後只讀BC衝突檢查完成：五份共用文件dry merge僅ROADMAP一處追加衝突，未見已快照共用程式文字衝突；但E改動source字典鍵／env／predictor及runner／CLI，舊BC嚴格歷史receipt比對會拒絕。主線probe與recovery檔案必須保留，不能整包覆蓋或改寫hash繞過；後續需明確當前來源相容性證據及無訓練重播，整合本輪NOT_RUN。BC2/5、4/5、4/5共同gate仍FAIL。詳[§10.8](../../doc/notes/m7_predictor.md#108-與主線bc的衝突文字可合不代表歷史驗證可直接沿用)與本地conflict_audit.json；未回寫主線、合併、commit／push或操作真機。

### 2026-11-24：E-mean同維度／同初值比較與前置相容性診斷

使用者另行授權獨立E-mean比較，在新隔離工作區保留Actor135／Critic200、D reward／控制、Z2.3／Y±.015m與正式三seed各6144預算。常數只取原train6101的128serve／4868窗口教師label，先各serve平均再等權跨serve平均；顯式mean模式不建MLP、optimizer或RNG，沿用五點history／valid／reset及checkpoint契約。每seed一次官方ActorCritic建構，先逐值核對全部Actor／Critic／std與固定E-frozen initial，再載入。新CPU建構後RNG SHA與當前CUDA保存可驗；舊learned initial沒存CUDA RNG，歷史GPU完全相同仍不可證。

首輪CPU39/39 PASS、零skip／failure／error，session76354 exit0，test1.656s／含launcher wall3.539s。該來源frozen2、lifecycle16、parity16、control2、combined16、mean2六項GPU程序皆PASS／exit0；mean實測CUDA RNG保持、三seed全初始張量、same-step／invalid／gap／reset／checkpoint及來源一致。新的歷史相容入口`mean-compat`預定zero＋固定learned initial／final共224主要回合，首次在learned initial7101的索引7（batch0／env7，第8球）合法落點分量差10.2519989μm，超過原10μm容差，FAIL／exit1，未完成224回合；八份首輪CPU／GPU嘗試及log封存`initial_checks/`，原負結果不改寫。

兩個新GPU診斷程序完整退出：without-D64回合／31.335s再現同一落點超限、32個learned分類相同，checks FAIL；with-D96回合／53.723s的zero／D7101／learned initial7101分類與已記錄落點逐值重現，checks PASS。兩者exit0只表示診斷完成；實測前置評測順序敏感，尚未證明物理機制。PPO前只修正mean評測器與實驗規格，保持歷史zero→D7101→learned initial／final7101→D7102→…順序，不改10μm門檻、物理、reward、控制或訓練預算。D96上下文另列：相容224主要／320實際回合，正式416主要／512實際回合；source refresh保留舊／新hash與原因。session83832於新來源完整重驗CPU39與七GPU，全部PASS／exit0；CPU test1.762s／含launcher wall3.762s，相容320實際回合report257.154s／含launcher wall266.665s，十組分類與已記錄落點逐值重現、最大落點分量差0。

完整前置PASS後，session40806順序完成mean smoke1536、三seed各6144／共18432及共同416主要＋D96=512實際回合，三程序均PASS／exit0；report時間28.560／189.067／237.066s，含launcher wall46.003／204.830／246.344s。兩組initial合法均31／0／0；learned-final32／3／0、mean-final0／0／0，各分母32，目標全0。learned相對自身initial增1／3／0，mean增−31／0／0；final碰拍learned32／3／15、mean32／32／20。主要416回合完整分類legal99／wrong_bounce165／unexpected_contact152，其餘fault0；訓練另有arm_fault1／9／10、out13／31／9，未丟失失敗。learned-final提示覆蓋2399/2783、教師RMSE9.525685mm；mean-final2425/2809、11.321587mm，暖機／invalid契約一致。

本次同維度／全初值配對後，learned在固定短比較的合法回球指標上優於mean，最大差異是保住7101初始能力；不是三seed一致從零學會或跨題目因果優勢。舊CUDA RNG未保存、動態較早歷史、三seed、上下限及順序敏感限制保留。52輪mean更新記錄全為optimizer0／常數不變，Actor／Critic均確實更新；7201工程144回合、兩次診斷64／96、combined首輪96＋重跑96、D上下文均另列，MLP歷史3060步／2.641s與未獨立計時的mean平均歸約不混算。首個失敗只有錯誤摘要，未據此補算持久逐球回合總數。

獨立CPU只讀重算最終session54652 PASS／exit0，核對定稿execution SHA、權重／RNG／來源、全分母、差值與加權提示統計，未新增GPU或訓練。舊E-frozen182/182保護項不變；主工作區273項中13項在BC並行工作改變，本任務未寫入／回復主線，不能宣稱其全部不變。完整結果、程式資料流與負結果詳[E筆記§11](../../doc/notes/m7_predictor.md#11-e-mean讓動態提示與固定平均點使用同一初值)；公開命令見[README](../../README.md)，證據另存`m7/e_mean/`。本輪完成並停止，未合併、commit、push、真機或追加訓練，既有E-frozen與BC結果不改寫。


### 2026-11-24：E-frozen／E-mean選擇性回寫主工作區

使用者授權回寫後，先保存主工作區基線，再以原E分流snapshot的SHA還原共同文件基線，僅移入E專屬差異。SPEC／DESIGN／README／notes索引保留主線新增BC段落，ROADMAP保留兩方各自追加紀錄；E專題筆記補齊隔離實驗，M7主筆記不覆蓋。原隔離工作區與歷史receipt不改寫，先前E的39項CPU／七GPU及研究評測PASS仍屬各自來源。

文件回寫完成，當前整合來源CPU兩選組mean39／BC28，合計67項PASS，零skip／failure／error、兩程序exit0；固定Docker內只跑CPU，沒有新增研究訓練。證據為`e_integration/cpu_mean.json`與`cpu_bc.json`；GPU工程frozen／mean／lifecycle／parity／control／combined／bank七關全部PASS／exit0且完成token有效，證據為`gates/execution.json`。另行CPU preflight_v2嚴格載入12模型、核對286保護檔／40歷史receipt／5份歷史來源重建，PASS但不另加單元測試數。專用重播首輪launcher缺URDF importer，FAIL／exit1、0球；改為正式入口網路設定後，bank_v2 60球與probe_v2 2球皆PASS／exit0（session41404，程序wall102.998／28.460s），分類完整重現。首輪CPU的`hashlib.file_digest`相容性FAIL與原launcher失敗均保留。主線最新BC候選2的3/4/3碰拍與PPO後1/0/1、全部0合法均保留；較早2/4/4 FAIL不冒充最新BC狀態。資料流、歷史／當前證據邊界及後續結果統一見[E筆記§12](../../doc/notes/m7_predictor.md#12-主工作區選擇性回寫與整合驗證)，本地整合證據另存`m7/e_integration/`；本輪沒有新增訓練、commit／push或真機操作。


整合相容重播的legacy BC／recovery BC／candidate2 BC／candidate2 PPO碰拍依序0/0/0、2/4/4、3/4/3、1/0/1（每格5球），全0合法；60球包括miss28／wrong_bounce16／out12／unexpected_contact3／arm_fault1。probe9378自主miss、接管wrong_bounce，當次共同前綴及舊自主終局標量差0。bank無合法落點樣本，不能把空分支的0誤差當落點驗證；probe不涵蓋全歷史軌跡，其餘probe／噪聲／訓練續跑NOT_RUN。15份程式類檔與95份歷史證據已轉入，286份主線舊證據不變；獨立CPU只讀重算`verification.json`亦PASS／exit0（session33503），核對527個輸入SHA、分母、分類與probe trace，沒有新增物理回合或訓練更新。入口`tests/pace_integration.py`及execution receipt精確命令見筆記§12，舊訓練來源validator維持嚴格，本次沒有新增訓練或發布。


2026-11-24：依使用者指示，後續優先從單拍任務定義回球獎勵與PPO時間尺度，取代先掃噪聲的建議。只讀既有程式／上游／332回合紀錄，確認A欠缺失敗球路的出球品質排序、平滑成本約−.001/回合，且gamma≈.948使1s後獎勵只保留約7.7%。這是設計適配問題的線索，不是新判定PPO原碼錯誤。整半桌離拍品質、折扣及BC保留方案尚未凍結；詳見[任務設計討論](../../doc/notes/deepmind_ppo.md#從單拍任務重審獎勵與ppo2026-11-24設計討論)。本輪無程式／配置修改，未跑新測試、重播或訓練；原驗收與失敗保留。


### 2026-11-25：M7研究進展整理至個人main

本次使用者授權push，依發布偏好以個人倉庫main承接已合入的前次基線。發布範圍包括C／D與三種predictor接合、DeepMind可行性與固定五球研究、BC恢復／短PPO、工作空間診斷及相應研發筆記。BC初始化3/4/3、PPO後1/0/1且合法全0等負結果保留；新的任務導向獎勵與PPO配方仍屬設計討論，未實作或訓練。

發布前重新核對既有整合驗證的527個輸入SHA全部一致，沿用67項CPU、七GPU工程關卡及已完成相容重播，不新增訓練或物理測試。改動檔的Python／JSON／Shell語法及Git差異檢查通過，17張新增代表圖均被文件引用。原始資料、模型、影片與本地PLAN維持runs／本地目錄，不納入Git；上游資產與原檔未改。本次目標為origin/main，團隊倉庫仍依成熟後同步的既有規則處理。
