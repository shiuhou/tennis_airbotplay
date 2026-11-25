# AIRBOT Play 仿真設計

## 獨立相機標定

預設交接路徑先驗證出廠參數。`factory-check`僅對validation影像估板姿態，不重估K／畸變；`alignment-check --model-source factory`沿用同一出廠模型，只有顯式選estimated才讀自行估計內參。兩種彩深報告分檔，保留可比較基線。原廠檢查不依賴intrinsic／stereo求解先完成。

`scripts/camera_calibration/`提供主機Python入口與隔離依賴。`camera_official.py`按configs/orbbec_examples.json下載／驗證固定commit的官方原檔與LICENSE，直接執行裝置資訊、參數、模式、錄影和回放範例。`camera_recording.py`只開PlaybackDevice，重用官方sidecar載入、回放設定與彩色轉換，補本工程PNG／CSV、出廠參數JSON、來源雜湊和獨立資料切分；不維護自寫即時相機控制。`camera_calibration.py`負責離線OpenCV驗證與可選重標，與機械臂隔離。舊inspect／capture及camera_device.py已退役。

錄影BAG與sidecar成對保存；同session的裝置／韌體／SDK／profile／可見幾何模式須一致，同一錄影禁止重複匯入或跨fit／validation。匯入原始callback不重套官方顯示濾波，depth保留uint16與每幀尺度；回放host時間不冒充採集時間。官方parameters只列印預設模式參數，正式出廠JSON依錄影profile產生。實際BAG整合需接機短錄驗證，尚未作硬體PASS聲明。

ChArUco設定集中於`configs/camera_board.json`；PDF圖案已驗證為11×8、ID0–43與legacy排列，相容DICT_4X4_50，原始字典後綴不能唯一反推。init要求實測長度，板與CSV雜湊記入結果，補拍後需重求相關內參。資料依session、batch、frame_id、stream分層；出廠與自行估計分檔。

相機外參使用`p_to=R@p_from+t_m`，SDK毫米轉公尺；不支援的畸變模型不默認為OpenCV Brown。Depth→RGB檢查由RGB板姿態及出廠深度模型建立平面，按射線交平面選板內ROI，再評估實測點殘差；獨立量距檢查另用使用者量得的軸向Z，避免把預測當真值。修正只寫離線候選，不改裝置。硬體精度需實機證據，合成測試僅驗證工程邏輯。

## 來源與座標

選用官方 MoveIt2 Humble 描述的六軸 URDF 鏈和七份 STL。DISCOVERSE force 模型用於運動鏈交叉檢查與仿真力矩基線。其 default 模型及解析 IK 有不同關節局部座標與 TCP。Isaac Lab 2.1.0 的 UrdfConverter、Articulation、ImplicitActuator、DifferentialIKController 和 ContactSensor 提供導入、物理與控制接口。來源提交、授權、差異、硬體 SDK 版本與實測靜態比對見 [SOURCES](SOURCES.md)。

固定底座無旋轉，世界位置 (0, 0, 0.02) m，地面上表面 z=0。關節順序為 `joint1`…`joint6`，初始指令全零，該模型在零位為腕部奇異姿態。所有 API 使用 SI；姿態四元數順序 wxyz。FK 與 IK 在 `base_link` 座標系。Jacobian 上三列為 TCP 線速度、下三列為角速度，均在 `base_link` 表示。

`T_link6_flange` 取自官方連接座固定關節的安裝位置，平移 (0, 0, 0.0865) m，旋轉為單位矩陣。裸臂 `T_flange_tcp = I`。兩個座標由軟體計算。自製末端可透過 `Kinematics(tcp_transform=...)` 傳入 flange→TCP 的剛體變換供 FK／IK／Jacobian 使用。安裝於出廠連接座時，需要先核對連接座與零件的安裝變換，並獨立驗證幾何及負載。

## 驅動與控制

第一次啟動以 Kit URDF Importer 2.3.14 由本地 URDF 轉換固定底座 USD，寫入本地執行目錄。Lab 2.1 未暴露慣量匯入開關，轉換器覆寫設定並顯式匯入 URDF 慣量張量。回歸比對 PhysX 的連桿質量、質心與慣量（裸臂七個，工具模式九個）。DT 為 1/240 s，開啟重力與 PhysX 自碰撞。位置與速度目標同時送給隱式力驅動，另透過 Isaac Lab effort target 加入 PhysX 依當前姿態回傳的重力補償。DISCOVERSE force 的 10／10／10／5／5／5 Nm 分別限制隱式驅動和補償項，兩者總和沒有另作限幅。剛度、阻尼均是仿真調校值，見 `src/airbot_tt/config.py`。指令峰值 0.15 rad/s、0.30 rad/s²，PhysX 速度上限 0.40 rad/s；URDF 保留官方文件的速度上限作為模型上限。

五次 rest-to-rest 插值提供解析的每段最短時間；指定時間須大於或等於該最短時間。限時速度指令用加速度受限的對稱線性 ramp，達到指定速度後持續至減速階段，再保持終點。回饋記錄包含實際與目標位置、速度、目標加速度與時間。互動模式可用 `trace FILE` 匯出最近一次路徑的樣本。長路徑採等步距降採樣以限制單次記錄為 14,400 筆以內，並保留終點。求解 Cartesian 目標時，先用 Isaac Lab 姿勢 DLS 在 URDF FK／Jacobian 上疊代，必要時用 SciPy 有界局部求解。IK 回代門檻為 0.2 mm 與 0.002 rad。

## 碰撞與生命週期

凸包預檢使用官方 collision STL，在每個姿態以 AABB 粗篩與線性可行性檢查非相鄰連桿和靜態方盒，另取最低點檢查地面。清距 1 mm，路徑採樣每軸每次至多 0.01 rad。運動中每八個物理步檢查實際位置；每步檢查 ContactSensor，峰值超過 5 N 或離散幾何發現碰撞即鎖定。相鄰連桿關節界面不納入自碰撞預檢。PhysX convex cooking 與離線完整凸包可能有差異。離散結果只代表已檢查姿態。

Stop 以實測位置的驅動目標保持並要求 Reset。Reset 向固定底座關節寫入零位置與零速度，清除計畫和故障。CLI 結束時先解除 Lab 的 SimulationContext timeline callback，再以 Kit 快速關閉。啟動器用每輪唯一 token 和進程退出碼驗證這輪 PASS 報告，避免接受舊報告或在 Kit 啟動／關閉失敗時回傳成功。

## 固定球拍工具

`configs/racket.json` 為使用者確認的質量及分布設定，透過 `--tool-config` 啟用。`scripts/build_racket_assets.py` 使用 CadQuery／OCP，從已審查裝配 STEP 產生公尺 STL、CAD 體積質心、每公斤慣量及候選 TCP；資料在 `assets/derived/racket/geometry.json`，包含來源與網格 SHA-256。執行時不需要 CadQuery。

`tool.racket_model` 以設定質量乘 CAD 每公斤慣量，建立本地衍生 URDF；原七個連桿及上游資產不改動。`link6 → fixer → paddle` 為固定關節，關閉固定關節合併，讓 PhysX 保留各自的剛體、質量、COM、慣量及接觸感測。兩個工具連桿座標與 link6 相同，網格直接採已配合的 link6 SI 座標。總末端載荷為 0.247 kg，含螺絲；螺絲只計入固定件的均勻分布，不另畫碰撞。

Racket TCP 在 link6 的位置約為 (−0.000212542, 0.007352473, 0.271280851) m，精確姿態與位置以幾何資料為準。`paddle/RacketTCP` 是具名座標框架，不額外建立無質量剛體。控制模型使用 flange→TCP 變換，TCP 回饋由 PhysX 的實際 paddle 剛體姿態計算，因此固定關節誤差會反映在 FK 驗收。

工具視覺保留 CAD 三角網格，fixer 碰撞採單一凸包，球拍在 link6 Z=205 mm 切成拍柄／拍頭兩個凸包，與離線預檢採相同來源；PhysX cooking 可能簡化凸包。預檢只排除原相鄰裸臂、link6／fixer 和 fixer／paddle 的配合對，仍檢查 link6／paddle 及 link5／工具等其他配對。工具模式將實際碰撞形狀的 contact offset 設為 0.2 mm、rest offset 設為 0，以配合名義腕部淨空；生成網格若被匯入器設為 instance，先解除碰撞子樹的 instance，再設定並檢查形狀存在。

重力前饋直接取自含工具的 PhysX articulation，沿用既有六軸驅動限制；載荷不經額外姿態寫入或視覺跟隨補償。裸臂仍使用原 URDF 與原碰撞偏移設定。

## 球物理參考場景

`verify_ball.sh` 以三個獨立程序執行 `--mode ball --ball-hz 240/480/960`；各自在固定 Isaac 映像啟動場景，由 `BallTestbed` 建立球、桌、網與靜態 CAD 拍面。世界原點在桌中心下方地面，X 跨網、Y 橫向、Z 向上，桌頂 0.76 m，網位於 X=0。球拍參考 TCP 位於 (−1,0,1.1) m，TCP X 朝世界 +Z、TCP Z 朝世界 +X。此測試安置不變更原裝配與三孔關係，也不代表機械臂可到達該位姿。

阻力為 `F = −0.5 rho Cd pi r² |v| v`，每個物理步依當下世界速度計算。Lab 2.1 的 RigidObject 外力 API 只接受局部座標，因此以當下球姿態反旋轉後設力，再 write/step/update。重力由 PhysX 積分。初始化／每次 trial reset 寫入球姿態與速度，其餘步驟不寫狀態。球與場景均啟用 CCD。

桌、網採固定的 kinematic 剛體以支援逐對接觸感測，參考拍面是靜態碰撞，球的材料 restitution=1、與表面採 min combine，使各表面設定直接決定法向恢復係數；摩擦預設零以隔離法向回彈驗證。球網為 2 mm 剛性薄板，無網布變形；零旋基線不模擬 Magnus／旋轉傳遞。係數屬未標定測試假設，不是實物材料量測。

固定參考保留原 CAD 三角面；`--ball-collision compound` 使用與可動球拍相同的兩塊凸包，對照法向出球與落點。整拍凸包曾把厚柄與薄拍頭連成斜面；205 mm 分割排除柄端過渡曲面，保留真正擊球平面。分割只改碰撞，視覺、質量、慣量、TCP 及三孔安裝不變。

## 單拍任務

`SingleShot` 將 `BallTestbed` 接入既有 `AirbotSimulation` 的步進，沿用六軸驅動、回饋與停止保護。設定見 `configs/single_shot.json`，單拍場景採480Hz（裸臂／工具Foundation仍為240Hz）；底座直接鎖固於己方桌端中軸線，base_link 原點為 (−1.275,0,0.760) m。底座後緣 X≈−1.35923 m，距桌端 −1.370 m 約10.77 mm，底面貼桌頂，沒有架高支座。初始 TCP 約 (−0.800,0,1.120) m，拍面法線約 (0.766,0,−0.643)，以新站位重求 IK；理想鎖固不模擬鬆動。reset 先恢復關節、讓驅動穩定，再發球並清除事件。

離線預檢和 PhysX 都只排除 base_link／table 的固定安裝接觸；其餘連桿、固定件、球拍與桌面的碰撞保持啟用。基座高度與整個底座落在桌面內由幾何測試驗證。

球的接觸感測分別追蹤桌、網及各個手臂／工具剛體。只扣除球對拍面的實測反作用力，避免合法擊球觸發一般 5 N 停止；拍柄及 fixer／arm 接觸仍是失敗。事件以接觸起始去重；過網位置由相鄰物理步插值。順序為己方反彈 → 拍面 → 網上方穿越 → 對方桌面，才算合法回球；超時為 truncated，其他結果為 terminated。

`positioning_candidate` 使用既有 IK 及五次 rest-to-rest 軌跡，驗證反應延遲、到達時間、關節限制和桌／網碰撞。當前低速限值只支援三組預先就位接球，偏球作拒絕案例；不是反應式揮拍能力。保持球拍的基線會把球反彈回己方，應判失敗。M4 預測已接入（見下節）；M5另增有速度的擊球規劃與scripted基線，見後節；此處M3的預先就位結果不變。

## 球軌跡預測

`prediction.predict_ball` 是不依賴 Isaac 的連續預測器。輸入世界座標 `[x,y,z,vx,vy,vz]`、預測時間與球參數，使用 DOP853 積分重力和完整三維二次阻力；求根取得事件時間及球心位置。桌面接觸平面為桌頂加球半徑，法向速度乘恢復係數並反向，預測到下一次落桌。`first_table_event='landing'` 用於已測到碰拍之後。

預測器不保留跨發球歷史。`SingleShot.forecast` 從當前實測狀態預測；未來拍面／手臂碰撞不在模型內，驗收將碰拍前後分段。球網用來球側的有符號接觸面偵測，避免一步跨過薄網漏根；網頂與桌緣擦碰回報 unsupported。位置一律是球心，不與桌面接觸點混用。

`--mode prediction` 使用無球拍的 PhysX 球桌場景，按固定24例獨立步進，以接觸力及實測位置／速度抽取參考事件。按事件種類各自計算P95，防止大量容易案例掩蓋反彈後誤差。球參數未擬合，無阻力解析測試另行驗證公式。

240Hz對M4的反彈後落點時間P95誤差為11.883ms，未過10ms；相同案例改480Hz後為5.633ms。單拍場景以設定的physics_hz啟動，控制時間、感測更新、穩定等待與球步進使用同一dt，沒有在運行中切換步長。未改動物理係數或驗收門檻。

## M5 有速度的擊球與 scripted 基線

`StrikeTrajectory` 在正規化時間上求兩段五次多項式：靜止起點 → 指定時間的 `(q_hit, qd_hit, qdd=0)` → 靜止終點。每個關節以多項式導數的實根找位置／速度／加速度極值，拒絕中段越界；碰撞採樣實際曲線，時間間隔同時受步長和峰值關節速度限制。擊球態速度不歸零，兩段位置、速度、加速度連續；沒有jerk限制。

`planning.plan_return` 從預測的己方反彈向後取候選擊球時刻；`return_velocity` 用DOP853加局部最小平方求解能在指定飛行時間到達目標的出球速度，再以M4預測器拒絕撞網／提前落桌。無摩擦近似下 `n = normalize(v_out-v_in)`，拍面法向速度 `u_n = n·(v_out+e v_in)/(1+e)`。IK求TCP於球心減半徑乘法線的位置；`a=nᵀJ_linear`，最小範數關節速度 `qd=aᵀu_n/(a aᵀ)`。不額外強迫切向速度或角速度為零，兩者由關節解產生並實測檢查。這不是全域可達性證明。

`configs/strike.json` 獨立設定M5準備姿態、1.5 rad/s／8 rad/s²指令上限、2 rad/s仿真上限、50 ms反應等待及0.3 s減速；M5物理步長獨立於M3／M4。物理係數、桌面固定、工具及M3設定不變。此時指令限制仍低於URDF速度限制，沒有調高驅動力矩或PD增益。

目前每個完整候選約需數秒，主要成本是曲線碰撞預檢，不能放進約0.6 s的來球時限。公開回球測試因此明確採scripted路徑：發球前只用同一個名義來球，分別為三個目標算一條路徑；來球開始後等待50 ms，只選用已預檢路徑，不對開發或驗證樣本逐球重算。`start_prepared_strike` 檢查模型／碰撞檢查器身分、已採樣、實測靜止起點及速度上限，避免從錯誤狀態套用軌跡。這是限定來球下的模型規劃基線，並非已解決在線重規劃。

M5的來球、開發／驗證種子與數量固定於同一設定；三個目標共用準備姿態。驗證集不用來調軌跡；所有發球含漏接、拒絕和故障均在分母。測試前先對三條實際路徑做空揮拍，再以獨立短程發球測相對碰撞公式及240／480／960Hz收斂；480Hz失敗後追加1920Hz對照，未放寬門檻。較快揮拍不能沿用低速測試的PASS。

## PACE學習接口：M6離線與M7初始整合

原版接口與時序依[PACE完整資料流](../../doc/notes/pace_pipeline.md)核對：learned predictor獨立監督更新，PPO不反傳到predictor；教師含啟發式fallback，原版還有預測回填延遲與reset歷史問題。M6以獨立接口修正資料／推論生命週期，其離線驗收不包含PPO；M7初始接合見本頁後節。

M6／M7沿用六軸關節驅動、固定工具和球任務。預期資料流為「可部署的球歷史／本體觀測 → 可選learned predictor → PACE-style policy → 六維有界關節位置residual → 驅動 → 物理步進」；physics predictor提供訓練引導或監督目標，真值只供teacher／critic及驗證。各觀測欄位、控制頻率與動作縮放仍須於實作時逐項核查。

M5的「逆向落點 → IK／Jacobian → 定時軌跡」保留作外部基線，不插入policy每次推理。完整規劃耗時與policy推理耗時分別量測；本輪未建立RL環境或啟動訓練。改成分層規劃控制、技能庫或物理殘差模型屬後續候選，須由基線實驗支持。


## 真機準備接口（僅規劃）

H0–H4先用相符版本的官方服務／SDK／範例，確認副作用後才補最小只讀記錄層；不另建通訊協定、FK或控制器。主機盤點不建立AirbotClient，H1記錄層與取得Lease的運動層分開；尚未實作。既有kinematics／tool負責仿真對照，SDK end pose與Racket TCP的語意須核實。完整關卡、資料時序與未決契約見[真機筆記](../../doc/notes/hardware_bringup.md)。

## M6離線實作

`pace_guidance.py`包裝既有DOP853作CPU參考教師，明確區分有效反彈後apex與無效狀態；`return_teacher`從離拍末態求落點／過網。`ReleaseLatch`接收合法拍面接觸、分離方向及TCP速度，只在首次有效離拍評分；M7才接實際RL事件。引導reward輸出raw／weighted／total；M6階段只做離線驗證，M7的成功獎勵、正則及C接合見下文。

`pace_predictor.py`校驗58份來源後直接import官方MLP，沒有複製trainer。資料五點／83.33ms、同serve及固定間隔；獨立RNG的Adam／MSE訓練與no-grad推論分離。`PredictionHook`每環境清history、在回傳前完成prediction；`actor_frame`建立23維或含預測的27維單幀，完整五幀actor／critic尚未建立。checkpoint保留來源與球參數hash，載回不能把未訓練模型誤作可用。

`tests/pace_m6.py`沿用BallTestbed採獨立無球拍PhysX軌跡，以垂直速度過零插值作真實apex；訓練／驗證／heldout分seed。`scripts/verify_m6.py`先按validation選checkpoint且通過開發門檻才開heldout；同資訊基線以五個位置二次擬合估速，跨反彈弱點分段報告。批量GPU教師、近期在線buffer和官方RL wrapper整合留M7，不在離線驗收中冒稱完成。

## 研究工作流

`verify_groups.sh` 分別調度主機標準庫、固定 Isaac image 與相機 venv；`verify.py` 回報測試／skip 與退出碼。`research.py` 保存來源與輸入雜湊、條件、啟動命令、結果及產物雜湊，獨立比較只使用標準庫；共同條件與方法變因分開。它不是 trainer、resume 或任務管理器。M6 以 training.json 綁定來源／設定／資料／checkpoint，train 和 evaluate 可分離；繪圖只讀結果。來源雜湊不是 dirty 程式備份，科學推論仍需事先設計。

## M7 六軸初始整合

`pace_env.py` 使用 Lab 2.1 InteractiveScene 複製共用 URDF／BallTestbed，環境局部座標、隔離接觸與 tensor 事件。`pace_control.py` 的事件語義與原 ShotEvents 對照；位置 residual 經三次插值及每物理子步的速度／加速度限制，再送原驅動和重力補償。普通步進不寫關節狀態，reset 後依官方流程 forward 更新連桿運動學。

原先逐位置目標煞停的 limiter 未通過控制 gate，已退役；不保留作預設或隱藏回退。三次插值把要求解讀為下一策略區間末端位置，以過去位置要求估終點速度；不是讀取未來球路或呼叫 M5 planner。

初始 A／B 使用相同 115 維 actor、200 維 critic，無 MLP 提示。發球 reset 時 CPU 參考教師求 apex，物理子步依相位／剩餘時間遮罩並累積引導；不宣稱已完成 batched GPU teacher 或在線 predictor。critic 教師資訊兩組共用。

`train_pace.py` 校驗固定 PACE 原檔後直接載 ActorCritic／PPO／Storage，僅處理 AIRBOT rollout、terminal-value bootstrap、保存與評測。使用 no_grad 讓 Isaac 狀態緩衝可在不同階段更新；不在可變仿真緩衝上混用 InferenceMode。評測只取 actor 均值，按預先指定的 env／serve 收齊結果。原版更新器不改寫；觀測歷史、動作 clip、reset 與稀疏事件屬明確機械臂適配。

首次更新診斷由 `tests/pace_update.py` 使用原 PPO 的深拷貝與 optimizer hooks 完成，不改上游 update。各候選共用同一 rollout、初始權重與更新 RNG；先核對重算的 mu／sigma／log-prob，再記每次 optimizer step 前後的整批 KL 和裁切後梯度範數。診斷以獨立來源／結果留存，不寫回訓練權重。訓練的可選初始 learning-rate 覆寫會進入實際 cfg、protocol 和 checkpoint；reward、action、teacher 仍使用原設定。

`ShotEvents`／`TensorShotEvents`共用`rally_net_contact_v2`規則識別：net只記接觸，跨網判定先於同子步落桌，取消net接觸及截面高度的立即終止。tensor保留歷史outcome編號3以讀舊報告，現行不再發出該結果；net_touched獨立重置並寫入episode。PhysX網碰撞材質／幾何不變，解析預測器的net_hit／unsupported仍表示預測範圍終止，不是任務失敗。`pace-eval`驗證checkpoint雜湊與完整cfg，使用原均值策略／7201來球，只重評不建立optimizer；報告包含規則版本、新舊分類及實測episode時長。

## 凸包碰撞預檢與單變因來球對照

`_Hull.overlaps` 以原半空間與 clearance/2 膨脹，最大化共同平面餘量ρ（`A x + ρ ≤ b`）。最優ρ≥−1e-7m保守判作碰撞；非最優／非有限結果拋錯。使用SciPy官方HiGHS，無失敗放行或重試分支。一般1mm、工具手腕0.5mm裕度不變；ρ不是歐氏距離。此形式取代零目標可行性LP，以避免已重現的分離密集凸包Unknown；數值／完整M3驗證見M7筆記第13節。

M7短對照允許透過命令列覆寫發球Y半範圍，記入有效設定與checkpoint，只改PaceEnv記憶體副本，不改M5檔案。訓練可選A／B／C／D子集；比較範圍時固定A。最終評測按同一seed重設各模型及zero，記錄合法落點XY，故障不剔除；效用門檻以SPEC為準。

M7.4的發球中心高度覆寫由`--pace-serve-z`進入有效cfg及checkpoint，PaceEnv只改記憶體內的serve_position_m[2]，重新評測讀回原值。這是版本化任務輸入，不改M5設定、控制包絡或教師定義。動態可達性見證利用既有球預測／IK／Trajectory，再走正常關節動作介面；只供開發驗證，不是Actor的一部分或新的引導點。

## C一次性離拍評分

`TensorReleaseLatch`在物理子步追蹤有效接觸→解除接觸且相對分離，依env發出一次release脈衝；非法、50ms等待超時為終止狀態，partial reset只清指定env。C在脈衝時呼叫既有CPU `return_teacher`，保留原落點／網高函式與事件權重；CPU求解只在離拍發生時運行，不把它冒稱為批量GPU教師。

`episode_return`保存單次引導總額，策略步只領取該步前後差，避免跨20個子步或後續幀重複發分。記錄離拍狀態、時刻、預測、分項及invalid原因；return teacher新增預測事件相對時間供PhysX比對。A/B分支不運行此教師，觀測及動作不變。後續接觸可改變實際結果，因此預測分數不是合法回球保證；本實作不在未來非法事件後回溯改寫已發放的分數。


D復用同一環境分支：B/D領取本策略區間的hit累積差，C/D領取return累積差。沒有另建reward管理層或改公式；release與reset完全共用。`pace-check combined`在固定C重播證據上驗證相同出球／episode、總分加法、return不變及reset隔離。D訓練要求當前combined關卡，C單獨訓練仍要求return關卡。


## E-online位置歷史提示與獨立在線MSE

`OnlinePredictor`復用官方MLP與既有`Predictor`，保存128個策略時刻的有效五點窗口；每個rollout後獨立Adam/MSE一個epoch、batch1024，只有前20次更新資格，空資料記跳過。初始化未訓練時提示為0／invalid，離線驗證模型不載入主線。模型／optimizer／局部RNG／更新計數及近期buffer進checkpoint；環境恢復時清觀測歷史。

E只在原23維actor單幀後加3D prediction與valid，形成27×5=135；critic維持40×5=200，教師及D獎勵不變。predictor輸入來自同一環境局部球位置，教師只用於MSE標籤／診斷，不能進actor。策略步以整數索引產生採樣時間，避免960Hz浮點累加破壞48Hz等間隔窗口。reset只更新指定env；正常步先predict再組obs，MSE更新後只重算最新一幀提示，不重複推入同一筆歷史。

五點predictor看83.3ms，actor五幀各帶當時prediction，使最遠資料延伸至166.7ms；D的直接五位置歷史只有83.3ms。D/E比較的是整個模組與時間資訊的增加，不能據此分離物理特徵和長歷史壓縮的貢獻。E自身initial actor另存，不假定115維D與135維E同seed初始化相同。

runner逐更新記錄hint真正使用步數、教師有效期間內的提示誤差、hit引導步數／總分、離拍／有效預測／非零return事件及分數；逐episode記首次prediction與首次使用時刻、擊球前提示步數、history不足／untrained／invalid。valid只代表可計算且至少更新過，不等於在線模型精度已達離線門檻。PPO更新前後核對predictor權重／梯度，MSE前後核對policy權重，並記實際MSE epoch和optimizer步數。

`pace-check predictor-curve`在E環境生成及更新提示，但將actor五幀各裁回原23維交給固定D權重；D不受prediction改變影響，隔離PPO與提示造成的行為變化。每條流重建未訓練predictor及相同發球RNG；每24步只呼叫原online.update一次，以固定validation量測誤差，不使用test或修改模型。這不是E聯合學習效果驗證。

`pace-check predictor-batch`復用學習曲線診斷，在同一環境資料流旁建立batch32的獨立OnlinePredictor。每輪將原buffer條目同步給它，模型／optimizer獨立；初始化權重及每輪shuffle RNG一致性均斷言，資料窗口SHA入報告。固定D不讀任何提示，batch32不影響環境。batch只在診斷實例配置改動，configs/pace_online.json維持1024；不引入第二套訓練器或改正式E設定。


`tests/pace_predictor_audit.py`提供封存推論與真資料契約診斷。交叉評測直接載入原Predictor與checkpoint的buffer，對教師／PhysX分開計分；按階段以已保存速度／事件標記分組，不從無metadata的舊buffer猜env。短重播僅在診斷內包裝observe／collect，同步保存真正被收集的env／serve／時刻／origin／狀態／phase，逐值對上原buffer；生產checkpoint結構與資料流不變。

E-online的固定reset教師與離線逐狀態重算教師是兩種已記錄語義。診斷保留重算失敗原因及原窗口，不以篩掉不一致資料改善模型分數。研究將在線取得準確提示與提示對策略的價值拆開；E-frozen採以下獨立模式，不取代E-online。

### E-frozen的模式與資料所有權

D不建立提示欄位；E-online與舊名稱E沿用獨立MSE；E-frozen經顯式配置載入原離線MLP，只做推論。三者共用原D教師／reward、48Hz六關節action、960Hz物理及200維critic。E-frozen的135維actor與E-online相同：五個27維frame各由原23維加prediction XYZ／valid組成；它不要求離線動作示範，也不把M5規劃器接入Actor。

凍結配置綁定模型SHA、訓練receipt、資料manifest、官方來源與球參數。載入先驗證此鏈，再取得權重；訓練receipt中的20epochs／3060 optimizer步屬既有預訓練，不能算成此次在線更新。執行時沒有predictor optimizer或近期MSE buffer，PPO只持有actor／critic參數；模型`eval`、無梯度推論及參數凍結各有不同作用，須合併驗證，不能僅靠eval宣稱不會更新。

`FrozenPredictor`復用`Predictor(inference_only=True)`與`PredictionHook`；前者不建立Adam或shuffle RNG，後者仍管理每env窗口。歷史receipt來源對照準備時的`source_snapshot.json`，當前物理／教師／官方來源另行核對；不因新增接口修改了推論檔案，就重寫歷史hash。`PaceEnv.predictor_module`共用frame路徑，runner的`set_predictor()`及`update_ppo()`明確分流；凍結checkpoint保留歷史更新計數，但不攜帶可續訓optimizer／RNG。

凍結模型在CPU建構，僅於保存／恢復CPU RNG的區間設定CPU default generator，避免`torch.manual_seed()`連帶改CUDA抽樣流；舊在線初始化語義保留。history入口同時檢查位置與時間有限，NaN時間不能被一般間隔比較放行。initial評測結束恢復抽樣RNG並重新設定正式serve序列，避免工程量測消耗之後的訓練資料。

當步球位置先減去env origin，依原時序進入獨立的五點history，再由同一凍結MLP算提示，最後組裝actor frame。窗口不足、reset、缺樣或非有限觀測沿原hook契約清理，其他env歷史不受影響。可用性只取決於可觀測history與可用模型；教師／事件遮罩用於擊球前診斷，不得反向控制Actor的valid位元。較舊的actor frame保留當時提示；凍結模式沒有MSE後回填需求。

策略checkpoint保存模式、凍結來源及權重，載回須恢復同一135維觀測契約及推論內容，不能從檔名猜測模式。環境history在新回合重新建立；此checkpoint保證模型重載與評測契約，不宣稱保存PhysX狀態後能逐步續跑。runner在首次PPO前保存自身initial，後續以權重與梯度檢查保證predictor沒有進PPO更新路徑。

配對研究的共同題目、樣本預算及判讀邊界由[SPEC](SPEC.md#e-frozen固定離線提示的獨立比較)定義。新protocol需記當前source與固定D模型雜湊；原D的環境來源已早於E與其他接口改動，故當前重評是一項相容性檢查，不能只搬移舊PASS報告。實作資料流、來源鏈、歷史成本及未執行關卡詳見[E筆記第10節](../../doc/notes/m7_predictor.md#10-e-frozen固定準確提示再問策略能否使用)。

### E-mean同維度常數與初始化配對

`FixedMeanPredictor`讀取`configs/pace_mean.json`，核對train檔、資料manifest、原cross報告的train平均點、凍結基準設定及新工作區source snapshot。常數以float32保存於`_MeanModel`的buffer，沒有可優化parameter；`_MeanPredictor`不建立MLP、optimizer或RNG，`fit_epoch()`拒絕呼叫。資料配方為每條train serve的有效五點窗口教師label先平均，再跨serve平均，避免窗口較多的serve取得較大權重。建構器只校驗既有常數／來源，不開raw validation或held-out選點。

mean沿用`PredictionHook`，由實際可觀測位置／時間維護同serve五點history；完整有效窗口才讓常數XYZ與valid=1進入原27維frame。這保留learned的暖機、缺樣、reset與五幀135維結構，差異發生在完整窗口後的XYZ內容。`PaceEnv.predictor_mode`區分mean／frozen／online；mean走不可更新提示介面，不走MSE收集或更新，critic及D reward仍共用。constant buffer、來源及模式共同進checkpoint，載回清history並拒絕其他模式或常數篡改。

`pair_mean_initialization()`在runner正常建構一次ActorCritic後，先核對所有policy state張量與固定learned initial逐值相同，再載入同一權重，不新增第二個網路而改變RNG。CPU建構後狀態SHA由同官方建構流程獨立重建並事前固定；當前CPU／CUDA在載入與initial評測前後保存／核對，正式rollout前狀態寫入新initial checkpoint。舊learned initial沒有保存當時CUDA RNG，這個缺口保留為證據限制；不能因當前流程可重建就改寫舊receipt。

`configs/pace_mean_experiment.json`固定既有learned initial／final、舊報告及共同設定。`mean-compat`使用當前程式無更新重播224個主要回合建立相容證據，正式`mean-eval`再共同評416個主要回合；兩者保留原zero→D7101→learned initial／final7101→D7102→…順序，D三seed合計96個上下文回合另列，實際分別320／512回合。這是首次10.252μm落點超限及兩次獨立順序診斷後、PPO前修正的評測契約：恢復歷史順序、不放寬原10μm門檻；只量到順序敏感，尚未證明物理機制。兩個入口共用評測與舊結果比對，既有模型／報告不寫入。新mean訓練只接受當前來源mean／compat與原控制關卡，正式短跑再要求mean smoke。研究可比較動態提示與常數提示在同維度／初值下的效用，但較長位置歷史的壓縮也隨動態提示一起被移除，並非只改單一「準確度」數字。解釋與資料流見[E筆記第11節](../../doc/notes/m7_predictor.md#11-e-mean讓動態提示與固定平均點使用同一初值)。

主線回寫的相容驗證由`tests/pace_integration.py`獨立記錄歷史到當前來源映射，核對固定模型／receipt及來源重建，再以既有bank與probe資料流重播。原BC／recovery／refinement的來源validator保持嚴格，舊receipt不改寫為當前hash；整合驗證不替換生產載入契約。最早runner未完整重建時，明確區分官方constructor／seed的全初始張量核證與歷史runner重跑。當前執行結果見[E筆記§12](../../doc/notes/m7_predictor.md#12-主工作區選擇性回寫與整合驗證)。


`tests/pace_motion.py`分獨立程序復用SingleShot原M5 prepared script與PaceEnv原zero/PPO。診斷在每個物理步旁讀取實際球／TCP／連桿狀態，接觸上升沿保存前後樣本，解除接觸與post-hit桌接觸另存；120Hz trace供繪圖。`scripts/plot_motion_comparison.py`純讀報告，重繪側視／俯視與同步3D簡圖／碰拍區放大，不參與控制或碰撞。未取得solver接觸點，因此表面點速度只是明標的幾何估計，主要比較取可靠的TCP剛體速度。


DeepMind audit是獨立資料診斷，不改訓練env。`configs/deepmind_sources.json`固定官方來源／SHA／Git blob，`prepare_deepmind.py`只下載驗證原檔；`configs/deepmind_audit.json`定本輪低旋與搜尋條件。`tests/deepmind_audit.py`做座標／分布、重用predict_ball的零旋預測、原Trajectory／PositionCommand預檢、原M5有限規劃、PaceEnv獨立重播及固定action交叉矩陣。原始資料、衍生資料與來源分開，完整證據留本地deepmind_audit，summary.json統整分母。

合法回球候選與contact診斷分開導出，保留raw omega／ID／split／來源與零旋近似，沒有production consumer。未取得合法回球時audited_candidates為空且NOT_READY；不得將資料契約相容或有效碰拍冒稱可訓練回球分布。


`deepmind_audit.py --stage return-witness`是獨立的有限回球見證搜尋：從既有train接觸案例預先選前三筆，沿原飛行／碰撞公式，只將完整姿態IK改為接觸球心位置＋拍面法向約束，允許面內轉角自由。保留原M5與production控制；候選合格仍需完整命令及PhysX驗證，不能將局部IK失敗當不可達證明。各輪協議與結果獨立保存，不覆寫歷史audit summary。


安裝工作空間診斷`tests/racket_workspace.py`重用原FK及CollisionChecker，固定Sobol採樣目前／完整關節box，完整集合額外包含目前清晰姿態，避免獨立稀疏採樣造成假缺失。正拍面CAD裁切到既有blade分界，法向外移球半徑形成潛在球心集合；只做靜態手臂碰撞及球對桌／網／地面檢查，未驗球對手臂遮擋、路徑連通或限時動態。NPZ保存q／R／TCP／normal及20mm voxel實際代表的姿態和面點索引；不能將occupancy變成整格可達。底座XYZ覆寫僅供另存幾何比較，不改正式安裝或控制配置；支撐與旋轉安裝須另建模。


`deepmind_audit.py --stage return-radius`重用return_witness_search，僅在記憶體覆寫診斷rl的action_scale，分別保存.6／1.0結果與hash。重跑.6對舊結果驗理由／誤差，兩組逐候選核對相同球態、目標、時間及法向；保留IK的q與active-bound joints，區分action box與URDF邊界。未改生產配置；局部邊界解是診斷線索，不是全域不可達證明。


固定五球探索以`pace_serves.FixedServeBank`核對官方原檔／raw-ID train split與轉換，保留raw omega但simulated omega=0。`PaceEnv`可選bank，每環境獨立五球洗牌循環，自動reset不回退人工分布，指定bank_indices評測不消耗循環；episode在reset前保存raw_id/initial_state。bank與人工覆寫互斥，無bank維持原RNG行為。runner固定A/16env/3seed/6144，保存initial及manifest，評測後恢復Torch RNG和訓練序列；訓練每步先按當前ID計數，sequence在最終評測前存檔。原terminal與PPO更新不改。

`pace_bank.evaluate_bank`用首5環境評canonical五球，等待全部首回合完成、排除padding／後續reset，必要時旁讀960Hz接觸量測，逐env首次終止即凍結。局部點速仍為幾何投影估計；實測碰拍後落桌與legal outcome分開。runner對initial/final/重載評估，測試資料與checkpoint契約不符即拒絕；通用pace-eval亦支援顯式同manifest，無隱式換資料。測量helper hash與bank gate hash保存。


### 接觸示範初始化

`pace-bc`在原固定五球環境重播腳本，於動作前複製Actor115觀測／請求action，以serve_id排除各環境第一次reset後的資料。離線MSE先按軌跡內步與joint平均，再對五軌跡等權；直接訓練固定PACE的actor子網，不採上游依賴StudentTeacher的Distillation介面。BC後的完整閉環重播及帶噪聲試驗均由Actor輸出，不混入腳本。

`--pace-actor-init`只接受已驗證三seed的五球A研究產物。與同seed原initial逐值核對後只載actor，cfg保留原控制／PPO契約，示範來源／BC配方另存provenance。PPO建構前恢復原model-construction後Torch RNG、重設bank序列與serve計數，建立新optimizer；Critic和std初值不變。噪聲診斷使用每球／seed／重複獨立的局部亂數流，使原始與BC Actor共用標準常態抽樣而不污染PPO。

`tests/pace_bc_rollout.py --feedback-audit`在評估邊界複製115維Actor觀測，從同tick已保存示範替換每幀的qd[6:12]及／或前action[12:18]。實測q／qd、真正previous_action、actor_history與PositionCommand記憶均不改；每次推論前後逐值核對。示範自身含一拍延遲和初始歷史填充，以首serve_id隔離reset；示範耗盡恢復真實觀測，缺失參考明標。原始／介入／參考完整觀測、真實q／qd、action與控制記憶另存本地，早期偏差從真實狀態計算；混合觀測只用於機制診斷，不進訓練或部署路徑。

五球修正示範分兩次啟動。三組固定時刻下，`tests/pace_recovery.py` 對既有 `evaluate_bank` 提供按 tick 切換動作來源的 policy，首 episode 的當步觀測配實際請求，依原事件及物理碰拍時刻決定訓練區間。前綴可對既有自主量測逐步核對。通過覆蓋門檻後，`scripts/contact_recovery_bc.py` 以球、來源、軌跡、步四層平均原六關節 MSE，只更新 Actor 並獨立重載閉環驗證。新權重與舊研究證據分目錄、用獨立 schema；保留既有 PPO 初始化驗證。研究入口先啟動 Isaac，再匯入隨 Kit 提供的 PyTorch。

`tests/pace_recovery_probe.py`以凍結的新Actor作兩個單球回合，重用既有接管記錄器及物理事件擷取，只計16環境中的env0首episode。原`step(auto_reset=False)`後量測正常運動state write為零；其他已終止padding可單獨reset，env0終局即停止。保存逐tick真實q／qd／previous_action／Actor請求／腳本請求及控制記憶，與三條舊成功後綴逐樣本核對，再分單位、限共同碰拍前比對。既有hash鎖定的採集與訓練來源不改；獨立protocol連到原fit及collection證據。繪圖由既有`plot_recovery_bc.py --probe`只讀產生。


四案接管擴充沿用原probe與記錄器，保留球在共同評測中的環境索引；只計指定env首episode，其餘padding排除。成功後綴以採集Actor SHA／raw ID／切點識別，避免同時刻的不同實際狀態被錯當重複。`tests/pace_bc_refine.py`直接重用原資料加權、PACE ActorCritic、完整五球評測及配對噪聲；792筆資料另存固定來源及逐軌跡質量。候選1的固定Adam與候選2的官方CosineAnnealingLR僅在此研究入口選擇，未改生產PPO預設。兩配方保留以重現受控比較，不繼續新增未驗變體。

`airbot-contact-refinement-v1`明確描述變更後的擬合配方。初始化驗證重算三seed的五球及重載結果、核對最後update數、資料／protocol／權重／原initial及目前來源SHA；原BC v1繼續保留原配方與來源限制。通過後仍由原`--pace-actor-init`只載Actor，PPO使用fresh optimizer。歷史證據的逆差異只用於重建精確舊來源，不把歷史PASS改認為新來源PASS。詳細資料流、限制與命令見[自主修正BC筆記](../../doc/notes/deepmind_ppo.md#自主修正bc最多三個候選)。
