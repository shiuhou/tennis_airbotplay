# AIRBOT Play 仿真設計

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

## PACE學習接口的方向（尚未實作）

M6／M7沿用六軸關節驅動、固定工具和球任務。預期資料流為「可部署的球歷史／本體觀測 → 可選learned predictor → PACE-style policy → 六維有界關節位置residual → 驅動 → 物理步進」；physics predictor提供訓練引導或監督目標，真值只供teacher／critic及驗證。各觀測欄位、控制頻率與動作縮放仍須於實作時逐項核查。

M5的「逆向落點 → IK／Jacobian → 定時軌跡」保留作外部基線，不插入policy每次推理。完整規劃耗時與policy推理耗時分別量測；本輪未建立RL環境或啟動訓練。改成分層規劃控制、技能庫或物理殘差模型屬後續候選，須由基線實驗支持。
