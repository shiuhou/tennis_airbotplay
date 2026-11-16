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

工具視覺保留 CAD 三角網格，碰撞使用各自單一保守凸包，與離線預檢採相同來源；PhysX cooking 可能簡化凸包。預檢只排除原相鄰裸臂、link6／fixer 和 fixer／paddle 的配合對，仍檢查 link6／paddle 及 link5／工具等其他配對。工具模式將實際碰撞形狀的 contact offset 設為 0.2 mm、rest offset 設為 0，以配合名義腕部淨空；生成網格若被匯入器設為 instance，先解除碰撞子樹的 instance，再設定並檢查形狀存在。

重力前饋直接取自含工具的 PhysX articulation，沿用既有六軸驅動限制；載荷不經額外姿態寫入或視覺跟隨補償。裸臂仍使用原 URDF 與原碰撞偏移設定。
