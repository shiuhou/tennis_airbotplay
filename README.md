# AIRBOT Play · Isaac Lab

AIRBOT Play 六軸固定底座仿真，支援裸臂與固定球拍工具。包含關節位置及限時速度控制、姿態 FK／IK、TCP、Jacobian、多路點軌跡、非零速度定時擊球、限定三目標scripted回球、靜態環境碰撞預檢、接觸監測、Stop、Reset 及 Kit 離屏錄影。

研究主線是將PACE的預測、物理引導獎勵與關節策略適配到固定六軸AIRBOT，邊實作邊理解。M0–M5提供仿真與scripted比較基線；M6已完成離線物理引導與predictor驗證，M7已接通PPO並進行小規模A／B診斷，即時規劃加速不作為其前提。

過程中的原理、踩坑與驗證邊界整理於 [研發知識筆記](doc/notes/README.md)。論文、開源專案、閱讀狀態及授權統一記錄於 [研究參考與來源](.agents/docs/SOURCES.md)。

## 啟動

需要 Linux、NVIDIA GPU 驅動、Docker、NVIDIA Container Toolkit 和本機的固定映像。此工程使用既有的 Isaac Lab 2.1.0／Isaac Sim 4.5 映像。啟動時由手臂 URDF 產生本地 USD。首次轉換可能需要 NVIDIA Kit extension registry。第一次錄影會建立 RTX 著色器快取。

```bash
docker image inspect sha256:9c5b00fb775c65913f78f22eadb56912853c1f9f1c38828e53fb99369e29c49e
./run.sh --headless --mode state
./run.sh --headless
```

完整回歸需要主機的 Python 3、`ffprobe` 和 `ffmpeg`。

```bash
./verify.sh
```

回歸依序執行純數值測試、headless 運動與拒絕測試、PhysX 真實接觸測試、Kit 離屏影片及影片校驗。預設證據寫入 `.agents/runs/foundation/` 的 `verify.json`、`contact.json`、`demo.json` 與 `airbot_play.mp4`。可用 `--report` 或 `--video` 指定倉庫內的其他路徑。退出碼 0 表示該輪產生 PASS 報告。另一台主機需預先具備同一映像 ID。連同環境與來源版本見 [設計](.agents/docs/DESIGN.md) 和 [來源](.agents/docs/SOURCES.md)。

## 控制命令

所有關節向量依 `joint1`…`joint6`，角度 rad、速度 rad/s、時間 s。姿態在底座座標系中使用 m 和 wxyz 四元數。下面可逐行送入互動模式，運動命令之間要等回饋顯示 `active: false` 且手臂已穩定。

```text
state
fk
jacobian
joint 0.10 -0.30 0.40 0.15 -0.20 0.10
velocity 1.0 0.05 0 0 0 0 0
trajectory examples/trajectory.json
trace .agents/runs/foundation/my_motion.json
collision
stop
reset
quit
```

- `fk`、`jacobian` 可帶六軸角度。Jacobian 回傳 base frame 的 6×6 矩陣，線速度三行在前。
- `ik x y z qw qx qy qz` 顯示局部解、回代誤差與終點碰撞狀態。`move_pose` 採相同格式並預檢到達該姿態的關節路徑。
- `trajectory FILE` 讀取 `waypoints_rad` 和可選 `durations_s`。第一段從當前測量角度開始。若未指定時間，依速度及加速度限制推導；時間不足的方案會拒絕。
- `velocity` 參數為總時間及六軸速度。總時間包含加減速 ramp。旋轉速度與積分終點都會預檢。
- `trace FILE` 匯出最近一次路徑的時間、角度、速度、目標加速度和誤差樣本。
- `stop` 清除路徑並鎖定保持。從 `state` 查看故障原因，`reset` 回到零位後才可下新命令。
- 加 `--obstacles examples/obstacles.json` 載入在世界座標的碰撞方盒。每盒含 `name`、`center_m`、`size_m`。

## Flange / TCP

`link6` 到裸臂 flange 沿 `link6` 局部 +Z 平移 86.5 mm，無相對旋轉。裸臂 TCP 與 flange 重合。狀態包含 flange 與 TCP 在 base/world 的位姿。自製末端的 TCP 可在 Python 以 `Kinematics(tcp_transform=T_flange_tcp)` 設定。出廠連接座的安裝變換與實物零位須按手上的機械臂核對。Cartesian 指令執行預檢過的關節路徑，碰撞檢查採離散樣本。需求與邊界見 [SPEC](.agents/docs/SPEC.md)。

## Fixer 與球拍裝配

目前來源為使用者提供的 `fixer_airbot(2).STEP`，原樣保存於 [fixer 原檔](assets/fixer/fixer_airbot.STEP)（SHA-256 `d77d494aa49798720320b591ac28fcef114d2b8f6031b8903d1fef98b1c1da46`）。球拍使用 [既有 STEP](assets/paddle_for_fixer_38836ada.STEP)，柄尾置中靠止擋。

- [裝配 STEP](assets/derived/fixer_fit/airbot_j6_fixer_paddle_fit.STEP)：J6、fixer 和球拍，採 link6 毫米座標。
- [配合與剖面圖](assets/derived/fixer_fit/airbot_j6_fixer_paddle_fit.png)。

CadQuery 2.8.0／OCP 7.9.3.1 驗證三個 Ø4.20 mm 孔對準 J6 的 Ø24.30 mm 孔心圓，安裝面貼合，三個有效實體兩兩交集體積為零。後方套筒已移除；依官方 link5 網格的連續 360° 軸向支撐界，fixer 與 link5 名義淨空至少 0.946981 mm。球拍柄尾接觸面積 489.21 mm²，應先固定 fixer 再插入球拍。

J6 的 Ø4.00 mm 孔徑及 4.00 mm 孔深僅為 STL 參考，使用者已確認真機鎖固；實物螺紋、加工公差與強度尚未獨立驗證。CAD 配合結果與工具物理模型的限制見下節。

## 固定件與球拍的物理仿真

使用 [已確認的載荷設定](configs/racket.json) 啟用工具，預設不帶參數仍為裸臂：

```bash
./run.sh --headless --mode state --tool-config configs/racket.json
./run.sh --headless --tool-config configs/racket.json
./verify.sh --tool-config configs/racket.json
```

工具模式保留六個驅動自由度，新增 fixer、paddle 兩個由固定關節連接的剛體，包含碰撞、質量、COM、慣量與接觸感測。固定件含螺絲 77 g、球拍 170 g，總載荷 247 g。質心與慣量採使用者同意的 CAD 均勻分布近似；螺絲合併到固定件分布，不代表實測球拍內部結構或列印填充。

`fk`、`ik`、`move_pose`、`jacobian` 與 `state` 的 TCP 改為 `paddle/RacketTCP`：朝 link6 +Y 的拍面幾何面積中心，X 沿柄向拍頭，Z 朝該面外法線，Y=Z×X。其在 link6 的位置約為 (−0.212542, 7.352473, 271.280851) mm；精確變換見 [幾何資料](assets/derived/racket/geometry.json)。flange 座標仍可獨立讀取。TCP 回饋來自實際 paddle 剛體。

fixer 碰撞採單一凸包，球拍採拍柄與拍頭兩個凸包，空腔也視為實體；預檢是離散採樣，不能據此宣稱所有連續軌跡皆無碰撞。link5／fixer 的規劃裕量為 0.5 mm，其餘檢查配對仍為 1 mm；工具模式 contact offset 為 0.2 mm、rest offset 為 0。鎖固接觸對由固定關節排除，其餘工具與手臂、地面及障礙物均納入檢查。

完整工具驗證入口會執行數值測試、含載荷的 FK／IK／Trajectory／Collision、球拍與固定件各自的 PhysX 接觸測試，以及實際渲染影片與解碼驗證。證據預設寫入 `.agents/runs/foundation/racket/`，與裸臂證據分開。

若 CAD 裝配變更，在已安裝 CadQuery 2.8／OCP 7.9、NumPy、SciPy 的 Python 環境執行 `python scripts/build_racket_assets.py` 重新產生 SI STL、COM、每公斤慣量與 TCP，再核對載荷設定及重新驗證。正常仿真只讀取生成資產，不依賴 CadQuery。仿真依使用者確認採理想鎖固；真機擊球、實物螺紋／強度與擊球彈性未驗證。

## 球物理驗證

```bash
./verify_ball.sh
```

此入口建立獨立的球桌／球網／固定 CAD 拍面場景，依序檢查重力飛行、二次阻力、桌面反彈、拍面撞擊、球網與桌緣、重現性，以及三種物理步長的收斂。每種步長使用全新 PhysX 程序，結果寫入 `.agents/runs/foundation/ball/dt_240.json`、`dt_480.json`、`dt_960.json`，總判定為 `convergence.json`；只有完整測試與程序退出均成功才返回 0。單獨診斷可用 `./run.sh --headless --mode ball --ball-hz 240`，它只驗證單一頻率，不代表收斂通過。

使用 [球參數](configs/ball.json) 的 40 mm／2.7 g 零旋球與未標定接觸係數。固定拍面使用原 CAD 三角面；可用 `--mode ball --ball-collision compound` 檢查可動球拍的兩塊凸包。模型假設及驗收見 [SPEC](.agents/docs/SPEC.md) 和 [DESIGN](.agents/docs/DESIGN.md)。

## 單拍場景與驗證（M3）

```bash
./run.sh --unit
./run.sh --headless --mode shot
```

機械臂、固定球拍、桌、網與球共用物理場景。底座直接鎖在己方桌端中軸線的桌面上，原點 (−1.275,0,0.760) m，底座後緣距桌端約10.77 mm；這是仿真安裝設定，尚非實物安裝孔測量值。入口依 [單拍設定](configs/single_shot.json) 重複四組發球，檢查接觸、成敗、重置、限時可達性及正常步進沒有直接寫入狀態，報告在 `.agents/runs/foundation/shot.json`。

目前三組來球能碰到預先就位的球拍，一組太偏而漏接；保持球拍的三組接球均落回己方，因此合法回球為 0/4。這是任務與判定驗證，不是主動回球策略。軌跡預測已接入（見下節），下一步為擊球速度與回球規劃。

## 球軌跡預測（M4）

```bash
./run.sh --unit
./run.sh --headless --mode prediction
./run.sh --headless --mode shot
```

預測反彈、最高點、過網、落點的時間和球心位置，並辨認撞網／桌外落點。`prediction` 跑24組固定物理比對，報告為 `.agents/runs/foundation/prediction.json`；`shot` 同時檢查桌上機械臂場景的碰拍前和實測碰拍後預測。門檻為每類事件P95位置20mm、時間10ms。單拍與預測驗收採480Hz；基礎手臂模式維持240Hz。

Python 入口為 `airbot_tt.prediction.predict_ball(state)`；`state` 是世界座標的 `[x,y,z,vx,vy,vz]`。測到碰拍後傳入新狀態並指定 `first_table_event='landing'`。目前採零旋與未標定係數；擦桌緣／網頂回報不支援，未來球拍撞擊須由後續規劃提供。

### M5 擊球規劃驗證

```bash
./run.sh --headless --mode strike
# 每個步長使用獨立程序；報告放本地，不加入Git。
for hz in 240 480 960 1920; do
  ./run.sh --headless --mode impact --ball-hz "$hz" \
    --report ".agents/runs/foundation/impact_${hz}.json" || exit 1
done
python3 tests/check_ball_convergence.py --directory .agents/runs/foundation --impact --finest-hz 1920
./run.sh --headless --mode return --return-split development \
  --report .agents/runs/foundation/return_development.json
./run.sh --headless --mode return --return-split validation \
  --report .agents/runs/foundation/return_validation.json
```

設定在 `configs/strike.json`，程式解說見 [M5筆記](doc/notes/m5.md)。目前M5使用960Hz，是固定桌面站位、限定高反彈零旋來球的三目標scripted基線；獨立驗證36/36合法且命中（每目標12球）：發球前由物理模型算三條軌跡，發球後等待50 ms選用。驗證球不逐球重規劃；完整在線IK及碰撞預檢尚無法滿足來球時限。所有發球納入成功率，不代表實機或廣泛來球表現。

M5也提供實際仿真慢放影片入口：`--mode return-demo --video .agents/runs/foundation/m5_visuals/m5_returns.mp4`。三目標回球路線及36次落點圖見[M5視覺化](doc/notes/m5.md#6-m5現在做到哪裡)。

## PACE 原始方法學習

PACE學習閱讀：先看[完整控制與PPO資料流](doc/notes/pace_pipeline.md)，再對照[獎勵公式與算例](doc/notes/pace_rewards.md)。M6已完成官方MLP的獨立監督訓練、教師／獎勵及接入契約；M7已接通actor／critic與PPO的短跑診斷。完整方法、失敗與限制見[M6筆記](doc/notes/m6_notes.md)。

```bash
./verify_m6.sh --collect .agents/runs/foundation/m6/new_trial  # 校驗官方來源、採集PhysX資料、測試並訓練／評測predictor
./verify_m6.sh --reuse-data .agents/runs/foundation/m6/new_trial    # 重用匹配的本地資料；仍重新訓練／評測
```

M6資料、checkpoint與報告留在`.agents/runs/foundation/m6/`。驗收限定M5窄分布、無球拍飛行、乾淨合成觀測；不是已訓練的回球policy。

## Gemini 335L 相機標定

獨立主機工具入口：`bash scripts/camera_calibration/setup.sh`，接著執行 `bash scripts/camera_calibration/run.sh official info`。交接順序是先讀出廠參數、做幾何與深度驗證，有明確問題才重標。官方讀參數／錄影→離線匯入與驗證、交回檔案及完整操作流程見[相機標定操作筆記](doc/notes/camera_calibration.md)。原始資料留本地runs；不含手眼標定或機械臂操作。離線測試通過，實機採集與精度尚未驗證。


## AIRBOT Play 真機入門（規劃）

先看[H0–H4版本盤點與實作順序](doc/notes/hardware_bringup.md)。目前真機硬體／服務／SDK版本及連接方式未知；第一步是在預計控制機械臂的電腦查已安裝版本，尚無可執行的真機控制入口。硬體驗證與PACE學習分開，不執行M5高速軌跡。

## 研究流程工具

`./run.sh --unit [research|core|pace|camera|all]` 按環境分組驗證。研究命令紀錄與離線比較使用 `python3 scripts/research.py --help`；M6 重畫圖使用 `scripts/plot_m6_results.py`，不會重新訓練。詳細操作、結果格式與限制見 [研究流程筆記](doc/notes/research_workflow.md)。

## M7 初始 PPO 整合與 A／B 診斷

使用固定 PACE ActorCritic／PPO 與六軸向量環境。物理 960 Hz、策略 48 Hz；先通過場景對照、reset／timeout 和動作追蹤，再啟動短試跑。步驟與完整命令見 [M7 筆記](doc/notes/m7_notes.md)，包含控制失敗原因、修正、資料流與結果限制。這不是五組正式研究或真機控制入口。

首次更新異常可用以下入口檢查（需要已下載並校驗的 PACE 來源、Isaac 容器及 GPU）：

```bash
./run.sh --mode pace-check --pace-check update --pace-envs 16 --device cuda:0 --headless --pace-output .agents/runs/foundation/m7/update_check --report .agents/runs/foundation/m7/update_check.json
```

此診斷收集 A／B 三個固定 seeds 的首批經驗，從相同權重與 RNG 比較三個初始學習率，輸出資料一致性與逐 minibatch KL。它會執行短暫仿真及更新診斷，不是只讀繪圖，也不是完整訓練。輸出目錄必須未存在。`pace-smoke`／`pace-ab` 可用 `--pace-learning-rate` 明確指定單變因實驗值；實際設定寫入 protocol 和 checkpoint，未指定時讀 `configs/pace_rl.json`（目前預設初始學習率 `1e-5`）。


回球採 `rally_net_contact_v2`：擦網不立即失敗，繼續追蹤實際落桌／出界／超時。原模型的只評測入口如下，不建立optimizer、不重訓；需本地原checkpoint，輸出目錄不可已存在：

```bash
./run.sh --mode pace-eval --pace-checkpoints .agents/runs/foundation/m7/ab_lr1e5 --pace-output .agents/runs/foundation/m7/new_net_eval --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/m7/new_net_eval.json
```

新舊規則結果見[M7筆記第12節](doc/notes/m7_notes.md)，不要直接比較不同規則下的成功率。


受控來球對照可在完成當前來源的 lifecycle／control／parity 關卡後，固定A只改Y半範圍。以下會真正啟動三個seed的短訓練，輸出目錄必須未存在：

```bash
./run.sh --mode pace-ab --pace-groups A --pace-serve-y-half-range 0.12 --pace-envs 16 --device cuda:0 --headless --pace-output .agents/runs/foundation/m7/serve_challenge --report .agents/runs/foundation/m7/serve_challenge.json
```

用 `0.015` 和另一個輸出目錄可跑窄分布對照。兩者各6144 transitions/seed，M5原設定不改。訓練內建seed7201評測屬開發診斷；[M7筆記第13節](doc/notes/m7_notes.md)另記凍結規則後的共同評測、零動作對照與效用門檻。完整M3的HiGHS預檢錯誤已由等價的最大共同平面餘量LP修正，未放寬碰撞間距。

M7.4首段A/B使用經有限動態見證驗證的初階接球題目：來球初始高度中心2.3m、左右±1.5cm。可用 `--pace-serve-z 2.3 --pace-serve-y-half-range 0.015 --pace-groups A B` 啟動同預算對照；完整例子：

```bash
./run.sh --mode pace-ab --pace-groups A B --pace-serve-z 2.3 --pace-serve-y-half-range 0.015 --pace-envs 16 --device cuda:0 --headless --pace-output .agents/runs/foundation/m7/m74_new_ab --report .agents/runs/foundation/m7/m74_new_ab.json
```

這會真正啟動六次短訓練，要求當前來源的lifecycle／control／parity已PASS，輸出目錄不可已存在。高度寫入protocol／checkpoint，不改M5原檔；`pace-eval`會讀回有效高度，但其內建7201評測仍是開發診斷。前置失敗、可達性範圍與研究限制見[M7筆記第14節](doc/notes/m7_notes.md)。
