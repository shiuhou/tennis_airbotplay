# AIRBOT Play · Isaac Lab

AIRBOT Play 六軸固定底座仿真，支援裸臂與固定球拍工具。包含關節位置及限時速度控制、姿態 FK／IK、TCP、Jacobian、多路點軌跡、非零速度定時擊球、限定三目標scripted回球、靜態環境碰撞預檢、接觸監測、Stop、Reset 及 Kit 離屏錄影。

研究主線是將PACE的預測、物理引導獎勵與關節策略適配到固定六軸AIRBOT，邊實作邊理解。M0–M5提供仿真與scripted比較基線；M6已完成離線物理引導與predictor驗證，M7已完成A／B／C／D短對照，並驗證E的資料與在線接口，即時規劃加速不作為其前提。

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
./verify_m6.sh --collect .agents/runs/foundation/m6/new_trial .agents/runs/foundation/m6/new_data  # 校驗官方來源、採集PhysX資料、測試並訓練／評測predictor
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

C（return-only）沿用相同題目與預算。先用已保存的M7.4 A/B模型與評測資料執行離拍／回球預測關卡，通過後才可開訓：

```bash
./run.sh --mode pace-check --pace-check return --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/m7/return.json
./run.sh --mode pace-ab --pace-groups C --pace-serve-z 2.3 --pace-serve-y-half-range 0.015 --pace-envs 16 --device cuda:0 --headless --pace-output .agents/runs/foundation/m7/c_new --report .agents/runs/foundation/m7/c_new.json
```

同時要求當前來源的lifecycle／control／parity PASS，輸出目錄須未存在。return關卡讀取本地`m7/m74/ab`與`m7/m74/final_summary.json`的固定參考，不會下載模型或自動重訓；缺少這批參考時不能冒稱關卡通過。詳細時序、資料流及適用範圍見[M7筆記第15節](doc/notes/m7_notes.md)。

C目前已通過離拍／預測驗證及三seed短對照，工程PASS但整組效用FAIL：新題目合法回球0／24／3（各32球）。包含A/B與初始化的832球共同評測、桌邊預測限制及比較圖見[M7筆記第15節](doc/notes/m7_notes.md)。此為C階段結果；後續D/E更新見下文。


D（hit + return）使用同題目、控制與預算，先驗證與固定C重播的分數加總／reset隔離：

```bash
./run.sh --mode pace-check --pace-check combined --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/m7/combined.json
./run.sh --mode pace-ab --pace-groups D --pace-serve-z 2.3 --pace-serve-y-half-range 0.015 --pace-envs 16 --device cuda:0 --headless --pace-output .agents/runs/foundation/m7/d_new --report .agents/runs/foundation/m7/d_new.json
```

仍要求當前lifecycle／control／parity PASS及未存在的輸出目錄。combined沿用return的固定A/B模型，另需本地`m7/m74c/return_gate.json`及`protocol.json`作已驗證C基準；缺少資料時不自動跳過檢查。方法與結果見[M7筆記第16節](doc/notes/m7_notes.md)。

D已完成三seed短對照及1024球共同評測：工程PASS，新題目合法回球0／7／0（各32球），跨seed效用FAIL，未顯示兩種引導互補。E資料與接口的後續結果見下節，D/E正式效用對照尚未開始。


E前置驗證：新Z=2.3m資料的MLP 3D RMSE10.88mm，原30mm／優於等速門檻PASS；真向量接口與四輪runner smoke已驗證。從零在線模型只有5次optimizer更新，提示對教師仍約2.4m；不能把離線PASS當成在線提示已準確。E actor135維、D115維，E另外保存初始Actor；最大歷史跨度166.7ms。方法、覆蓋診斷、異常退出紀錄及重跑命令見[E predictor筆記](doc/notes/m7_predictor.md)。沒有宣稱E策略優於D。

`--mode pace-data`支援`--pace-data-dir`與`--pace-serve-z`，保護原始資料不被覆蓋；`verify_m6.sh --reuse-data NEW_OUTPUT DATA_DIR`可讀指定新資料。`--pace-check predictor`驗真向量契約，`--pace-groups E`接在線獨立MSE，`pace-eval`可重載E的policy和predictor而不訓練。E短對照須有當前離線、向量接口、runner smoke及原控制／獎勵關卡，smoke不代替效用驗證。

在線predictor的20次資格診斷已完成：固定三個D行為、不更新PPO，實際optimizer步數20／37／36，validation誤差仍1.88／1.28／1.33m。`--pace-check predictor-curve`可重跑該固定預算檢查；完整曲線、可提供提示的覆蓋與限制見[E筆記第7節](doc/notes/m7_predictor.md)。後續`--pace-check predictor-batch`已完成同資料／同初始化的1024對32比較：32誤差降至0.69–0.80m，仍未達30mm；正式配置維持1024。詳見同筆記第8節，正式D/E效用比較仍未執行。


Predictor資料／擬合診斷已完成：`PYTHONPATH=src python tests/pace_predictor_audit.py`純推論交叉評測封存模型與buffer／validation；`--pace-check predictor-audit`以固定D補採72步metadata，不訓練。離線模型在在線buffer約10mm，在線32仍有自身擬合不足及較晚反彈後覆蓋差異；教師差約1.5mm，未查到公尺級座標錯誤。完整表格、圖與重跑前提見[E筆記第9節](doc/notes/m7_predictor.md)。E-frozen的獨立接合與配對規格見下文；原E-online及正式配置保留。

主工作區已依2026-11-24授權完成E-frozen／E-mean選擇性回寫。下列E研究PASS及模型屬各自隔離來源；本次整合來源CPU的mean39＋BC28合計67項PASS，零skip／failure／error、兩程序exit0；GPU工程frozen／mean／lifecycle／parity／control／combined／bank七關亦PASS／exit0。獨立BC／PPO 60球＋probe2球重播亦PASS／exit0，逐球分類重現；兩次工具／launcher前置FAIL保留。bank無合法落點樣本、probe只覆蓋9378兩案，獨立只讀重算PASS／exit0；進度與來源邊界見[E筆記§12](doc/notes/m7_predictor.md#12-主工作區選擇性回寫與整合驗證)。主線較新的BC候選2與短PPO結果保留於本頁後段，舊receipt不直接當成當前整合PASS。

E-frozen顯式載入已驗Z2.3離線MLP並凍結推論，保持D的reward／控制與六維action。D actor115；E-online／E-frozen actor135，critic皆200；舊名稱`E`仍是在線從零。凍結模式要求`--pace-frozen-predictor configs/pace_frozen.json`，校驗模型／來源／資料receipt及來球分布；不符Z2.3／Y±.015m時拒絕，不能當成Z2.2泛化入口。

先前CPU契約27/27 PASS、零skip／error／failure，涵蓋真實既有validation的1223窗逐值推論一致性、checkpoint／history、環境張量與官方PPO合成更新隔離。該輪是CPU／AST與合成rollout驗證，沒有啟動Isaac物理或GPU實驗；以下保留其重跑入口，會執行合成PPO更新，不做predictor MSE：

```bash
./run.sh --unit frozen
```

2026-11-24的GPU工程frozen2／lifecycle16／parity16／control2／combined16均PASS／exit0；combined重現既有96球、41個自由飛行精度樣本通過。隨後經使用者追加授權，E-frozen smoke1536 transitions、三seed各6144 transitions及共同320episode也全部PASS／exit0，predictor沒有新增MSE更新。共同7531／Z2.3上，D合法0／7／0，E-own-initial31／0／0，E-final32／3／0，各分母32；**尚未證明跨seed穩定優於D**，7101初始化已接近上限。工程覆蓋限制、所有故障、提示約9.5mm教師誤差及成本見[E筆記第10節](doc/notes/m7_predictor.md#107-三seed短比較提示已準確策略效用仍不穩定)。

以下是五個工程入口。配置指向既有`m7/m74e/offline`模型與報告、資料manifest及完整固定PACE來源；固定D三模型位於`m7/m74d/d`，combined另使用上述C／A-B參考。隔離工作區已帶入這些依賴；在別的工作區缺失時不自動下載新模型、重訓或跳過驗證。依序執行凍結檢查與當前來源的lifecycle／parity／control／combined；重跑時另選未使用的gates目錄，以下路徑是本輪證據位置：

```bash
mkdir -p .agents/runs/foundation/m7/e_frozen/gates
./run.sh --mode pace-check --pace-check frozen --pace-frozen-predictor configs/pace_frozen.json --pace-envs 2 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_frozen/gates/frozen.json
./run.sh --mode pace-check --pace-check lifecycle --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_frozen/gates/lifecycle.json
./run.sh --mode pace-check --pace-check parity --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_frozen/gates/parity.json
./run.sh --mode pace-check --pace-check control --pace-envs 2 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_frozen/gates/control.json
./run.sh --mode pace-check --pace-check combined --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_frozen/gates/combined.json
```

每一步檢查報告與process exit code，失敗先停下定位。下列為本次追加授權後已執行的入口；`pace-smoke`實際執行1536 transitions的PPO工程驗證，29.402s及32個工程評估episode獨立於三seed比較。重跑須另選尚不存在的輸出目錄，保留本次證據：

```bash
./run.sh --mode pace-smoke --pace-groups E-frozen --pace-frozen-predictor configs/pace_frozen.json --pace-gate-dir .agents/runs/foundation/m7/e_frozen/gates --pace-serve-z 2.3 --pace-serve-y-half-range 0.015 --pace-envs 16 --device cuda:0 --headless --pace-output .agents/runs/foundation/m7/e_frozen/smoke --report .agents/runs/foundation/m7/e_frozen/gates/frozen_smoke.json
```

當前來源關卡及`frozen_smoke.json`完整通過後，下列命令完成三seed各6144 transitions的PPO及共同評測，預先固定最後模型。訓練report記187.301s（含112個內建工程評估episode），共同評測152.094s／320episode；兩者分開計帳。以下是本次證據路徑，重新訓練時smoke及paired_train輸出目錄必須另選未使用路徑：

```bash
./run.sh --mode pace-ab --pace-groups E-frozen --pace-frozen-predictor configs/pace_frozen.json --pace-gate-dir .agents/runs/foundation/m7/e_frozen/gates --pace-serve-z 2.3 --pace-serve-y-half-range 0.015 --pace-envs 16 --device cuda:0 --headless --pace-output .agents/runs/foundation/m7/e_frozen/paired_train --report .agents/runs/foundation/m7/e_frozen/training.json
./run.sh --mode pace-check --pace-check frozen-eval --pace-frozen-predictor configs/pace_frozen.json --pace-checkpoints .agents/runs/foundation/m7/e_frozen/paired_train --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_frozen/paired_evaluation.json
```

`frozen`與`frozen-eval`檢查在函式內固定Z2.3／Y±.015m，不接受訓練用的來球覆寫參數。配對評測固定已讀過的seed7531、Z2.3共32球：zero1＋固定D3＋E自身initial3／final3，共320計分episode；當前程式重評D，不重訓D。這是開發診斷，不能稱新held-out；initial與final都讀同一凍結MLP，缺失／失敗留在分母。額外預訓練128train／32validation、3060 optimizer步與原2.641秒擬合成本另報；E九時刻166.7ms、D五時刻83.3ms，因此比較不能只歸因預測內容。資料流、來源異常、CPU/GPU驗證狀態與判讀詳見[E筆記第10節](doc/notes/m7_predictor.md#10-e-frozen固定準確提示再問策略能否使用)。上述E研究在隔離工作區完成；當時的主線BC唯讀衝突檢查未代稱整合PASS，BC gate 2／4／4是該時點結果。主線後續BC候選2與PPO結果保留於下文，本次回寫驗證見[E筆記§12](doc/notes/m7_predictor.md#12-主工作區選擇性回寫與整合驗證)。


E-mean是後續獨立授權的同維度比較：train128條serve等權平均的固定apex取代動態learned XYZ，Actor135／Critic200、全部同seed初始Actor／Critic／std張量、五點history／valid／reset、D reward及訓練預算保持配對。顯式使用`--pace-groups E-mean --pace-mean-predictor configs/pace_mean.json`，常數不建MLP、optimizer或RNG，不使用validation／held-out選點；原E-frozen模型與報告保持固定。完整配方、初始化、RNG證據邊界及動態歷史差異見[E筆記第11節](doc/notes/m7_predictor.md#11-e-mean讓動態提示與固定平均點使用同一初值)。

2026-11-24隔離E-mean研究已完成：其來源CPU39/39、七GPU關卡、mean smoke1536、三seed各6144 transitions、共同416主要＋D96上下文=512實際回合及獨立CPU重算均PASS／exit0。兩組initial合法均31／0／0；learned-final32／3／0、mean-final0／0／0，各分母32，目標全0。learned在本次固定短比較的合法指標較好，主要差距是保住7101原有合法能力；7102碰拍反而較少、7103仍0合法，不能宣稱三seed都從零學會或普遍優勢。完整失敗、提示加權統計與成本見[E筆記§11.7–11.8](doc/notes/m7_predictor.md#117-完成的短比較learned保住了7101的合法回球mean沒有)。

首次mean-compat的10.251999μm落點差超過原10μm容差，FAIL／exit1保留在`m7/e_mean/initial_checks/mean_compat.json`；該次只存錯誤摘要，後續兩個獨立診斷保留逐球證據並量到順序敏感，物理機制尚未證明。只修正評測順序與規格、未放寬門檻；恢復歷史D前綴後的相容重播224主要＋D96=320實際回合，十組分類一致、已記錄落點分量差0。以下是完整重跑順序，每個前置必須實際PASS才進下一階段：

```bash
./run.sh --unit mean
mkdir -p .agents/runs/foundation/m7/e_mean/gates
./run.sh --mode pace-check --pace-check frozen --pace-frozen-predictor configs/pace_frozen.json --pace-envs 2 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_mean/gates/frozen.json
./run.sh --mode pace-check --pace-check lifecycle --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_mean/gates/lifecycle.json
./run.sh --mode pace-check --pace-check parity --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_mean/gates/parity.json
./run.sh --mode pace-check --pace-check control --pace-envs 2 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_mean/gates/control.json
./run.sh --mode pace-check --pace-check combined --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_mean/gates/combined.json
./run.sh --mode pace-check --pace-check mean --pace-mean-predictor configs/pace_mean.json --pace-envs 2 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_mean/gates/mean.json
./run.sh --mode pace-check --pace-check mean-compat --pace-mean-predictor configs/pace_mean.json --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_mean/gates/mean_compat.json
```

`mean-compat`做zero＋固定learned initial／final共224個主要回合的當前來源相容重播；另依歷史順序插入固定D三seed合計96個上下文回合，實際320回合，不更新PPO／MSE。通過後才執行下列會更新PPO的smoke；smoke結果也必須PASS，才能進正式三seed短跑：

```bash
./run.sh --mode pace-smoke --pace-groups E-mean --pace-mean-predictor configs/pace_mean.json --pace-gate-dir .agents/runs/foundation/m7/e_mean/gates --pace-serve-z 2.3 --pace-serve-y-half-range 0.015 --pace-envs 16 --device cuda:0 --headless --pace-output .agents/runs/foundation/m7/e_mean/smoke --report .agents/runs/foundation/m7/e_mean/gates/mean_smoke.json
./run.sh --mode pace-ab --pace-groups E-mean --pace-mean-predictor configs/pace_mean.json --pace-gate-dir .agents/runs/foundation/m7/e_mean/gates --pace-serve-z 2.3 --pace-serve-y-half-range 0.015 --pace-envs 16 --device cuda:0 --headless --pace-output .agents/runs/foundation/m7/e_mean/paired_train --report .agents/runs/foundation/m7/e_mean/training.json
./run.sh --mode pace-check --pace-check mean-eval --pace-mean-predictor configs/pace_mean.json --pace-checkpoints .agents/runs/foundation/m7/e_mean/paired_train --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/m7/e_mean/paired_evaluation.json
```

smoke固定1536 transitions；正式7101–7103各6144，從對應learned初始張量重新開始、不接續smoke。共同`mean-eval`固定7531、每策略32球，zero1＋learned initial3／final3＋mean initial3／final3，共416主要計分episode；保持同一歷史前綴後再評mean，D96上下文另列，實際512回合。所有失敗留各自分母，initial與final都要評，這是已讀開發題目。`mean`／`mean-compat`／`mean-eval`的檢查函式內固定Z2.3／Y±.015m，不能加入訓練用的`--pace-serve-z`／`--pace-serve-y-half-range`旗標。

上述路徑保留本輪證據；重跑須另選未使用的gate、smoke與paired_train目錄並同步命令參照。當前source改變須重建相容證據，不能改寫舊receipt。舊learned initial未保存CUDA RNG，當前mean保存／核對RNG不能補證歷史GPU狀態完全相同；同維度比較也一併移除了提示對較早動態歷史的壓縮，不能只歸因預測精度。

同球可視化見[動作對照筆記](doc/notes/motion_comparison.md)：M5／零動作／A7101／B7101在同一validation_0上的實測球路、拍面、拍速與落桌，附0.25倍慢放生成入口。`--pace-check motion`擷取zero／既有PPO，`motion-m5`擷取原M5腳本，`python scripts/plot_motion_comparison.py --video .agents/runs/foundation/m7/motion_comparison/comparison.mp4`重畫並產生本地短片；checkpoint及影片留本地，未新增訓練。


DeepMind資料的[Dataset-to-AIRBOT可行性審查](doc/notes/deepmind_feasibility.md)已完成：原始15,792筆只讀分析，814筆低旋篩選、343筆符合目前任務階段；8個動態腳本能碰球，但沒有合法回球見證；343筆未整體接入M7，五顆train球的獨立探索另見下方。`python scripts/prepare_deepmind.py`取得固定commit官方原檔；CPU階段見`tests/deepmind_audit.py --stage ...`，獨立物理入口為`./run.sh --mode pace-check --pace-check dataset-audit --pace-envs 16 --device cuda:0 --headless --report .agents/runs/foundation/deepmind_audit/physics_gate.json`。資料、逐筆證據與action留本地；圖用`PYTHONPATH=src python scripts/plot_deepmind_audit.py`重畫。完整先後順序、既有目錄保護與零旋／分組限制見筆記，沒有增加訓練。

球拍幾何範圍與安裝比較見[工作空間筆記](doc/notes/racket_workspace.md)。`PYTHONPATH=src:tests python tests/racket_workspace.py`採樣並按原碰撞模型過濾，目前與完整關節範圍分報；`python scripts/plot_racket_workspace.py`重畫。保留姿態／法向與潛在接觸點，未做限時接球或真機操作；更換底座XYZ需指定獨立輸出並重新檢查碰撞。

[五球A組PPO探索](doc/notes/deepmind_ppo.md)已完成：固定五個DeepMind train初態、三seed各6144transitions，最後模型皆0/5有效碰拍與合法回球，訓練偶發接觸未轉成可重現回球。新增`--pace-serve-bank`接入驗證過的固定資料，`--pace-gate-dir`指定本輪來源一致的關卡；未指定bank時沿用原人工分布。manifest生成、先驗關卡、訓練與只評測命令見筆記。完整權重與原始結果留本地，`scripts/plot_deepmind_ppo.py`只讀重畫。

[接觸BC初始化對照](doc/notes/deepmind_ppo.md#接觸bc初始化結果離線擬合很好閉環仍未碰球)已完成固定2000步×3seed：237筆示範的動作誤差很小，但閉環均0/5碰拍，未過事前3/5門檻，噪聲與BC＋PPO均NOT_RUN。`--mode pace-bc`收集／擬合／驗收，`--pace-actor-init`只接受三seed門檻已過的產物；本輪產物會被拒絕。完整資料流、閉環偏離量測及重跑入口見筆記；新增研究變體未更改原PACE訓練預設。

追加[同球三方式慢放](doc/notes/deepmind_ppo.md#同球影片原腳本bc預算動作與bc自主控制)：seed7101在原五球上，原script與BC預算動作皆5/5碰拍，自主BC仍0/5，三者皆0合法。`tests/pace_bc_rollout.py --capture-comparison`擷取、`scripts/plot_motion_comparison.py --bc-report ... --video ... --poster ...`生成本地影片；詳細命令、結果與限制見筆記。

[四組回饋診斷](doc/notes/deepmind_ppo.md#bc前動作與速度回饋診斷)已完成且不重訓：自主／前動作介入／速度介入／共同介入碰拍0／1／4／4（各5球），全部0合法；介入只改Actor觀測副本，原自主BC門檻仍FAIL。`tests/pace_bc_rollout.py --feedback-audit`擷取，`scripts/plot_deepmind_ppo.py --feedback <feedback.json>`只讀重畫；早期偏差、逐球限制與詳細入口見筆記。

[偏離後接管與修正示範 BC](doc/notes/deepmind_ppo.md#偏離後接管與修正示範-bc) 在五球的三個接管時刻均恢復碰拍（15/15）。加入 452 筆實際修正樣本，三個自主 Actor 由 0/5、0/5、0/5 提高到 2/5、4/5、4/5，均無合法回球；噪聲與短 PPO 依門檻停在 NOT_RUN。當時以 `bash scripts/run_recovery_bc.sh collect <新輸出目錄>` 收集，覆蓋條件通過後以 `fit` 執行固定三 seed；這些入口要求當時來源，現版重算方式見筆記的歷史來源章節。

[新7101單球接管診斷](doc/notes/deepmind_ppo.md#新7101的9378250ms接管診斷)：9378自主漏接，250ms由原腳本接管後932ms有效碰拍、仍非合法回球。無重訓，原三seed門檻不變。當時由`bash scripts/run_recovery_bc.sh probe <新輸出目錄>`執行；目前需依[歷史來源重算說明](doc/notes/deepmind_ppo.md#用歷史來源重算接管與候選-1-紀錄)恢復來源，不能直接套舊receipt。筆記含三條舊修正軌跡對照與限制。


2026-11-24[自主修正BC與短PPO](doc/notes/deepmind_ppo.md#短ppo完成初始化已可用但原配方未保住碰拍)已完成：候選2三seed自主碰拍3/5、4/5、3/5且重載一致，通過初始化門檻。300次噪聲診斷後，各6144 transitions的PPO將最後碰拍降為1/5、0/5、1/5，合法回球全0。`tests/pace_bc_refine.py --launch --stage fit|noise`提供獨立研究入口；通過來源與三seed門檻的新版receipt可由原`--pace-actor-init`載入，PPO預設不改。完整命令、固定五train球限制、舊來源重算與全部失敗見筆記。`python scripts/plot_recovery_bc.py --refinement <ppo.json>`只讀重畫結果。
