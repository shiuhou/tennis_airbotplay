# AIRBOT Play · Isaac Lab

AIRBOT Play 六軸固定底座裸臂。包含關節位置及限時速度控制、姿態 FK／IK、TCP、Jacobian、多路點軌跡、靜態環境碰撞預檢、接觸監測、Stop、Reset 及 Kit 離屏錄影。

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

J6 的 Ø4.00 mm 孔徑及 4.00 mm 孔深僅為 STL 參考，實物螺紋、鎖固、加工公差與強度尚未確認。上述為 CAD 配合成果；仿真仍為裸臂物理模型。
