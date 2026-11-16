# AIRBOT Play 資產與來源

## 選型與版本

| 來源 | 版本與用途 |
| --- | --- |
| AIRBOT Play Hardware with MoveIt2 | `feature/humble` 的 `7119068de085d396a203183124ed3eba1cd4e1b9`。採用官方 SolidWorks 匯出的六軸 URDF 幾何、關節座標、質量與慣量。其 URDF 的速度與 effort 均為零，衍生版依下文填入有效值。[官方 Description 套件](https://github.com/DISCOVER-Robotics/AIRBOT-Play-Hardware-with-Moveit2/tree/7119068de085d396a203183124ed3eba1cd4e1b9/src/airbot_play_description) |
| DISCOVERSE | `d67f47c084aba0e0cf422a8725235f8b9238655a`，MIT。使用 force 版本交叉驗證關節運動學，引用其力矩上限作仿真預設；初始六軸零位、命令與回饋分離及逐步追蹤的用法可供參考。下文列出舊版預設模型差異。專案無 MuJoCo 或 DISCOVERSE 執行依賴。[force 資產](https://github.com/discoverse-dev/DISCOVERSE/tree/d67f47c084aba0e0cf422a8725235f8b9238655a/models/mjcf/manipulator/airbot_play_force) · [預設環境](https://github.com/discoverse-dev/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/discoverse/robots_env/airbot_play_base.py) · [路點範例](https://github.com/discoverse-dev/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/examples/tasks_airbot_play/place_block.py) |
| NVIDIA Isaac Lab | `v2.1.0`，上游 tag `21f7136325136ca3f6ca4e0a8125edffe5c24f7e`，BSD-3-Clause。與 Isaac Sim 4.5 組合使用；實際容器映像 ID 見 README。採用 URDF 匯入、Articulation、隱式驅動、PhysX 與 Differential IK API。[版本授權](https://github.com/isaac-sim/IsaacLab/blob/21f7136325136ca3f6ca4e0a8125edffe5c24f7e/LICENSE) · [資產匯入](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/how-to/import_new_asset.html) · [Articulation](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/tutorials/01_assets/run_articulation.html) · [Differential IK](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/tutorials/05_controllers/run_diff_ik.html) |

Isaac Sim 4.5 是已安裝的 NVIDIA 執行依賴，授權與 BSD-3-Clause 的 Isaac Lab 分開。倉庫使用本機已安裝映像，原始碼與資產交付不包含 NVIDIA 執行檔。[Isaac Sim 與 Isaac Lab 授權](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/refs/license.html)

## 衍生 URDF

`assets/upstream/airbot_play_moveit2/` 原樣保留官方 URDF、ROS 套件 manifest、七個裸臂 STL、原始 MIT `LICENSE` 及上游檔案的 `SHA256SUMS`。該 commit 的根目錄授權為 MIT，Description 的 `package.xml` 則標記 Apache-2.0。兩處授權標示、MIT 版權及 Apache-2.0 授權全文均保留。原始 URDF 可用於比對，所引用的夾爪與轉接件模型未納入資產目錄。[上游 MIT 授權](https://github.com/DISCOVER-Robotics/AIRBOT-Play-Hardware-with-Moveit2/blob/7119068de085d396a203183124ed3eba1cd4e1b9/LICENSE) · [套件 manifest](https://github.com/DISCOVER-Robotics/AIRBOT-Play-Hardware-with-Moveit2/blob/7119068de085d396a203183124ed3eba1cd4e1b9/src/airbot_play_description/package.xml)

`assets/airbot_play/airbot_play.urdf` 供仿真使用。保留 `base_link` 到 `link6` 七個實體連桿、`joint1` 到 `joint6` 六個旋轉關節。所有軸皆為關節局部 +Z，保留原始零位及 URDF XYZ/RPY 變換。移除連接座、G2 夾爪本體及三個滑動關節、mimic 分支和夾爪末端 `end_link`。七個連桿仍各自以同一個官方 STL 定義視覺及碰撞，URI 改成相對路徑。沿用原始慣量矩陣與質心，原樣加總質量為 3.770 kg。固定底座由 Isaac 以 fixed-base 匯入方式建立。[原始 URDF](https://github.com/DISCOVER-Robotics/AIRBOT-Play-Hardware-with-Moveit2/blob/7119068de085d396a203183124ed3eba1cd4e1b9/src/airbot_play_description/urdf/airbot_play_description.urdf) · [官方關節座標與工作空間](https://docs.discover-robotics.com/document/en/airbot-play/quick-start/mechanical-structure.html)

位置限制取原始 URDF 與官方技術規格的交集，其中 J5 採用 URDF 的 ±1.7628 rad，J6 採用規格的 ±172°。表格為有效限位的近似值，程式讀取 URDF 完整精度。原始零速度占位改用官方規格的最大關節速度，軌跡則依本專案執行限制配置減速。原始零 effort 占位改用 DISCOVERSE force MJCF 的馬達範圍，J1–J3 各 10 N·m，J4–J6 各 5 N·m。這些 effort 是仿真假設，真機力矩與韌體配置待實測。[官方規格](https://docs.discover-robotics.com/document/en/airbot-play/quick-start/specifications.html) · [DISCOVERSE force 驅動](https://github.com/discoverse-dev/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/models/mjcf/manipulator/airbot_play_force/airbot_play_force_control.xml)

| 軸 | 原始 URDF 位置 (rad) | 官方規格位置 (°) | 有效位置 (rad) | 最大速度 (rad/s) |
| --- | --- | --- | --- | --- |
| J1 | [-3.1416, 2.0944] | [-180, 120] | [-3.141593, 2.094395] | π |
| J2 | [-2.9671, 0.17453] | [-170, 10] | [-2.967060, 0.17453] | π |
| J3 | [-0.087266, 3.1416] | [-5, 180] | [-0.087266, 3.141593] | π |
| J4 | [-3.0107, 3.0107] | [-172.5, 172.5] | [-3.010693, 3.010693] | 2π |
| J5 | [-1.7628, 1.7628] | [-105, 105] | [-1.7628, 1.7628] | 2π |
| J6 | [-3.0107, 3.0107] | [-172, 172] | [-3.001966, 3.001966] | 2π |

Flange 是 `link6` 局部 (0, 0, 0.0865) m，旋轉為單位矩陣，對應原 `arm_connnect_joint` 的連接座根座標。官方 `link6.STL` 最大 Z 為 0.08651586 m，與名義安裝面相差約 16 µm。裸臂 TCP 與 flange 重合，往後以 `T_base_tcp = T_base_link6 · T_link6_flange · T_flange_tcp` 接入自製工具。這是移除官方連接座後的名義安裝面；真機出廠附連接座，若工具裝在連接座外側，需按實際裝配測量偏移、旋轉和定位細節。原 `end_link` 有 0.2466 m 偏移與額外旋轉，對應該 MoveIt G2 配置。[原始 URDF](https://github.com/DISCOVER-Robotics/AIRBOT-Play-Hardware-with-Moveit2/blob/7119068de085d396a203183124ed3eba1cd4e1b9/src/airbot_play_description/urdf/airbot_play_description.urdf) · [官方安裝說明](https://docs.discover-robotics.com/document/en/airbot-play/quick-start/installation/end-effector.html)

## DISCOVERSE 交叉驗證

- **PASS，靜態比對**。Force URDF 的六軸順序、連接、joint origin、軸向及原始位置上下限與官方 Description 完全一致。從 force MJCF 的 body 位置和 wxyz 四元數獨立合成變換，在四組 q（全零、`[0,-0.5,0.65,0.1,-0.3,0.2]`、`[0.2,-0.7,1.1,-0.3,0.45,-0.05]`、`[-0.2,-1.4,1.6,0.5,-0.45,0.8]`）取相同 `link6 → flange` 變換，比對最大差為 3.825e-7 m / 1.934e-6 rad。Force MJCF 自帶 `endpoint` 為 G2 工具點，且合併末端附件質量後的 `link6` 質量為 0.617009 kg。七個 STL 的 Git blob 只有 `link6.STL` 與官方來源相同，故幾何保持官方版本。[force URDF](https://github.com/discoverse-dev/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/models/mjcf/manipulator/airbot_play_force/_play_force.urdf) · [force MJCF](https://github.com/discoverse-dev/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/models/mjcf/manipulator/airbot_play_force/airbot_play_force.xml)
- `AirbotPlayBase` 預設載入帶雙指滑動關節與 tendon 的舊模型，動作向量為六軸加 Gripper。它與同庫 v3 Gripper URDF 的局部座標與限制不同於本專案模型。例：預設 MJCF 的 J1 高度 0.1172 m（官方選用 URDF 0.1127 m），J3/J4 局部框架不同，J6 鏈含 0.23645 m 的工具鏈偏移。配套解析 IK 使用這些幾何常數與專用末端轉換；本工程的有界 IK 和 Jacobian 使用所選 URDF，並以 Isaac Articulation 狀態交叉核對。[預設 MJCF](https://github.com/discoverse-dev/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/models/mjcf/manipulator/airbot_play/airbot_play.xml) · [舊版 URDF](https://github.com/discoverse-dev/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/models/urdf/airbot_play_v3_gripper.urdf) · [解析 IK](https://github.com/discoverse-dev/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/discoverse/robots/airbot_play/airbot_play_ik.py)

## 硬體 SDK 版本參考

三代接口均為文件參考。兩個公開 Git SDK 在所查 commits 未見根目錄 LICENSE，沒有移植其程式或二進位檔。

| 參考版本 | 狀態與控制接口 |
| --- | --- |
| 2024 舊 `sdk` Git `d63d3780a2e6be4ccece3e0d703d45630db84b99` | `create_agent`、`get_current_joint_q/v/t`、`set_target_joint_q/v` 與 `record_* / replay_start` 基於 `arm-control` 函式庫。[舊 Python binding](https://github.com/DISCOVER-Robotics/sdk/blob/d63d3780a2e6be4ccece3e0d703d45630db84b99/python/src/main.cpp) |
| 2025 Hardware Git `56ea4901144bdf43ad4dcb231cff4c7bb54fc8dc` | `ArmState` 含關節 ID、位置、速度、effort 與錯誤資料，`state()` 取快照。低階命令為逐幀 `pvt` 位置、`csv` 速度或 `mit` 混合控制；高階控制由 `airbot_state_machine` 提供。官方下載頁另列低階 Hardware 0.2.9，與高階 5.x 屬不同軟體線。[Hardware Arm API](https://github.com/DISCOVER-Robotics/AIRBOT-Play-Hardware/blob/56ea4901144bdf43ad4dcb231cff4c7bb54fc8dc/airbot_hardware/handlers/arm.hpp) · [下載頁](https://docs.discover-robotics.com/document/en/airbot-play/changelog.html) |
| 官方 5.x 文件 | 廠方 2026-03-13 發布的 V5.2 使用 server 5.2.5 與 `arm_sdk` 5.2.3 的 `AirbotClient`，取代 5.1.6 的 `airbot_py` / `AIRBOTArm`。控制需取得 lease，選擇 `Controller` 後使用 `get_arm_joint_state`、`move_joint`、`move_end_pose`、多路點接口或 `set_arm_emergency_stop`。關節狀態為 rad、rad/s、電流 A，Cartesian pose 以 m 和 **xyzw** 回饋。仿真採 **wxyz**。未識別借用臂的馬達型號、韌體版本或校零；未來選擇單一 SDK 世代並核對其版本與實機參數。[改版記錄](https://docs.discover-robotics.com/document/en/airbot-play/changelog.html) · [控制模式](https://docs.discover-robotics.com/document/en/airbot-play/sdk/concepts/control-modes.html) · [API](https://docs.discover-robotics.com/document/en/airbot-play/sdk/api/reference.html) · [關節狀態](https://docs.discover-robotics.com/document/en/airbot-play/sdk/api/types/arm-state.html) · [笛卡爾姿態](https://docs.discover-robotics.com/document/en/airbot-play/sdk/api/types/cartesian-pose.html) · [韌體欄位](https://docs.discover-robotics.com/document/en/airbot-play/sdk/api/types/firmware-info.html) |

## 球物理來源

球的二次阻力公式及起始參考係數另核對 [PACE v5 §IV-A2](https://arxiv.org/html/2509.21690v5) 與 [官方程式 `fdb192f8`](https://github.com/purdue-tracelab/PACE-ICRA2026/tree/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078)。其 `tt_env.py` 外力使用 rho=1.225、Cd=0.4378，但 `ball.py` 是 3.4 g Vicon 球，預測器另用 2.7 g／Cd=0.47；本工程明確使用 2.7 g 的單一球參數設定，不把上游預測當作數值真值。球物理程式為本地實作，沒有複製 PACE trainer／機器人資產或安裝其依賴。PACE 根授權 MIT，內含 RSL-RL BSD-3-Clause、部分原始檔 Apache-2.0；將來複用須依各檔授權保留來源。此處參考係數不代表本機實物標定。

## 固定執行環境

- 已驗證的本機 OCI 映像 ID 為 `sha256:9c5b00fb775c65913f78f22eadb56912853c1f9f1c38828e53fb99369e29c49e`。Isaac Sim `4.5.0-rc.36+isaaclab_minimal_45.19180.b735e627.gl`，Isaac Lab `2.1.0`，Kit URDF Importer `2.3.14+106.5.0.lx64.r.cp310`。Sim 與 Kit 擴充依 NVIDIA 授權使用。映像與 Kit 擴充由執行環境提供。[Isaac Sim 4.5 授權](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/isaac-sim/4.5.0)
- Python 3.10.15。Kit session 使用 NumPy 1.26.0／SciPy 1.10.1，獨立數值回歸使用同一映像的 NumPy 1.26.4／SciPy 1.15.2。NumPy、SciPy、PyTorch（2.5.1+cu124）為 BSD-3-Clause；ImageIO 2.37.0 與 imageio-ffmpeg 0.6.0 為 BSD-2-Clause。相依模組由映像安裝。
- FFmpeg 編碼使用 imageio-ffmpeg 所附 7.0.2，主機以 Ubuntu 的 FFmpeg/ffprobe 6.1.1-3ubuntu5（GPL build）獨立驗證 H.264。程式庫與影音編碼器未納入 Git。
