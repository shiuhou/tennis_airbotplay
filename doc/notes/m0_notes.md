# M0：把機械臂變成可驗證的控制基線

補充日期：2026-11-16。以下對照目前程式並回顧階段實作；歷史結果不代表本輪重跑驗證。

## 做了什麼

建立固定底座AIRBOT六軸仿真，支援裸臂與固定球拍。確認座標鏈、限制、FK／IK、Jacobian、軌跡、碰撞停止及Reset都能使用同一套模型；固定件77g、球拍170g的質量／慣量近似已獲使用者同意。

## 怎麼完成：沿著命令走一遍

1. [run_airbot.py](../../scripts/run_airbot.py)解析關節、速度或Cartesian命令，交給 [runtime.py](../../src/airbot_tt/runtime.py)。
2. [kinematics.py](../../src/airbot_tt/kinematics.py)按URDF合成每節座標，計算FK、TCP及Jacobian；IK從seed開始求有界局部解，再用FK檢查位置／姿態誤差。因此IK失敗不是全域不可達的數學證明。
3. [trajectory.py](../../src/airbot_tt/trajectory.py)把路點轉成時間函數q、qd、qdd，按速度／加速度限制決定時間，並呼叫碰撞預檢。
4. runtime每步取軌跡樣本，送位置、速度目標及重力補償effort給關節驅動，讓PhysX推進，再讀實際關節及接觸回饋。正常運動不直接寫關節角。
5. 超限或碰撞觸發Stop，清掉路徑並保持實際位置；Reset才直接恢復初始狀態。

## 讀程式時要理解什麼

- **FK與TCP**：FK回答給定關節角時拍面在哪；TCP是拍面參考框架，不是底座或法蘭。工具由 [tool.py](../../src/airbot_tt/tool.py)接入，安裝幾何見[裝配筆記](assembly_notes.md)。
- **Jacobian**：將六軸關節速度映射到TCP線速度與角速度，是後續指定擊球速度的重要接口。
- **軌跡**：原本五次rest-to-rest插值令每段兩端速度為零，適合到位與停止，不能直接當成中途保持非零速度的揮拍方案。
- **碰撞**：[collision.py](../../src/airbot_tt/collision.py)先用包圍盒粗篩，再用凸包判斷；路徑是離散採樣，不能聲稱所有連續姿態都被證明安全。

## 如何驗證與邊界

入口 `./verify.sh` 及 `./verify.sh --tool-config configs/racket.json`，比較模型FK和PhysX實際姿態、IK到位、軌跡跟蹤、物理接觸、Stop／Reset及影片解碼。歷史M0完整回歸通過；M4改步長接口後另重驗裸臂／工具運動，預設Foundation仍240Hz。

固定底座、工具質量與慣量是仿真模型；不是實物慣量量測，也沒有驗證高速擊球或真機操作。

## 展開：FK、IK和Jacobian在程式裡怎麼連起來

`Kinematics.frames(q)`由base的單位矩陣開始，逐節乘URDF的零位變換與關節旋轉。以下是等價流程示意，非獨立可執行程式：

```python
T_next = T_parent @ T_joint_zero
T_next[:3, :3] = T_next[:3, :3] @ Rotation.from_rotvec(axis * angle).as_matrix()
T_tcp = T_link6 @ T_link6_flange @ T_flange_tcp
```

4×4矩陣左上3×3是方向，最後一欄前三項是位置。乘法順序不能交換。`q`依joint1至joint6排列，角度rad，位置m，姿態四元數wxyz。裸臂flange沿link6局部Z偏移86.5mm；工具TCP再接拍面變換。`fk()`輸出base座標的TCP，不是世界座標；固定底座的世界平移另加。

Jacobian每個旋轉關節的一欄為：

```text
線速度部分 = axis_world × (tcp_position - joint_position)
角速度部分 = axis_world
[v_tcp, omega_tcp] = J(q) @ qd
```

此處程式的共同表達座標實際為base_link，`axis_world`在公式示意中指已轉到共同座標的軸。J為6×6，上三列m/s、下三列rad/s。零位接近腕部奇異姿態，所以不能把Jacobian可計算等同任意方向速度都可達。

`runtime.solve_pose()`先用Isaac Lab的DLS求局部姿態解，每次增量限制在±0.20rad並裁到關節範圍；最多80次。未收斂則用 `Kinematics.ik()` 的SciPy有界least_squares。後者將旋轉誤差乘0.2平衡尺度，但最後以未加權的位置／角度誤差分別驗收，門檻0.2mm／0.002rad。`success`只表示此姿態求解通過，之後仍要看終點碰撞及整條路徑。

## 展開：五次軌跡如何限制速度和加速度

對一段從q0到q1、時間T的運動，令u=t/T：

```text
s(u) = 10u³ - 15u⁴ + 6u⁵
q(t) = q0 + (q1-q0)*s(u)
qd(t) = (q1-q0)*s'(u)/T
qdd(t) = (q1-q0)*s''(u)/T²
```

兩端速度與加速度都是0。程式利用 `max(s')=1.875`、`max(|s''|)=10/sqrt(3)`，按每軸位移推算最短時間，再取各軸最大值。使用者指定的T若不足就拒絕，不會強行超速。預設命令限速0.15rad/s、加速度0.30rad/s²，這是低速驗證設定，不是已驗證的擊球能力。

`Trajectory.sample(t)`回傳q、qd、qdd；runtime把q與qd送驅動，qdd用於目標記錄，並沒有完整的加速度逆動力學前饋。速度模式另由 `VelocityTrajectory` 做有限時間加減速，不是無限延續的速度命令。

## 展開：驅動目標不等於實際狀態

`_drive()`的核心接口：

```python
robot.set_joint_effort_target(gravity_effort, ...)
robot.set_joint_position_target(q_target, ...)
robot.set_joint_velocity_target(qd_target, ...)
robot.write_data_to_sim()
```

這些是送目標，不是改位置。隱式驅動和PhysX決定實際運動，下一步讀q／qd才能算跟蹤誤差。重力補償來自實際含載荷的PhysX模型，經限幅後另送effort；驅動與補償各有限制，但兩者總和沒有額外統一限幅，不能把此模型當成已校準的真機力矩控制器。

早期只調PD仍有J2／J3靜止速度偏差；加入重力前饋後基礎靜止測試通過。這也說明只看命令曲線平滑不能證明手臂真的跟上。

## 展開：測試到底覆蓋哪些錯誤

| 測試 | 比較的對象 | 防止的誤判 |
| --- | --- | --- |
| FK／TCP | URDF計算 vs PhysX實際連桿／球拍姿態 | 只驗自己的FK公式卻裝錯工具 |
| IK | 目標 vs FK回代，再vs實際到位 | 有數學解卻沒有真的走到 |
| 軌跡 | q／qd目標 vs實際回饋，限制及拒絕案例 | 插值正確但驅動失控 |
| 工具慣量 | URDF質量／COM／張量 vs匯入後PhysX | 視覺有球拍但物理載荷沒接入 |
| 碰撞 | 離線拒絕 + 真實接觸及停止 | 只算幾何，不驗碰撞感測 |
| Reset／退出 | 清除命令與故障、每次run token及程序退出碼 | 舊報告或啟動失敗被當成PASS |

測試位置是 [foundation.py](../../tests/foundation.py)，運行串接見 [verify.sh](../../verify.sh)。工具初次物理接入歷史結果：FK位置最大約3.48e−7m、IK實際到位約3.27e−5m、最大關節跟蹤誤差約5.14e−4rad；這些是當時低速案例，不是高速規格。後續改動的結果以ROADMAP和對應報告為準。
