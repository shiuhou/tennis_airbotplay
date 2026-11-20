# M3：把球與機械臂接成一拍任務

補充日期：2026-11-16。以下對照目前程式並回顧階段實作；歷史結果不代表本輪重跑驗證。

## 做了什麼

將驅動手臂、固定工具、球、桌與網放進同一PhysX場景，加入發球、Reset、事件順序、成敗分類和限時可達性。後續按使用者要求，把底座直接鎖在己方桌端中軸線的桌面上。

## 程式如何協作

[shot_scene.py](../../src/airbot_tt/shot_scene.py)的 `SingleShot` 將 `BallTestbed` 接到 `AirbotSimulation`：每步先準備球外力，PhysX推進後讀球／關節／接觸回饋，再更新任務。reset先恢復關節、穩定驅動，再重置球與事件；各次發球不沿用上一球狀態。

[single_shot.py](../../src/airbot_tt/single_shot.py)的 `ShotEvents` 保存 `incoming → own_bounce → hit → crossed` 階段。己方反彈後碰拍，從網上方穿越並落在對方桌面才算合法回球。持續接觸以起始事件去重；碰固定件／拍柄、撞網、二次己方反彈、漏接等分開記錄。

球的接觸感測分別辨認桌、網及每個工具／手臂剛體。一般手臂碰撞停止門檻為5N；只扣除實測球拍／球反作用力，讓正常拍球事件不被誤停，其餘碰撞仍受保護。

`positioning_candidate()` 用IK求目標關節角，再建立受速度／加速度限制、含碰撞預檢的rest-to-rest軌跡；時間必須小於來球到達時間減反應延遲。這只認證到位能力，不認證擊球速度。

## 碰撞外形及安裝的修改

球拍以link6 Z=205mm分割拍柄／拍頭兩塊凸包，共用同一paddle剛體。200mm切分仍包含突出拍面的柄端過渡，沒有解決問題；205mm分割後正面出球方向、速度及落點與靜態CAD參考一致。

底座原點設為(−1.275,0,.760)m，後緣距桌端約10.77mm，初始TCP約(−.8,0,1.12)m。只排除base_link/table固定安裝接觸，其餘連桿仍檢查桌面碰撞。桌／網設為固定kinematic剛體，以支援逐對接觸感測。

## 怎麼驗證與結果

[tests/single_shot.py](../../tests/single_shot.py)由 `--mode shot` 啟動，四組固定來球各跑兩次，比較事件、結果及球狀態；正常步進的狀態寫入接口被guard禁止。另測撞網、碰fixer、故障重置及固定底座。

三組能碰到預先就位拍面後落回己方，一組偏球漏接，合法回球0/4。桌面安裝當時20項數值、9項單拍通過；M4之後同場景升為480Hz並增加預測驗證。通過的是任務基礎，不是會主動回球。

## 原理與踩坑

- **視覺外形正確，碰撞外形仍可能有問題**

  單一凸包會把厚拍柄與薄拍頭連成略斜的表面，讓正面來球多出側向速度。測試中，整拍凸包產生約 0.023 m/s 的側速；改用原 CAD 靜態表面後，該案例側速為零。M3 再把可動球拍分成拍柄／拍頭兩塊凸包，該案例的出球速度與落點和 CAD 參考一致；原安裝、質量及 TCP 不變。

- **碰到球，不代表回球成功**

  M3 把手臂和球放進同一場景，按「己方反彈、拍面接觸、過網、對方落桌」判定。保持球拍雖能接到三組來球，但球落回己方，仍須記為失敗。

- **位置可達，也要來得及**

  IK 有解只表示能到那個姿態，還要檢查速度、加速度、反應時間和路徑碰撞。目前通過的是預先就位接球；主動移動和擊球速度需要後續規劃。

- **桌上固定時，要區分安裝接觸和意外碰撞**

  底座直接貼桌面是預期的固定接合，只排除底座／桌面這一對。手臂、固定件和球拍仍不能穿桌。底座外形前後不對稱，所以用外形後緣決定距桌端的位置，不能把底座原點直接放在桌邊。

- **移動底座後，要重新驗證接球姿態**

  固定到桌面後，原本的拍面位置與方向未必仍可達。這次重新求解初始姿態，保留四組來球重新驗證；20項數值與9項單拍檢查通過，仍是三組碰拍後己方落桌、一組漏接。幾何變更前的通過結果不能直接套用。

- **兩塊凸包不等於兩個球拍零件**

  凸包像拉緊薄膜包住外形，會填平凹處。拍柄與拍頭各用一個凸包，可以避免厚柄到薄拍頭之間出現假的斜面；兩塊碰撞形狀仍屬於同一個球拍剛體，不會相對移動。

## 展開：共享場景的一個物理步

```text
runtime取關節軌跡目標 → _drive()送關節命令
    → SingleShot.prepare_step()設定球外力
    → 同一個PhysX scene前進一次
    → 更新手臂及球的實測狀態／接觸感測
    → SingleShot.after_step()分類接觸、更新事件、調整保護力
    → runtime檢查剩餘碰撞力／限位／姿態
```

手臂和球不能各自step一次，否則兩者時間會錯開。M3初版240Hz，M4改成設定檔的480Hz後，所有時間累加、感測與settle都依相同dt。`step()`回傳球位置／速度／旋轉、關節角／速度、phase、outcome、contacts、terminated、truncated及reward。

## 展開：事件狀態機為什麼要保存上一個接觸集合

```python
contacts = set(contacts)
new = contacts - self.contacts
self.contacts = contacts
```

如果球連續兩步貼在拍面，兩步都有接觸力，但應只算一次擊球。`new`只保留這一步新出現的接觸；離開後再次接觸才是新事件。

| 當前階段 | 新事件 | 結果 |
| --- | --- | --- |
| incoming | 己方桌面反彈 | own_bounce |
| own_bounce | 球拍接觸 | hit |
| hit | 向對方穿越網平面且球底高於網頂、橫向在範圍內 | crossed |
| crossed | 對方桌面接觸 | legal_return |
| 不符合規定的階段 | 球拍／桌接觸 | illegal_hit／wrong_bounce |

過網不是看某一步剛好x=0，而是前後狀態跨過0後插值：

```python
fraction = -old_x / (new_x - old_x)
crossing = old_position + fraction * (new_position - old_position)
```

只在hit後、球向對方運動時判合法過網。先碰arm／fixer／handle或發生arm fault直接失敗；網接觸也優先結束。outcome一旦設定便鎖住，不會被之後的接觸改成成功。timeout回傳truncated，其他終止回傳terminated；這還不是已建成向量化RL環境。

## 展開：為什麼球拍接觸力可以扣除，但不能關閉保護

感測矩陣的每一欄對應球與一個目標剛體的接觸力。球側量到 `F_ball`，拍側對應反作用力是 `-F_ball`，因此：

```python
adjusted[paddle_sensor_index] += measured_ball_paddle_force
```

加回球側力相當於扣除拍側那一份反作用力，只讓這一對接觸不觸發一般5N停止。若拍面同時碰桌，其他力仍留下。拍柄接觸則把球心轉到paddle局部座標，按link6 Z=205mm的切分位置分類為handle失敗；這是目前基線的分類方法，不是高精度接觸點重建。

## 展開：有IK解為什麼還會被拒絕

`positioning_candidate()`的實際邏輯可簡化為：

```python
available = arrival_s - reaction_delay_s
solution = model.ik(target, start_q)
path = Trajectory(model, [start_q, solution['q_rad']],
                  vmax=position_speed, amax=position_acceleration,
                  require_clear=checker.require_clear)
accepted = path.duration <= available
```

逐層失敗會保留 `deadline`、`unreachable` 或 `collision_or_limits`。計算的是從給定start_q出發的rest-to-rest到位；沒有檢查指定擊球速度，也不是扭矩受限的全動態可達性證明。

實測碰到預置拍面時，測試另記 `measured contact at prepositioned pose` 候選，用保持姿態路徑確認到達時限及碰撞。這不能拿來宣稱手臂能在球飛行期間追上其他位置。低速限值下即使只移幾毫米，也可能因加減速花掉比可用時間更長的時間。

## 展開：reset和失敗測試如何防止假成功

reset先把球停到遠處，重置關節及控制故障，用驅動穩定1秒；若仍有fault就報錯。然後清掉ShotEvents與接觸history，發射實際來球。正常步進用mock guard讓任何球狀態／關節狀態直接寫入立刻拋錯，確認不是靠傳送球或手臂完成任務。

四組發球各重複兩次，事件順序與結果須一致，球位置最大差<10µm、速度差<0.1mm/s。另用物理撞網及初始化時球與fixer重疊測試失敗分類；故意重疊只發生在測試reset，不是正常運動的實作。

合法回球狀態轉移由單元案例驗證；目前實際物理基線仍0/4，所以不能寫成「已有成功回球演示」。測試結果分別證明規則能判成功、物理任務能辨認失敗，兩者不可混用。

## 歷史問題與結果如何解讀

- 整拍凸包在−3m/s正面撞擊產生約−0.0233m/s側速；205mm分塊後限定靜態案例的側速0、法向出球2.398623m/s、落點X約−0.622929m，與CAD三角面參考一致。這不是所有角度高速移動拍都已驗證。
- 原靜態桌／網能反彈，卻沒有期望的逐對感測資料，導致事件誤判；改固定kinematic剛體後，物理與事件測試通過。動畫看得出反彈不代表事件接口正確。
- 底座由桌外移到桌面後，原TCP姿態不再可直接沿用，重求IK並調整拍面角度；四組來球保留，重新驗證。僅安裝底面那一對接觸被放行。
- `./run.sh --headless --mode shot` 現在包含M4預測比較；若只想理解M3，先讀ShotEvents、reset及positioning_candidate，再讀測試中的forecast區塊。
