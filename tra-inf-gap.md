# SDAR + PUMA 式 Remask 訓練改造說明

## 目的

希望在 **不破壞 SDAR / MDM 原本還原能力** 的前提下，加入一個 **remask 機制**，讓模型不只會把 `[MASK]` 補回來，還會在推理過程中發現：

- 某些目前已經 visible 的 token 其實不可信
- 這些 token 應該先被遮回 `[MASK]`
- 然後再由 MDM 重新還原

這樣可以縮小 **training / inference gap**：

- 一般訓練時，模型通常只看到「正確的 visible token + 一些 masked token」
- 但真實 inference 時，模型可能先補錯，再需要 remask 來修正

本次改造的核心是：

1. 用 **PUMA 式 teacher-forced rollout** 產生更像 inference 的中間狀態
2. 在某一步額外模擬「如果真的讓模型自己補，哪些位置會補錯」
3. 讓 **remask head** 學會把這些不可信位置遮回去
4. 再讓 **MDM** 在 remask 後的 state 上做正常還原

---

## 一句話版本

先用 **teacher-force** 產生穩定的 rollout state，再用一次 **真實 proposal** 暴露模型這一步可能犯的錯，接著由 **remask** 把不可信的位置打回 `[MASK]`，最後再由 **MDM** 在修正後的 state 上重新還原正確 token。

---

## 設計原則

### 原則 1：training state 要更像 inference state
不要只做獨立 random masking。

改成：

- rollout 順序盡量沿用 inference policy
- 也就是每一步翻開哪些位置，盡量跟推理時一致

### 原則 2：主 MDM supervision 盡量保持乾淨
可以讓模型在訓練時「看到」錯 token，
但不要直接拿大量錯 token 當主 denoising loss 的 conditioning。

比較安全的方式是：

- 錯 token 出現在 proposal / remask 判斷階段
- remask 把不可信的位置打回 `[MASK]`
- 主 MDM loss 在 **projected / remasked 後的 state** 上算

### 原則 3：remask 要真的影響 MDM 的輸入
如果 remask 只是一個獨立分類器，
但它的輸出不會改變 MDM 最後看到的 state，
那 remask 和 MDM 就只是弱耦合。

這次希望做到的是：

- remask 的決策
- 真的會改變下一個用來算 MDM loss 的 state

這樣兩者才有明確關聯。

### 原則 4：teacher-force 只用來穩定 rollout，不是替代 remask
teacher-force 的功能是：

- 讓 rollout state 不會在訓練早期就一路崩掉
- 讓 reveal 順序仍然對齊 inference

真正的錯誤修正能力，還是要由 remask + re-denoise 來學。

---

## 想要的模型行為

以句子：

> I ate an apple this morning

為例，希望模型在訓練中學到的是：

1. teacher-forced rollout 先產生一個穩定中間 state
   - `I ate a [MASK] this morning`
2. 如果這一步真的讓模型自己補，它可能會提議
   - `I ate a banana this morning`
3. remask head 要能判斷 `banana` 不可信
4. 將它打回 `[MASK]`
   - `I ate a [MASK] this morning`
5. MDM 再根據上下文把它還原成 `apple`

這樣模型會學到的是：

- **先補**
- **發現補得不可信**
- **遮回去**
- **再補一次**

而不是只有單純的 masked reconstruction。

---

## 整體流程（工程版本）

### 0. 基本假設
假設目前 SDAR codebase 已經有：

- backbone / denoiser
- MDM loss
- inference policy（例如依 confidence 選要 finalize / reveal 的位置）
- blockwise attention / blockwise state 表示

本次改造盡量沿用既有 SDAR 架構，不重寫主要模型，只新增：

- rollout state 管理
- proposal branch
- remask head
- projected-state 訓練流程

---

### 1. 先維護一個 teacher-forced rollout state
訓練時每個 sample 不直接從 clean sequence 獨立 random corrupt，
而是維護一個 rollout state，記為：

- `x0`: ground truth sequence
- `z_tf`: 目前 teacher-forced state
- `step_idx`: rollout step

`z_tf` 的來源：

- reveal 順序由當前 inference policy 決定
- 但每一步被 reveal 的位置，內容先填回 GT token

目的：

- 讓 training state 的翻開順序更像 inference
- 但又不讓訓練一開始就被錯 token 帶歪

---

### 2. 在 `z_tf` 上做一次 backbone forward
這一步先取得：

- token logits
- hidden states
- confidence / entropy 等可供 policy 使用的資訊

用途有兩個：

1. 決定 rollout 下一步可能要 reveal 哪些位置
2. 提供 proposal 與 remask head 所需的資訊

---

### 3. 額外做一次 proposal，模擬「如果這一步真的讓模型自己補」
在當前步要新 reveal 的位置上，額外建立一個 raw proposal state：

- 可先用 argmax，避免太多隨機噪音
- 將模型這一步提議的 token 填進去
- 得到 `z_raw`

此時 `z_raw` 可能包含非 GT token。

注意：

- `z_raw` 的主要用途是 **暴露真實可能的錯誤**
- 不是直接拿來做主 MDM loss

---

### 4. 建立 remask target
最小可行版本可這樣定義：

只看「本步新 proposal 的位置」：

- 若 `z_raw[i] != x0[i]`，則此位置是 `remask_target = 1`
- 若 `z_raw[i] == x0[i]`，則此位置是 `remask_target = 0`

也就是：

- 這一步模型真的補錯的位置
- 就拿來當 remask 的正樣本

這比人為手動替換 token 更自然，
因為它反映的是 **當前模型真實會犯的錯**。

---

### 5. remask head 預測哪些 visible token 應該打回 `[MASK]`
新增一個 remask head：

- 輸入：backbone hidden states
- 輸出：每個候選 visible 位置是否應該 remask

建議第一版先只在：

- 「本步新 reveal / proposal 的位置」

上做這個判斷，先不要一開始就對所有 visible positions 開戰線。

這樣比較穩，也比較容易 debug。

---

### 6. 根據 remask 結果形成 projected state
根據 remask 判斷，將不可信位置打回 `[MASK]`，得到：

- `z_proj`

重要觀念：

- `z_raw` 是帶有可能錯誤的 proposal state
- `z_proj` 是經過 remask 後，真正拿來餵 MDM 的 state

這裡不要偷懶直接用 `z_raw` 去算主 loss。

比較合理的做法是：

- proposal 負責製造「可能出錯的 visible token」
- remask 負責把不可信位置收回
- MDM 只在 `z_proj` 上重新做 reconstruction

---

### 7. 再跑一次 denoiser / MDM forward
在 `z_proj` 上再做一次 forward，計算主 MDM loss。

loss 只需要算在：

- 目前是 `[MASK]` 的位置
- 或本步被 remask 回去的位置

不要在仍然 visible 的位置上再做一般 token CE。

這樣主目標仍然是：

- 在合法 masked state 上還原 GT

只是這個合法 masked state，現在是由 remask 決策影響而來。

---

### 8. 更新 teacher-forced rollout chain
訓練 step 結束後，要更新下一個 `z_tf`。

這一步建議仍然維持 teacher-force：

- reveal 哪些位置：由 policy 決定
- reveal 出來的內容：仍填 GT token

不要把 `z_raw` 的錯 token 直接滾進 teacher-forced chain，
否則很容易在 early training 造成整條鏈不穩。

也就是說：

- `z_raw` 是拿來產生 remask supervision 的
- `z_tf` 才是穩定 rollout 狀態的主線

---

## loss 設計

第一版只需要兩個 loss：

### 1. `L_mdm`
主 denoising loss。

- 在 `z_proj` 上計算
- target 是 `x0`
- 只在 masked / remasked positions 上算

### 2. `L_remask`
remask head 的分類 loss。

- target 來自 `z_raw` 與 `x0` 的比較
- 只在本步新 proposal 的位置上算
- 可先用 BCE

### 總 loss

`L_total = L_mdm + lambda_remask * L_remask`

第一版先不要把事情做太複雜，
不急著上 RL、utility ranking、REINFORCE、through-rollout gradient。

---

## 弱耦合與強耦合的差別（工程理解）

### 弱耦合
- remask head 只是學會抓 proposal 錯誤
- 但它的輸出不真的改變 MDM 訓練 state

這樣 remask 比較像附屬分類器。

### 強耦合
- remask 的輸出會決定哪些位置被打回 `[MASK]`
- `z_proj` 因此改變
- MDM loss 也因此改變

本設計希望做到的是這種 **強耦合版本**。

也就是說：

- remask 的好壞
- 會直接影響 MDM 最後看到的上下文與 loss

---

## 先做哪個版本最穩

建議先做 **MVP 版**：

1. teacher-forced rollout
2. 本步 proposal 用 argmax
3. proposal 錯誤位置 = remask 正樣本
4. remask 只作用於本步新 proposal 的位置
5. remask 後得到 `z_proj`
6. 在 `z_proj` 上算 `L_mdm`
7. chain 更新仍然 teacher-forced

先驗證下面三件事：

- proposal 真錯的位置，remask head 能不能學起來
- remask 後的 `z_proj`，MDM loss 會不會比不 remask 更好
- 最後 inference 時，模型是否更能修正自己的早期錯誤

---

## 之後可再做的強化方向

### 1. 將 remask 範圍擴到所有 visible positions
不是只看本步新 proposal，
而是整個目前 visible 區域都可以被 remask。

### 2. 不只學「有沒有錯」，還學「remask 之後對 MDM 有沒有幫助」
也就是把 remask 從 error detection，推進到 denoising utility。

### 3. token-level 改成 block-level
如果 SDAR codebase 本身 block 概念很重，
可以考慮 block 級別的 remask 決策，
避免 token 級 decision 太碎。

### 4. inference policy 與 remask policy 更深整合
例如：

- reveal policy 依 confidence 決定
- remask policy 依 inconsistency / uncertainty 決定
- 最後形成 unified decode policy

---

## 明確不要做的事

### 不要 1：直接用大量錯 token state 算主 MDM loss
這樣很容易把 denoising supervision 污染掉，
尤其在 early training 會不穩。

### 不要 2：讓錯 token 直接滾進 teacher-forced 主鏈
proposal branch 和 teacher-forced 主鏈要分開。

### 不要 3：一開始就讓 remask 決定所有位置
先從本步新 proposal 的位置開始，
不然難 debug、也容易類別不平衡。

### 不要 4：一開始就追求 fully differentiable through rollout
第一版先做 joint training 即可：

- backbone + remask head 一起更新
- 但 rollout / proposal / remask decision 本身不必強求端到端可微

---

## 建議改動範圍（給 Codex 的工程指示）

### 優先找的模組
1. **training step / trainer**
   - 增加 teacher-forced rollout state 管理
   - 增加 `z_raw -> remask -> z_proj -> second forward` 流程

2. **inference / decode policy**
   - 復用 SDAR 現有 confidence / remasking 邏輯
   - 抽出可在 training 時重用的 reveal policy

3. **model head**
   - 新增 remask head
   - 最好共用 backbone hidden states

4. **loss module**
   - 增加 `L_remask`
   - MDM loss 改成接受 `z_proj`

5. **state utils**
   - teacher-force update
   - proposal apply
   - remask apply
   - build valid masks / candidate masks

---

## Codex 實作時的高層要求

可以直接把下面這段當需求說明餵給 Codex：

> 請在現有 SDAR 訓練流程上加入一個 PUMA-style teacher-forced rollout + remask training 機制。目標不是改掉原本的 MDM，而是讓模型學會在 rollout 中發現錯誤 visible token，將其打回 `[MASK]`，再由 MDM 重新還原。請保持 backbone 與原本 MDM loss 盡量不變，只新增 rollout state 管理、proposal state、remask head、projected state、以及對應的 loss 與 trainer flow。第一版先只在本步新 proposal 的位置上做 remask supervision，proposal 錯誤位置可直接用和 ground truth 比較產生。chain 更新仍維持 teacher-force，不要把 proposal 錯 token 直接滾入主鏈。主 MDM loss 應在 remask 後的 projected state 上計算，而不是在 raw proposal state 上計算。

---

## 最後的定位

這個方法不是要取代原本 SDAR / MDM，
而是多加一個能力：

**讓模型除了會補 mask，還會在推理過程中承認「剛剛補錯了」，先遮回去，再補一次。**

這就是本次改造最核心的工程目標。
