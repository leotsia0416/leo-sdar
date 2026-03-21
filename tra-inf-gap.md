# SDAR 的 Train / Inference Gap 與目前採用的 Remask 設計

## 目的

這份文件改成直接描述**目前採用的做法**，不是再把幾種可能設計混在一起。

核心目標只有兩個：

1. 讓 training 看到更接近 inference 的 reveal / transfer 路徑。
2. 在不污染主 MDM objective 的前提下，額外訓練 remask 能力。

一句話總結：

> 回餵到 training 的是 inference 的 decision path 與 mask pattern，不是錯 proposal token 本身。

---

## 目前版本的定位

目前整體上採用的是：

- **PUMA 式 teacher-forced chain** 來決定 state 分布
- **Remask head 當 auxiliary head** 來學哪些 candidate 應該打回 `[MASK]`
- **主 MDM loss 維持乾淨**，不直接吃錯 proposal token

也就是：

- `PUMA` 負責處理 train / inference mismatch
- `Remask` 負責處理 proposal reliability

這兩件事有耦合，但**不能混成同一條髒主路徑**。

---

## 目前採用的硬規則

### 1. 主鏈維持 teacher-forced

training state `z_t` 的推進方式是：

- reveal / transfer 的位置由**當前模型的 policy** 決定
- 但一旦某位置在 training state 中變成 visible，內容仍回填 **GT token**

因此 training 對齊的是：

- 哪些位置被 reveal
- reveal 的順序
- 目前模型實際會走的 decoding path

不是去對齊：

- 錯 proposal token 的字面內容

### 2. 主 MDM loss 只在合法 masked state 上計算

主 loss 一律在目前 chain state `z_t` 上計算：

- 對所有目前仍 masked 的 target positions 算 MDM loss
- 不把 proposal token 直接塞回主 denoiser 當 conditioning context

換句話說：

- state 分布可以更像 inference
- 但 supervision 仍必須建立在乾淨的 teacher-forced state 上

### 3. Remask 只先學「該不該遮回去」

目前 remask head 的角色是 auxiliary：

- 輸入：當前 state 的 backbone hidden states
- 範圍：只在 candidate / transfer 位置上算
- target：proposal 與 GT 是否一致

最小可行標記：

- `proposal[i] != x0[i]` -> `remask_target[i] = 1`
- `proposal[i] == x0[i]` -> `remask_target[i] = 0`

### 4. Remask 影響 mask pattern，不直接改寫主鏈 token content

目前可以讓 remask 影響：

- 哪些位置保留 masked
- 哪些 candidate 不應該被 transfer

但不要讓 remask 直接把錯 proposal token 寫進主 MDM 路徑。

---

## 目前實際流程

每個 sample 維護：

- `x0`: ground truth sequence
- `z_t`: 目前 chain state
- `t`: chain step / rollout depth

每個 training step 做以下事情：

1. 在目前 `z_t` 上跑 backbone
2. 對 `z_t` 中所有仍 masked 的 target positions 計算主 `L_mdm`
3. 用同一次 forward 的 logits / scores 產生 inference-aligned candidate 與 proposal
4. 在 candidate positions 上建立 remask supervision，計算 `L_remask`
5. 根據 inference policy 決定這一步哪些位置可 reveal / transfer
6. 更新下一個 state `z_{t+1}`
7. 凡是 training 中被保留為 visible 的位置，一律回填 GT token
8. chain 結束後再 refill 新 sample

這裡最重要的是第 6, 7 步：

- 下一個 state 的**masking pattern** 可以受 inference policy 與 remask 決策影響
- 但 visible token 的**內容**仍然保持 GT-filled

---

## 用 PUMA 的地方，到底是什麼

真正需要借用的不是 second forward，也不是 proposal token 本身，而是下面三件事：

1. **Teacher-forced chain**
2. **Streaming / persistent chain state**
3. **Current-model policy 決定 reveal path**

這三件事保證了：

- training sample 分布越來越接近實際 inference
- 同時 backbone 不會一開始就暴露在大量錯 token 的髒 context 裡

如果只學了一個 rollout depth 或 second pass 的外形，但沒有保住這三件事，就不算真的把 PUMA 的關鍵用進來。

---

## Main Objective 應該長什麼樣

主 loss 目前應該定義成：

- 在**目前 chain state `z_t`**
- 對**所有仍 masked 的 target positions**
- 計算 MDM loss

也就是：

`L_total = L_mdm + lambda_remask * L_remask`

其中：

- `L_mdm` 是主角
- `L_remask` 是輔助項

目前不建議讓 auxiliary 項反客為主，更不建議讓 projected path 取代主 loss。

---

## Remask Loss 應該長什麼樣

第一版維持最小閉環即可：

- 只在 candidate positions 上算
- supervision 直接來自 `proposal vs x0`
- loss 用 BCE 或等價的 binary objective

這樣 remask head 學到的是：

- 哪些 proposal 很可能不可信
- 哪些位置比較應該維持 masked

而不是讓 backbone 去適應一個被錯 token 汙染過的上下文。

---

## 目前明確不做的事

下面這些做法先不要放進主線：

### 1. Proposal token 直接回寫到主 MDM state

這會讓 backbone 在 training 時看到大量自己猜錯的 token，
主 objective 也會從「在合法 masked state 上還原」變成「在髒 context 上自救」。

### 2. 第二次 forward 當成預設主路徑

`second forward` 或 `projected state` 不是目前的必要條件。

如果一開始就把它做成主路徑：

- loss 容易變高
- 訓練容易不穩
- 問題來源也會更難定位

### 3. Remask 候選集合與 inference 使用的集合不一致

如果 training 的 remask 候選與 inference 的 transfer / reveal candidate 不一致，
那 remask head 學到的就是另一個分布，推理時效果通常不穩。

### 4. 每步重新 random mask，卻聲稱已經對齊 PUMA

PUMA 的重點不是「有 rollout depth」而已，而是：

- 同一條 chain 持續往前走
- 中間 state 被真的重用

如果每步都重新從頭抽 masked pattern，train / inference gap 仍然很大。

---

## 用一個例子說清楚

句子：

> I ate an apple this morning

希望目前的訓練行為是：

1. chain 給出 state：`I ate a [MASK] this morning`
2. backbone 在這個 state 上 proposal 出 `banana`
3. remask head 判斷這個 proposal 不可信
4. 該位置在下一步仍保留 masked，或不被 transfer
5. 主 MDM 仍是在合法 masked state 上學會把答案還原成 `apple`

重點是：

- 我們讓模型學會「這個 proposal 該撤回」
- 不是讓模型在主路徑裡吃下錯 token 後再硬學自救

---

## 工程上應該拆成哪幾塊

### trainer

trainer 需要負責：

- 維護 chain state / rollout depth
- sample 結束後 refill
- 讓 micro-batch 可以對應到持續存在的 state

### model

model 需要清楚區分：

- `mdm backbone` 主路徑
- `remask head` 輔助路徑

最好不要一開始就把兩者寫成不可拆分的單一路徑。

### state utilities

建議獨立整理出：

- `refill_chain_state`
- `advance_chain_state`
- `select_candidates`
- `build_remask_target`
- `apply_remask_to_mask_pattern`

這樣之後調 reveal policy 或 remask policy 時，不會直接把主 loss 一起搞亂。

---

## 最值得看的 logging

比起一堆細碎統計，目前最值得固定追的指標是：

- `loss_mdm`
- `loss_remask`
- `rollout_depth`
- `streaming_refills`
- `candidate_accept_rate`
- `candidate_error_rate`

幾個判讀原則：

- `streaming_refills` 上升時，loss 有短暫抖動通常正常
- `loss_mdm` 不應因為加了 remask 就整體惡化
- `loss_remask` 要能下降，否則 head 沒學到東西
- `candidate_error_rate` 若偏高，代表 current policy 仍不穩，不能太早強耦合

---

## 後續若要升級，正確順序是什麼

建議順序固定如下：

### 第一步：先把 pure PUMA 主鏈跑穩

確認：

- 主 loss 正常下降
- 不再有早期爆炸
- rollout depth 拉高時仍可訓練

### 第二步：在不動主鏈的情況下加 remask head

確認：

- `L_remask` 可學
- candidate error detection 有訊號
- 不會拖垮 `L_mdm`

### 第三步：最後才考慮 projected auxiliary path

如果之後真的要加 `z_proj` / second forward，應該只把它當小權重 auxiliary loss：

`L_total = L_mdm + lambda_remask * L_remask + lambda_proj * L_proj`

而且要守住同一條硬規則：

> `z_proj` 中任何 training 時被保留為 visible 的位置，都應回填 GT，而不是保留錯 proposal token。

---

## 最後的結論

目前這套做法的正確定位是：

- **PUMA** 用來把 inference path 安全地回餵到 training
- **Remask** 用來學哪些 proposal 應該被撤回

所以真正要做的不是：

> 把模型自己猜的 token 全部直接餵回主訓練路徑

而是：

> 先用 teacher-forced chain 管住 state 分布，再讓 remask 對 candidate quality 提供輔助訊號。

這樣才能同時保住：

- SDAR / MDM 的主能力
- train / inference alignment
- remask 的額外價值
