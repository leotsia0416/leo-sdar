# SDAR Remask GRPO 當前設計稿

## 文件目的

這份文件只描述目前 SDAR 1.7B remask + GRPO 訓練的當前狀態。

文件重點是回答三件事：

1. 現在 training state 是怎麼構造的。
2. remask policy 現在實際作用在哪裡、怎麼 rollout、怎麼給 reward。
3. 目前預設設定、觀察指標與已知限制是什麼。

---

## 1. 方法總覽

目前版本的核心是：

> 先用 GT 固定一段 prefix，再讓模型自己生成一段局部 window，只在這段 self-generated window 的最後 3 個 blocks 內做 remask，並用 `baseline + top-k single-remask branches` 比較哪些位置的 intervention 最能改善 future diffusion loss 與 terminal answer correctness。

這個版本的目標不是讓主鏈暴露在大量錯 proposal token 的髒 context 中，而是先在一段受控的 generated window 上驗證：

- 哪些位置值得 remask
- remask 後是否真的能改善最終答案

---

## 2. 資料與基本假設

### 2.1 資料集

目前預設訓練資料是 hard `MATH`：

- dataset: `math_train_hard_local`
- row 單位: 一題一個 row
- 不使用 packing

### 2.2 為什麼不能 packing

目前 reward 包含 terminal answer correctness。

terminal reward 的定義是：

- rollout 到 terminal
- 抽 final answer
- 跟該題的 GT final answer 比對

如果一個 row 裡有多題：

- terminal reward 就不再是乾淨的 per-problem reward
- final answer extraction 會混到同一 row 的多題內容

因此目前版本明確要求：

- `packing = false`
- `neat_packing = false`

### 2.3 長度處理

目前使用：

- `cutoff_len = 2048`
- 超過 cutoff 的題目直接 drop

因此訓練總題數會是「原始 hard MATH 題數」扣掉超長樣本後的數量。

---

## 3. 當前 state 定義

### 3.1 記號

- $x_0$: ground-truth answer sequence
- $z$: 當前 training state
- $b_{gt}$: GT anchor block
- $b_{rm}$: remask anchor block
- $W = [b_{rm}-2, b_{rm}]$: 當前固定使用的 3-block remask window

目前 block 定義以 answer tokens 的相對順序切分，預設：

- `block_size = 4`

### 3.2 兩個 anchor

對每個 sample，先抽兩個 block：

1. `GT anchor`: $b_{gt}$
2. `remask anchor`: $b_{rm}$

限制條件：

- `b_rm` 必須至少晚於 `b_gt` 兩個 blocks，確保最後 3 個 blocks 都已經是模型自己生成的內容
- 若長度允許，`b_rm` 不會落在最後一個 block，保留一段 masked suffix 給主 denoising loss

### 3.3 state 的三段結構

目前一個 sample 的 state 由三段組成：

1. `blocks < b_gt`
   使用 GT-filled token，當成正確 prefix

2. `blocks in [b_gt, b_rm]`
   由模型自己 greedy rollout 生成，形成 visible generated window

3. `blocks > b_rm`
   維持 masked，作為主 denoising 區域與 reroll 後的續解區域

因此目前 state 不是純 GT-visible，也不是純 self-generated，而是：

- 正確 prefix
- self-generated local window
- masked suffix

### 3.4 這個 state 的意義

這個設計要同時滿足兩件事：

1. 前綴要夠穩，避免整條解碼路徑太亂
2. remask 必須真的作用在模型自己生成的區域上，而不是對 GT token 假裝做 intervention

---

## 4. Generated Window 與 Remask Window

### 4.1 Generated Window

從 `GT anchor` 到 `remask anchor` 的區間：

$$
[b_{gt}, b_{rm}]
$$

會先由模型自己用 greedy proposal 逐 block 生成，形成 visible generated window。

### 4.2 Remask Window

真正允許 remask 的範圍不是整段 generated window，而是其最後 3 個 blocks：

$$
W = [b_{rm}-2, b_{rm}]
$$

目前 remask 只作用在：

- generated
- visible
- 位於最後 3 個 blocks 內

的 token 上。

也就是說，remask 不直接作用在：

- GT prefix
- 仍為 masked 的 suffix

---

## 5. Remask Candidate 與 Policy

### 5.1 候選位置

候選位置來自 remask window 內的 visible generated tokens。

這些位置對應的是：

- 模型自己已經生成出來
- 目前可以被 rollback 重寫
- 位置上可能對後續解碼造成影響

### 5.2 Remask head

對每個 candidate token，remask head 輸出一個 score / logit。

這個 logit 的作用有兩個：

1. 做 BCE warmup / auxiliary remask loss
2. 當成 branch ranking 的依據

### 5.3 不是隨機選誰 remask

目前版本不是對 candidate set 做隨機 subset sampling。

branch 建法是：

- `branch 0`: baseline，不 remask
- 其餘 branches: 各自只 remask一個位置
- 這些位置由 remask score 做 `top-k` 排序後選出

若 candidate 數量不足，實際 branch 數量是：

$$
1 + \min(K, |C|)
$$

目前預設：

- `gap_grpo_num_samples = 8`

因此目前實際上是：

$$
1 + \min(7, |C|)
$$

---

## 6. 為什麼目前只做單點 remask

目前只做 single-remask branch，有三個原因：

1. credit assignment 較乾淨
2. branch 間差異更容易解讀
3. terminal reward 本身就高 variance，多點 remask 會讓訊號更難穩定

目前版本的研究問題更接近：

> 在當前 state 下，哪一個位置最值得被打回 `[MASK]`？

而不是：

> 哪一組多位置 remask subset 最好？

---

## 7. Rollout 與 Reroll 的語義

### 7.1 Baseline

baseline branch 不做 remask，直接從目前 state 往後 rollout 到 terminal。

### 7.2 Single-remask branch

若某個 branch 選到位置 $i$，且該位置位於 block $b$，則這個 branch 不只是把 token $i$ 打回 `[MASK]`。

目前的語義是：

1. 找出最早被動到的 block $b$
2. 將 generated window 中所有 `block >= b` 的 visible generated tokens 全部 rollback 成 `[MASK]`
3. 從 block $b$ 起重新往後 rollout 到 terminal

也就是說，當 remask 發生在較早的 block 時：

- 後續解碼路徑本來就應該被改寫
- 這是方法本身要評估的效果，不是副作用

### 7.3 後續 decode policy

目前 reroll 與 terminal rollout 都使用 greedy decode。

目前版本刻意不使用 stochastic token sampling，原因是希望：

- reward 與最終 greedy inference 對齊
- 多樣性只來自 remask intervention，而不是 token sampling 隨機性

因此 branch 差異目前主要來自：

- remask 哪個位置
- reroll 從哪個 block 開始

而不是 token-level sampling。

---

## 8. 主 loss

主 loss 仍然是 SDAR 的 diffusion / denoising loss。

對當前 state 的 masked suffix 計算：

$$
L_{\text{mdm}}
$$

這是整體訓練的主目標。

目前 remask policy 不是拿來取代主 loss，而是作為：

- 輔助 decision policy
- 幫助找到更好的 local rewrite action

---

## 9. Remask BCE 輔助項

目前版本仍保留 remask BCE loss。

它的 target 不是「這個位置在抽象上該不該重寫」，而是當前 generated token 是否與 GT 不同。

這個 BCE loss 的作用是：

- 當作 warmup
- 讓 remask head 先學到基本錯誤訊號
- 在 RL 訊號還不穩時提供穩定梯度

但設計上要清楚：

> 最終要學的不是 token correctness 本身，而是 action quality。

---

## 10. Reward 定義

目前 reward 由兩部分組成：

### 10.1 Dense Reward

dense reward 定義為：

$$
R_{\text{dense}} = -L_{\text{future}}
$$

其中 $L_{\text{future}}$ 是該 branch 在 action 之後，剩餘 masked 區域上的 future diffusion loss。

因此 dense reward 通常是負的：

- 越接近 `0` 代表越好
- 越負代表 reroll 後未來越難收尾

### 10.2 Terminal Reward

terminal reward 定義為：

- rollout 到 terminal
- 抽 final answer
- 和 GT final answer 做 normalize 後比對

目前給的是：

$$
R_{\text{terminal}} \in \{0, 1\}
$$

- 對：`1`
- 錯：`0`

這個 reward 只看最終答案是否正確，不直接評估整段 reasoning 是否逐步正確。

### 10.3 Total Reward

總 reward 為：

$$
R = w_d R_{\text{dense}} + w_t R_{\text{terminal}} - \alpha \cdot \mathrm{remask\_rate}
$$

目前預設：

- `w_d = 0.25`
- `w_t = 2.0`
- `alpha = 0.0`

因此目前 reward 主要由 terminal correctness 主導，dense reward 作為 shaping signal。

---

## 11. Terminal Answer Extraction

目前 terminal reward 的 final answer 抽取規則是：

1. 優先抓最後一個 `\\boxed{...}`
2. 否則抓 `#### ...`
3. 否則抓 `answer is ...`
4. 再不然退回最後一個非空行

normalize 後做 exact match。

這表示目前 terminal reward：

- 已經足夠描述最終答案對不對
- 但還不是完整數學等價判定

例如某些等價形式不一定會被判成相同答案。

---

## 12. GRPO 更新方式

目前版本沒有獨立 critic，也沒有 value head。

更新方式是：

1. 對同一個 state 建多條 branches
2. 算每條 branch 的 reward
3. 在同一個 group 內做 reward normalization
4. 用 PPO-style clipped objective 更新 remask policy

因此目前是：

- policy: `SDAR backbone + remask head`
- critic: 無獨立 critic，改用 group-relative reward normalization

---

## 13. 總 loss

目前總 loss 可以寫成：

$$
L_{\text{total}}
=
L_{\text{mdm}}
+
\lambda_{\text{rm}} L_{\text{bce}}
+
\lambda_{\text{grpo}} L_{\text{grpo}}
$$

其中：

- $L_{\text{mdm}}$: 主 diffusion / denoising loss
- $L_{\text{bce}}$: generated-window 上的 remask BCE
- $L_{\text{grpo}}$: branch-level action policy update

目前的設計原則是：

> 主 loss 仍然是主角；GRPO 是讓 remask policy 對 terminal quality 與 future loss 改善有用，而不是取代主 loss。

---

## 14. 當前預設設定

這裡記錄目前的實際預設，而不是候選方案。

### 14.1 資料

- dataset: `math_train_hard_local`
- `cutoff_len = 2048`
- `packing = false`
- `neat_packing = false`

### 14.2 狀態與 rollout

- `block_size = 4`
- `gap_rollout_steps = 4`
- `gap_rollout_strategy = low_confidence_dynamic`
- `gap_rollout_confidence_threshold = 0.95`
- generated window 的 remask 範圍固定是最近 `3` 個 blocks

### 14.3 Remask / GRPO

- `gap_remask_threshold = 0.15`
- `gap_remask_loss_weight = 0.05`
- `gap_grpo_loss_weight = 0.1`
- `gap_grpo_num_samples = 8`
- branch 結構：`1` baseline `+` 最多 `7` single-remask branches

### 14.4 Reward 權重

- `gap_grpo_dense_reward_weight = 0.25`
- `gap_grpo_terminal_reward_weight = 2.0`
- `gap_grpo_remask_penalty = 0.0`

### 14.5 Decode

- token rollout 仍是 greedy
- 不使用 stochastic decode

---

## 15. 目前最重要的觀察指標

### 15.1 主訓練

- `loss`
- `diffusion_loss`
- `remask_loss`

### 15.2 Remask / GRPO

- `grpo_reward`
- `grpo_dense_reward`
- `grpo_terminal_reward`
- `grpo_reward_std`
- `grpo_policy_loss`
- `grpo_branch_count`
- `candidate_tokens`
- `remask_pred_rate`

### 15.3 解讀方式

- `grpo_reward_std` 太小：
  表示 group 內 branch 差異太小，學習訊號弱

- `grpo_policy_loss` 長期接近 0：
  表示 reward 雖然有值，但 policy update 幾乎沒有效果

- `grpo_terminal_reward` 上升：
  表示 branch rollout 的最終答案正確率在提高

- `remask_pred_rate` 太低：
  表示 remask head 可能過度保守

---

## 16. 成功標準

這個方法真正要看的不是 BCE 是否下降，而是：

1. 在固定 decode budget 下，最終答案品質是否更高
2. `grpo_terminal_reward` 是否提高
3. `grpo_reward_std` 是否足夠大，表示 branch 間有可學差異
4. `grpo_policy_loss` 是否不是長期趨近 0
5. 最終 MATH / GSM8K 的答案正確率是否提升

---

## 17. 已知限制

### 17.1 Anchor sampling 仍偏簡單

目前 `GT anchor` 與 `remask anchor` 只是從合法範圍中抽樣，還沒有額外偏向中段或高影響位置。

### 17.2 Reward 還不是 baseline-relative

目前每條 branch 用的是自己的絕對 reward：

$$
R = w_d R_{\text{dense}} + w_t R_{\text{terminal}}
$$

尚未改成：

$$
R_{\text{branch}} - R_{\text{baseline}}
$$

若之後 group 常出現全對或全錯，這會是值得優先考慮的下一步。

### 17.3 目前只做單點 remask

這讓 credit assignment 清楚，但還抓不到多位置交互作用。

### 17.4 Terminal reward 還不是 symbolic equivalence

目前 final answer 判斷仍以 normalize 後的 exact match 為主，不是完整數學等價判定。

### 17.5 尚未做 KV cache 優化

目前 reroll 仍走完整 suffix 重算。當前優先順序是先把方法語義做對，再處理 cache 加速。

---

## 18. 當前版本的定位

目前這版設計的定位很明確：

> 固定一段正確 prefix，讓模型自己生成一段局部 window，只在這段 self-generated window 內測試哪個位置值得 remask，並用 future diffusion loss 與 terminal answer correctness 去更新 remask policy。

它不是用來證明模型已經能端到端自我修正整條推理，而是用來回答：

- 在局部 self-generated reasoning window 中
- 哪些 token 值得被打回 `[MASK]`
- 這樣做是否真的能幫助最終答案

這就是目前版本的完整狀態。        
