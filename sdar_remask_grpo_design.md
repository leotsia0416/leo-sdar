# SDAR Remask GRPO 當前設計稿

## 文件目的

這份文件只描述目前真正跑過、且已經驗證有用的 SDAR 1.7B remask + GRPO 作法。

這裡的「當前作法」不是最早的設計假說，也不是所有 script 的歷史預設，而是目前主線實驗實際採用的 recipe，核心對應：

- 訓練：`training_150446`
- 最佳 no-remask baseline checkpoint：`checkpoint-140`
- 主要 eval 路徑：
  - LMDeploy no-remask
  - HF gap decoder remask

---

## 1. 當前已驗證有效的 recipe

目前最有價值的設定不是從壞掉的 checkpoint 繼續修，而是：

1. 從 base chat model 起訓
2. hard MATH 訓練 prompt 直接加入 boxed 輸出要求
3. 降低 diffusion / CE 權重，但不拿掉
4. 補一個和實際 eval 對齊的 terminal correctness reward
5. 另外補 format reward，避免被 `####` 資料格式帶偏

這條線已經在 GSM8K 上超過原本 baseline：

- baseline：約 `80.06`
- `training_150446/checkpoint-140` 的 LMDeploy no-remask：約 `80.59`

這件事的重要性在於：

- 光靠之前的 SFT 沒有穩定超過 baseline
- 目前這條 remask + GRPO 線至少已經出現正增益

---

## 2. 目前實際訓練設定

### 2.1 資料

目前主線 profile 是 `hard_math`。

預設 raw hard MATH 來源仍是：

- `math_train_hard_local`

但真正用來訓練的 dataset 已改成：

- `math_train_hard_boxed_prompt_local`

也就是在 hard MATH 題目的 user prompt 後面，直接加上和 eval 對齊的 boxed 指令：

- step by step 但 concise
- 最後答案放進 `\boxed{}`
- boxed 後立刻停止
- 不要輸出 boxed 後的補充文字

這一步很重要。原因是目前部署與評測介面本來就要求 boxed 輸出；若訓練資料不帶這個 instruction，而 reward 又容忍 `####`，模型會被 MATH / GSM8K 原始資料格式帶偏。

### 2.2 訓練 profile

目前主線對應的有效 recipe 是：

- profile: `hard_math`
- learning rate: `5e-6`
- epochs: `3`
- cutoff length: `2048`
- packing: `false`
- neat packing: `false`

### 2.3 初始化

目前有效實驗不是從先前壞掉的 remask checkpoint 起訓，而是從 base model 起訓：

- base model: `Models/SDAR-1.7B-Chat-`

這點和早期做法不同。因為實驗上 base 明顯比某些中途 checkpoint 強，拿壞 checkpoint 當 anchor 只會把格式與生成習慣一起帶偏。

---

## 3. 訓練 state 與 action 設計

目前訓練語義仍然是 generated-window + single-remask branch。

### 3.1 State 的三段結構

對每個 sample，訓練 state 由三段構成：

1. 正確 prefix  
   前段使用 GT-visible token，當成穩定前綴。

2. self-generated local window  
   中段讓模型自己生成一段 visible window。

3. masked suffix  
   後段維持 masked，留給主 diffusion loss 與 reroll 後續解。

這樣做的理由是：

- 前綴要夠穩，避免整條 rollout 太亂
- remask 又必須真的作用在模型自己生成的區域上

### 3.2 Block 與 remask window

目前 block 單位仍是：

- `block_size = 4`

真正允許 remask 的範圍仍是 generated window 的最後 `3` 個 blocks。

也就是說，現在方法仍然不是在整條 answer 上隨便挑位置，而是在一個受控的 local generated window 內問：

> 哪個位置最值得被打回 `[MASK]`？

### 3.3 Branch 結構

目前 GRPO 仍然不是完整 stochastic subset sampling。

branch 結構是：

- `branch 0`: baseline，不 remask
- 其餘 branches：最多 `K-1` 條 single-remask branch
- candidate 位置依 remask score 做 `top-k` 選擇

目前預設：

- `gap_grpo_num_samples = 8`

所以實際 branch 數量是：

- `1 + min(7, candidate_count)`

### 3.4 Reroll 語義

目前 single-remask branch 的語義仍是 block-level reroll：

1. 先選到某個 candidate token
2. 找出該位置所在的最早 block
3. 將 generated visible 區間內 `block >= b` 的 token 全部 rollback 成 `[MASK]`
4. 從該 block 起重新 rollout 到 terminal

這裡不是只改一個 token，而是從被選中的 block 往後整段重解。

---

## 4. Reward 設計

目前 reward 已經不是只有 dense + terminal 兩項，而是三項：

1. dense reward
2. terminal correctness reward
3. terminal format reward

### 4.1 Dense reward

dense reward 還是：

$$
R_{dense} = -L_{future}
$$

其中 $L_{future}$ 是 action 之後剩餘 masked 區域的 future diffusion loss。

因此 dense reward 通常是負值：

- 越接近 `0` 越好
- 越負表示後續更難收尾

### 4.2 Terminal correctness reward

目前 terminal correctness reward 已經改成盡量對齊實際 eval。

如果 target 是單純數值答案，抽取邏輯會優先從：

1. `\boxed{}`
2. `answer is ...`
3. `</think>` 後半段
4. 尾端數字

這樣可避免把 `72` 和 `72.0` 當不同答案，也比較接近 GSM8K evaluator 實際吃的東西。

如果 target 不是單純 numeric，而是較像 symbolic / math expression，仍回退到較保守的文字正規化比對，避免把 `\frac{1}{2}` 這類答案誤判成 `2`。

### 4.3 Terminal format reward

目前新增了 format reward，目的是把模型拉回 deployment contract，而不是只要求它數值正確。

目前 format reward 主要獎懲：

- 有 `\boxed{...}`：加分
- boxed 後沒有尾巴：再加分
- 出現 `####`：扣分
- `<think>` / `</think>` 不配對：扣分

最後 format reward 被 clamp 到 `[-1, 1]`。

### 4.4 總 reward

目前總 reward 為：

$$
R =
w_d R_{dense}
+
w_t R_{terminal}
+
w_f R_{format}
-
\alpha \cdot remask\_rate
$$

目前主線有效設定是：

- `gap_grpo_dense_reward_weight = 0.25`
- `gap_grpo_terminal_reward_weight = 2.0`
- `gap_grpo_format_reward_weight = 0.25`
- `gap_grpo_remask_penalty = 0.0`

也就是：

- correctness 還是主導項
- dense reward 負責 shaping
- format reward 負責不讓模型被 `####` 與拖尾格式帶走

---

## 5. 總 loss 與目前權重平衡

目前總 loss 仍由三塊組成：

$$
L_{total}
=
w_{diff} L_{diffusion}
+
\lambda_{rm} L_{remask\_bce}
+
\lambda_{grpo} L_{grpo}
$$

目前主線有效設定是：

- `gap_diffusion_loss_weight = 0.5`
- `gap_remask_loss_weight = 0.05`
- `gap_grpo_loss_weight = 0.1`

這裡的重點不是把 CE 拿掉，而是：

- 保留 diffusion / CE 當 anchor
- 但把它降權，避免被資料格式直接帶走
- 同時讓 GRPO reward 開始真正影響 remask policy

目前不建議直接把 CE 歸零。原因是現在 GRPO 優化的核心仍然是 remask branch policy，不是把整個 decoder 直接改造成純 RL。

---

## 6. 為什麼這版比舊版更合理

相對於早期版本，這次真正改對的點是：

### 6.1 訓練 prompt 與 eval prompt 對齊

以前 base model 會 boxed，多半是因為它本來 instruction following 不錯；後續 hard-math 訓練如果不顯式要求 boxed，反而容易把這個習慣洗掉。

現在 hard MATH 訓練 prompt 已經直接加了 boxed instruction，所以模型不是只在 eval 才第一次看到這個要求。

### 6.2 reward 與實際 eval 更對齊

以前 terminal reward 太容易接受 `####` 或尾端數字，對 boxed 的偏好不夠明確。

現在至少分成兩件事：

- correctness 是否對
- 格式是否符合 contract

### 6.3 base 比壞 checkpoint 更適合當起點

實驗上 base 比某些舊 checkpoint 更穩。

因此目前做法已經不把「舊 checkpoint 分佈」當要維持的東西，而是直接從 base 出發。

---

## 7. 目前 eval 路徑的角色分工

### 7.1 LMDeploy no-remask

`test_gap.sh` 在：

- `SDAR_USE_REMASK=false`

時，實際會走 LMDeploy config。

這條路徑目前的角色是：

- deployment-style baseline
- 快速、穩定、接近實際部署的 no-remask 成績

目前最佳例子：

- `training_150446/checkpoint-140`
- LMDeploy no-remask：約 `80.59`

### 7.2 HF gap remask

`test_gap.sh` 在：

- `SDAR_USE_REMASK=true`

時，實際會走你自己寫的 HF gap decoder。

這條路徑目前的角色是：

- 驗證 remask policy 是否有用
- 做 method analysis
- 研究觸發時機、rollback 節奏與 remask score

同一個 `checkpoint-140`，HF gap 預設 remask 也跑到約 `80.59`。

### 7.3 HF gap no-remask 的 caveat

目前如果在 HF gap 路徑用：

- `SDAR_USE_REMASK=true`
- `SDAR_REMASK_THRESHOLD=1.0`

把 remask 關掉，這不是完全公平的「同路徑、只把 rollback 拿掉」控制組。

原因是目前 code 在 threshold >= 1.0 時，還會把：

- `gap_enabled = False`
- `remask_window_blocks = 1`

一起改掉。

這代表這條 no-remask control 不只是「不 remask」，而是連 window / cache commit 節奏都變了。

實驗上同一個 `checkpoint-140`：

- HF gap remask：約 `80.59`
- HF gap threshold=1.0 pseudo-no-remask：約 `79.15`

所以這個 `79.15` 不能直接解讀成「你的 decoder 完全不行」，而要理解成：

- 目前 pseudo-no-remask control 本身就帶了額外路徑改變

---

## 8. 當前最好怎麼解讀分數

目前比較合理的分工是：

1. `LMDeploy no-remask`
   看 deployment baseline

2. `HF gap remask`
   看你的方法本身有沒有價值

3. `HF gap threshold=1.0`
   只能當近似 control，不能當完全公平的 no-remask 等價路徑

也就是說，目前最該看的不是：

> HF gap pseudo-no-remask 能不能直接打贏 LMDeploy

而是：

> 同一個 checkpoint 下，HF gap remask 能不能在你自己的 decoder 上帶來幫助

---

## 9. 目前 remask trigger 的語義

目前 eval 端是否觸發 remask，要經過兩層判斷：

### 9.1 先看 cadence / 時機

只有在以下條件成立時才會檢查 remask：

- 已經進到 `first_remask_block` 之後
- `remask_progress >= remask_start_ratio`
- 命中 `remask_interval_blocks`

### 9.2 再看 remask head 分數

進到可檢查時，會對 window 裡 candidate token 算：

- `gap_remask_head(hidden_state)`
- `sigmoid`
- 取最佳 candidate 的 `best_score`

只有當：

- `best_score >= remask_threshold`

才真的觸發 rollback。

### 9.3 rollback 仍是 block-level

目前觸發後，仍然不是只改一個 token，而是從該 token 所在 block 往後 rollback。

在 `window=3`、`block=4` 的情況下，單次 rollback 規模其實沒有大到離譜；真正看起來極端的是某些題會連續觸發很多次。

---

## 10. 目前如何細看 remask 時機與頻率

舊版 trace 只會記：

- `steps_with_remask`
- `total_remasked_tokens`

這太粗，無法回答：

- 第一個 check 在哪個 block
- 第一個 trigger 在哪個 block
- 分數離 threshold 差多少
- 哪些題是一直被檢但從不觸發
- 哪些題是從中段開始連續觸發 97 次

因此目前已補了 event-level trace。

現在可選擇額外記錄：

- `generated_blocks`
- `remask_progress`
- `candidate_count`
- `best_score`
- `score_margin`
- `selected_block`
- `rollback_start`
- `triggered`
- `remasked_tokens`

建議分析順序：

1. 看 `first_check_block / first_trigger_block`
   判斷是不是開始得太晚

2. 看 `best_score` 與 `score_margin`
   判斷是 threshold 太高，還是 head 根本沒把高風險位置拉開

3. 看 `checks_per_example / triggers_per_example`
   判斷是不是 cadence 太密，導致少數題反覆介入

4. 看 `remasked_tokens_triggered_checks`
   判斷 rollback 粒度是否過重

---

## 11. 目前最重要的觀察指標

### 11.1 訓練期

目前至少要看：

- `loss`
- `diffusion_loss`
- `weighted_diffusion_loss`
- `remask_loss`
- `grpo_loss`
- `grpo_reward`
- `grpo_dense_reward`
- `grpo_terminal_reward`
- `grpo_format_reward`
- `grpo_reward_std`
- `grpo_branch_count`
- `candidate_tokens`
- `remask_pred_rate`

### 11.2 解讀方式

- `weighted_diffusion_loss` 是否真的反映降權後的 CE
- `grpo_terminal_reward` 是否有往上走
- `grpo_format_reward` 是否從負值往 0 靠近
- `remask_pred_rate` 是否仍遠高於真實 positive rate
- `grpo_reward_std` 是否太小，導致 group 內沒有效學訊號

目前從有效 run 來看：

- boxed prompt + format reward 確實能把格式拉回來
- 真正剩下的瓶頸多半是 correctness，不是格式

---

## 12. 目前已知的實驗結論

### 12.1 這條方法不是假的

目前已經有 checkpoint 超過 baseline，因此不能再把整條 remask + GRPO 線當成無效方向。

### 12.2 `LMDeploy wrong` 不等於 hard set

從 LMDeploy no-remask 挑出的錯題，換成 HF gap decoder no-remask，仍能救回一部分。

這代表：

- `LMDeploy wrong`

其實是：

- 真難題
- 路徑敏感題

的混合，不是純 hard set。

### 12.3 目前 remask 觸發仍偏 sparse

全量 GSM8K 上，預設 remask 並不是大量觸發，而是只在少數題介入。

所以目前重點還不是「每題都要 remask」，而是：

- 哪些題型值得介入
- 介入時機是不是太晚
- 介入後是否真的修到 correctness

---

## 13. 目前最重要的限制

### 13.1 branch 採樣仍然偏 deterministic

現在 branch 仍是：

- baseline
- top-k single-remask

不是真正 stochastic multi-action GRPO。

### 13.2 沒有 explicit KL anchor

目前沒有像 PPO ref-model 那樣的 explicit KL 去拉回 base distribution。

當前主要穩定器仍是：

- diffusion / CE
- remask BCE

### 13.3 HF gap no-remask control 仍不夠公平

目前 threshold >= 1.0 會順便改變 window / commit 行為。

若要做真正公平的 no-remask ablation，應該改成：

- 保留 `window_blocks=3`
- 只禁用 rollback branch

### 13.4 correctness 仍是主要瓶頸

目前格式問題已經比以前明顯好很多。

接下來最可能帶來增益的，不是再多修 boxed，而是提升 reasoning correctness。

---

## 14. 目前建議的工作順序

如果要沿這條線繼續做，我會建議：

1. 保留目前 recipe 作為主線
   base 起訓 + boxed prompt + diffusion 降權 + format reward

2. 把 HF gap 的公平 no-remask control 補好

3. 用 event trace 去分析真正的 remask trigger timing

4. 若要再往上推，再考慮：
   - explicit KL to base
   - 更合理的 branch sampling
   - correctness-oriented reward 強化

---

## 15. 一句話總結

目前這版方法的定位很清楚：

> 用 base chat model 當穩定起點，在 boxed-aligned 的 hard MATH 上做 generated-window remask policy learning；主 loss 仍是 diffusion，但透過 eval-aligned correctness reward 與 format reward，讓模型學會哪些 local generated positions 值得 rollback，並且不要被原始資料的 `####` 格式帶偏。

這就是目前真正有效、而且已經在 GSM8K 上產生正向訊號的作法。
