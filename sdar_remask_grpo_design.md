# From Schedules to Policies: Remasking for Inference-Consistent Discrete Diffusion

## 數學化整理筆記

這份筆記整理的是一個核心主張：

> 若模型在訓練時學會了任意 corruption state 下的正確條件還原，那麼 inference 中反覆進行 remask-and-restore，仍可被視為一個一致的 posterior update 過程。


1. 為什麼這個觀點能回應 train--inference gap。
2. 為什麼目前的 GRPO 版本仍然符合 **"From Schedules to Policies"** 這個題目。
3. 為什麼「局部條件還原正確」和「全局最優」之間仍然要做區分。

---

## 1. 問題背景：train--inference gap 在哪裡？

想處理的 gap 可以分成兩點。

### 1.1 Corruption schedule 的不一致

訓練時，離散 diffusion / masked denoising 常採用接近「整條序列隨機破壞」的 corruption：

- 任意位置都可能被 mask。
- 被破壞的位置分佈大致接近全序列平均。

但 inference 時的 state 並不是這樣。實際上更常見的是：

- 左側 prefix 已有 context。
- 中間有一小段模型自己生成的 visible window。
- 真正待決策的內容主要集中在右側 suffix。

因此，訓練所見 state distribution 與推理時實際遇到的 state distribution 並不一致。

### 1.2 Inference remasking 通常是 heuristic，而不是 learned policy

許多方法在 inference 時會加入一些 remask / rollback / revise 機制，例如：

- confidence 低就重做
- 固定 cadence 檢查
- 某些位置分數過低就 rollback

但這些規則常常只是 heuristic：

- 訓練時沒有明確學會「何時 remask」
- 訓練時也沒有明確學會「該 remask 哪裡」

因此，第二個 gap 是：

> inference 會用到 remask，但 remask decision 本身並未被 train 進模型。

這也是 "From Schedules to Policies" 的核心動機：

- **schedule**：手工設計的 remask 規則
- **policy**：依賴當前 state 做決策的 learned remask rule

---

## 2. 基本形式化

固定一個 conditioning context $c$，例如題目或 prompt。

令

- $x=(x_1,\dots,x_n)$ 為最終答案序列。
- $p_{\mathrm{data}}(x\mid c)$ 為真實資料分佈。

令 $m\in\{0,1\}^n$ 為一個 mask pattern：

- $m_i=1$：第 $i$ 個 token 被 mask
- $m_i=0$：第 $i$ 個 token 保留可見

定義 corruption operator $\mathcal M_m(x)$
表示將 $x$ 中所有 $m_i=1$ 的位置替換成 $[MASK]$ 後得到的 observation。

若以 $x_m$ 表示 masked subset，$x_{\bar m}$ 表示 visible subset，
則模型學習的條件分佈可記為
$$
p_\theta(x_m \mid x_{\bar m}, c, m).
$$

這個式子可被理解為：

> 在任意部分可見、任意部分被遮蔽的情況下，模型如何依條件分佈去還原被遮的部分。

---

## 3. 訓練時「學會任意 step 的還原」是什麼意思？

在 diffusion 語境下，常見說法是：

> 若模型對所有 noise levels 都學會正確的 reverse conditional，則將這些 reverse steps 串起來即可得到正確的 sampling process。

在 remask 觀點下，對應的命題是：

> 若模型對所有 mask patterns 都學會正確的 conditional posterior，則 inference 中任意有限次的 remask-and-restore 都不會偏離正確目標分佈。

更形式化地說，若對所有 $m$ 都有
$$
p_\theta(x_m \mid x_{\bar m}, c, m)=
p_{\mathrm{data}}(x_m \mid x_{\bar m}, c),
$$
則模型在任意 corruption state 下做的重建都是 Bayes-optimal reconstruction。

---

## 4. 單次 remask update 的正確性

固定 context $c$，定義目標分佈
$$
\pi(x):=p_{\mathrm{data}}(x\mid c).
$$

對任意 mask pattern $m$，定義 transition kernel
$$
K_m(x'\mid x)
:=
\mathbb 1\{x'_{\bar m}=x_{\bar m}\}
\, p_{\mathrm{data}}(x'_m \mid x_{\bar m}, c).
$$

這表示：

- 未被 remask 的位置保持不變
- 被 remask 的位置依真實條件 posterior 重新抽樣

### Theorem 1
對任意 mask pattern $m$，$\pi$ 對 $K_m$ 是 invariant 的，即
$$
\sum_x \pi(x) K_m(x'\mid x)=\pi(x').
$$

### Proof
由定義，
$$
\sum_x \pi(x) K_m(x'\mid x)=
\sum_{x_m} \pi(x_m,x'_{\bar m})
\, p_{\mathrm{data}}(x'_m\mid x'_{\bar m}, c).
$$

而
$$
\sum_{x_m} \pi(x_m,x'_{\bar m})=\pi(x'_{\bar m}).
$$

因此
$$
\sum_x \pi(x) K_m(x'\mid x)=
\pi(x'_{\bar m})\, p_{\mathrm{data}}(x'_m\mid x'_{\bar m}, c)=
\pi(x').
$$

證畢。

### 解讀
這個 theorem 的含義是：

> 只要一次 remask + reconstruction 使用的是真實 conditional posterior，那麼這一步更新本身不會破壞目標分佈。

---

## 5. 多次 remask inference 的正確性

若 inference 中依序使用一串 mask patterns
$$
m_1,m_2,\dots,m_T,
$$
其對應 kernels 為
$$
K_{m_1},K_{m_2},\dots,K_{m_T},
$$
則複合 kernel
$$
K:=K_{m_T}\circ\cdots\circ K_{m_1}
$$
仍以 $\pi$ 為 invariant distribution。

### Corollary 1
若每一步 remask update 都使用正確的 conditional posterior，則任意有限次 repeated remask 的整體 transition 仍 preserves
$$
p_{\mathrm{data}}(\cdot\mid c).
$$

### 解讀
局部條件還原正確 $\Rightarrow$ 多步 remask 整體正確。

也就是說，若模型真的學會了任意 step、任意 mask pattern 下的正確 conditional reconstruction，那 inference 時不論進行幾次 remask，本質上都只是反覆做合法的 posterior update。

---

## 6. 這和標準 diffusion 理論的對應

標準 diffusion 會寫成：

- forward process：
$$
q(x_t\mid x_{t-1}),\qquad q(x_t\mid x_0)
$$
- reverse model：
$$
p_\theta(x_{t-1}\mid x_t,c)
$$

若對所有 $t$ 都有
$$
p_\theta(x_{t-1}\mid x_t,c)=q(x_{t-1}\mid x_t,c),
$$
則 reverse chain
$$
x_T\to x_{T-1}\to\cdots\to x_0
$$
會精確生成 $p_{\mathrm{data}}(x_0\mid c)$。

remask 觀點只是把「固定時間步的 reverse」推廣成「任意 subset 的局部再加噪與再還原」：

- diffusion：沿固定 schedule 還原
- remasking：允許依當前 state 決定哪個 subset 需要重新還原

這也是為什麼題目會從 **schedules** 走向 **policies**。

---

## 7. 這個結論解決的是什麼？

上面證明的是：

> repeated remask 在**分佈層面**的正確性

也就是：

- 多次 remask 不會把系統帶離正確 target distribution
- inference 可被解讀為一連串 posterior-preserving transitions

這很重要，但它還不是「全局最優」的命題。

---

## 8. 為什麼這還不等於全局最優？

要非常明確地區分兩件事。

### 8.1 Distributional correctness
這是上面 theorem 已證明的內容：
$$
\pi \text{ is invariant under repeated remask updates.}
$$

### 8.2 Pathwise global optimality
這是另一個更強的主張：

> inference 時每一步都選當前最值得 remask 的 action，最後整條 remask schedule 是全局最優。

這個命題不能單靠「條件還原正確」直接推出。

原因是單步最優不必然等於多步全局最優。存在典型反例：

- action $A$ 現在看起來收益最高
- action $B$ 當下收益較小
- 但先做 $B$ 可能打開未來更大的收益

因此要主張全局最優，還必須再加上結構假設，例如：

- reward 的可加性
- action 之間沒有強互補作用
- Bellman optimality 可成立
- 或 global objective 對 remask set 具有 submodularity

---

## 9. 從 schedules 到 policies：policy 版本的數學化

為了描述 inference 中的多次 remask，可定義一個序列決策問題。

### 9.1 State
令 state $s$ 表示當前生成狀態，例如：
$$
s=(c, x_{\mathrm{prefix}}, x_{\mathrm{window}}, [MASK]_{\mathrm{suffix}}, b),
$$
其中：

- $c$：題目 / prompt
- $x_{\mathrm{prefix}}$：已穩定的 prefix
- $x_{\mathrm{window}}$：當前模型自己生成的 local visible window
- $[MASK]_{\mathrm{suffix}}$：待補完的 suffix
- $b$：可選的 remask budget 或其他控制變數

### 9.2 Action
令 action $a$ 為：
$$
a\in\{\varnothing\}\cup\{\text{rollback from block }j\}.
$$

也就是：

- $\varnothing$：不 remask
- 從某個 block $j$ 開始 rollback，然後 reroll 到 terminal

### 9.3 Return
令總回報為
$$
R=\sum_{t=0}^{T-1} r_t,
$$
或更貼近實作地寫成
$$
R = w_d R_{\mathrm{dense}} + w_t R_{\mathrm{terminal}} + w_f R_{\mathrm{format}} - \alpha\,\mathrm{remask\_rate}.
$$

這時即可定義 action value
$$
Q^\pi(s,a)=\mathbb E[R\mid s_t=s,a_t=a,\pi].
$$

若訓練中 single-remask rollout branch 給出的 return 可以視為對 $Q(s,a)$ 的估計，
那麼 inference 時反覆選擇
$$
a_t=\arg\max_a Q(s_t,a)
$$
就對應於一個 state-dependent remask policy。

這正是：

> 從手工 schedule 設計，轉向 learned remask policy。

---

## 10. GRPO 在這個框架下扮演什麼角色？

目前這條線更清楚的寫法，應該是：

> 先寫一般的 GRPO / PPO-style 公式，再把每個符號直接落到目前的 remask branch 流程。

### 10.1 一般 GRPO 形式

對某個 state $s$，令同一個 group 內可比較的 actions 為
$$
G(s)=\{a_0,a_1,\dots,a_K\}.
$$

一般的 PPO-style GRPO loss 可寫成
$$
\mathcal L_{\mathrm{GRPO}}(s)
=
-\frac{1}{K}\sum_{k=1}^{K}
\min\Big(
r_k \hat A_k,\,
\operatorname{clip}(r_k,1-\epsilon,1+\epsilon)\hat A_k
\Big)
-\beta\,\mathcal H(\pi_\theta(\cdot\mid s)),
$$
其中
$$
r_k
=
\exp\big(
\log \pi_\theta(a_k\mid s)
-\log \pi_{\theta_{\mathrm{old}}}(a_k\mid s)
\big),
$$
且
$$
\hat A_k
=
\frac{R_k-\bar R}{\operatorname{std}(R)+\varepsilon},
\qquad
\bar R=\frac{1}{K+1}\sum_{j=0}^{K}R_j.
$$

也就是說：

- 同一個 group 先產生一批 actions 的 returns $R_0,\dots,R_K$
- 再做 group-normalized advantage
- 最後只對真正要更新 policy 的 actions 做 clipped policy improvement

### 10.2 把公式代入目前的 remask 流程

在我們的實作裡，state $s$ 不是抽象的 token 狀態，而是「當前 generated window 的 remask 決策狀態」，可粗略寫成
$$
s=(c, x_{\mathrm{clean}}, x_{\mathrm{noisy}}, m_{\mathrm{cand}}, \text{block ids}, \text{visible mask}),
$$
其中包含：

- 題目 / prompt
- 目前 clean/noisy 的 answer 狀態
- 可被 remask 的 candidate positions
- 各 token 所屬 block
- 目前 generated visible 區間

接著對每個 candidate position $i$，remask head 給出 logit $z_i$，令
$$
p_i=\sigma(z_i).
$$
把 candidates 依 $z_i$ 由高到低排序後，得到
$$
i_{(1)},i_{(2)},\dots,i_{(N)}.
$$

目前的 action group 不是一般 PPO 裡抽象的一組 sampled actions，而是明確構造成
$$
G(s)=\{a_0,a_1,\dots,a_K\},
$$
其中：

- $a_0$：baseline，不 remask
- $a_k$：remask top-$k$ candidate positions，並從其中最早被選中的 block 開始 rollback，之後 reroll 到 terminal

所以目前 action 的語義其實是：

> 在當前 generated window 下，選擇一個 cumulative remask set，並從最早被打回的 block 起整段重解。

### 10.3 目前 branch log-prob 怎麼寫

令 branch 0 對應 no-remask，則
$$
\log \pi_\theta(a_0\mid s)=\sum_{i=1}^{N}\log(1-p_i).
$$

對於第 $k$ 條 remask branch，
$$
\log \pi_\theta(a_k\mid s)
=
\sum_{j=1}^{k}\log p_{i_{(j)}}
+
\sum_{j=k+1}^{N}\log(1-p_{i_{(j)}}).
$$

這正對應目前 code 裡的 cumulative top-$k$ branch construction：

- branch 0：全部 keep
- branch 1：只 remask 排名第 1 的位置
- branch 2：remask 排名前 2 的位置
- ...
- branch $K$：remask 排名前 $K$ 的位置

### 10.4 目前 branch reward 怎麼寫

對每條 branch $a_k$，先真的 rollout / reroll 到 terminal，然後用目前實作中的 reward 組合得到
$$
R_k
=
w_d R^{(k)}_{\mathrm{dense}}
+
w_t R^{(k)}_{\mathrm{terminal}}
+
w_f R^{(k)}_{\mathrm{format}}
-\alpha\,\rho_k,
$$
其中：

- $R^{(k)}_{\mathrm{dense}}$：future diffusion shaping reward
- $R^{(k)}_{\mathrm{terminal}}$：最終答案正確性 reward
- $R^{(k)}_{\mathrm{format}}$：格式 reward
- $\rho_k$：該 branch 的 remask rate

因此 advantage 不是抽象的 RL return，而是：
$$
\hat A_k
=
\frac{R_k-\bar R}{\operatorname{std}(R)+\varepsilon}.
$$

也就是「這條 remask rollback branch，相對於同一組其他 branches，到底好多少」。

### 10.5 實際更新的是哪一部分

在目前實作裡，baseline branch $a_0$ 的角色是：

- 提供 no-remask 對照
- 參與 group reward normalization
- 不直接放進 policy gradient 的求和項

真正進入 PPO-style policy term 的，是 remask branches $k\ge 1$：
$$
\mathcal L_{\mathrm{GRPO}}^{\mathrm{remask}}(s)
=
-\frac{1}{K}\sum_{k=1}^{K}
\min\Big(
r_k \hat A_k,\,
\operatorname{clip}(r_k,1-\epsilon,1+\epsilon)\hat A_k
\Big)
-\beta\,\mathcal H(\pi_\theta(\cdot\mid s)).
$$

所以從目前 remask 流程來看，GRPO 真正在學的不是一般性的「下一 token policy」，而是
$$
\pi_\theta(\text{which blocks should be rerolled}\mid \text{current generated window}).
$$

這樣寫之後，整個 GRPO 的角色就很明確：

- baseline / remask branches 對應的是一組明確的 rollback actions
- reward 來自 reroll 後的 dense + terminal + format 回報
- policy update 在學的是「哪種 remask rollback 比 no-remask 更值得」

因此更精確地說，GRPO 是：

> a policy-learning instantiation of inference-consistent remasking, specialized to our remask-and-reroll workflow.

---

## 11. 什麼時候可以進一步談全局保證？

若想把主張從「分佈正確」推進到「多次 remask 的全局優化」，通常要再加入額外假設。

### 11.1 Bellman / MDP 版本
若 state 是 Markov sufficient，且 future return 只由當前 state 與 action 決定，
且訓練得到的是正確的 $Q(s,a)$，
則 repeated remask 可被視為標準序列決策中的 optimal policy。

### 11.2 Submodular / greedy 版本
若把一整次 inference 中發生的 remask events 視為一個集合 $S$，並定義全局目標$J(S)$
若 $J$ 滿足：

- 單調性
- 遞減報酬（submodularity，或更強的 adaptive submodularity）

則 greedy 地每次選 marginal gain 最大的 remask，會有經典近似保證：
$$
J(S_{\mathrm{greedy}})\ge (1-1/e)\,J(S^\star).
$$

這條路通常比 exact global optimum 更實際，也更容易成立。

---

## 12. 因此，最穩的理論主張是什麼？

我建議將主張分成兩層。

### Claim A：生成正確性
若模型學會任意 corruption state 下的正確 conditional posterior，則任意有限次 repeated remask inference 都 preserves 正確 target distribution。

### Claim B：優化正確性
若再加入額外結構假設，例如：

- Markov sufficiency
- Bellman optimality
- 或 submodular global objective

則 repeated remask policy 可進一步擁有全局最優或近似全局最優保證。

這樣的寫法既保守，又有數學力度。

---

## 13. 回到題目：為什麼這仍然是 “From Schedules to Policies”？

題目的核心不是「完全消除所有 train--inference gap」，而是：

1. 承認固定 schedule 無法充分描述 inference state distribution。
2. 承認 heuristic remask 無法充分反映最終任務目標。
3. 因此將 remasking 改寫成 state-dependent policy learning 問題。

在這個意義下，GRPO 版本不但沒有背離題目，反而讓題目更成立：

- **From schedules**：不再只靠手工 cadence / threshold / rollback 規則
- **To policies**：改成依當前 state 評估不同 remask actions 的價值

只是必須誠實補充：

> 目前方法不只是在做 inference-consistency，也同時做了 reward-aligned policy learning。

---

## 14. 一句話總結

最濃縮的數學版主張可以寫成：

$$
\forall m,\quad
p_\theta(x_m\mid x_{\bar m},c,m)=p_{\mathrm{data}}(x_m\mid x_{\bar m},c)
\Longrightarrow
K_m \text{ preserves } p_{\mathrm{data}}(\cdot\mid c).
$$

因此對任意 remask schedule $(m_1,\dots,m_T)$，其複合 kernel
$$
K_{m_T}\circ\cdots\circ K_{m_1}
$$
仍 preserves $p_{\mathrm{data}}(\cdot\mid c)$。

也就是：

> **局部條件還原正確 $\Rightarrow$ 多步 remask 推理整體正確。**

而若想進一步聲稱 repeated remask 對最終 reward 是全局最優，則必須再引入額外的 optimal control / Bellman / submodular 類假設。

---

## 15. 可直接放進論文的表述

可以用下列版本作為 paper-style summary：

> We reinterpret remasking as a state-dependent posterior update rather than a hand-crafted inference heuristic. If the model learns the correct conditional posterior for arbitrary corruption states, then each remask-and-restore operation defines a posterior-preserving transition, and any finite sequence of such transitions remains consistent with the target data distribution. This establishes the distributional correctness of repeated remasking. Beyond this, when remasking decisions are optimized using rollout-based relative returns, remasking becomes a learned policy rather than a fixed schedule.
