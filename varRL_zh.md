# 從局部後驗不一致 KL 到 Rollout Reward 的變分式 RL 橋接（UPO / MDLM 風格記號版）

## 目標

本文希望建立一條數學上乾淨、且在記號上更貼近 discrete masked diffusion 文獻的橋樑：
從 **局部條件式的 posterior inconsistency KL**，推導到一個具有 variational / control-as-inference 詮釋的 **理想 rollout-level reward objective**；再進一步說明，實作上所優化的 **surrogate reward objective** 與理想 objective 之間的差距，可以被顯式分解並在溫和條件下加以控制。

本文採用與 UPO 類工作接近的序列記號，同時在機率寫法上盡量貼近 MDLM / ReMDM 常見的寫法：

1. 以 $q \sim \rho_Q$ 表示 prompt / context 的抽樣；
2. 以 $x_0 \sim p_{\mathrm{data}}(\cdot \mid q)$ 表示 clean target sequence；
3. 以 $q(x_n \mid x_0)$ 表示 forward masking / corruption kernel，其中 $x_n \in \mathcal X_n$ 為含有 $n$ 個 masks 的部分遮罩序列；
4. 以 $\pi_\theta$ 表示 MDM 的 denoiser / reverse conditional；
5. 以 $g_\phi$ 表示 remask / unmask position-selection policy；
6. 以 $p_{g_\phi,\theta}(x_{0:L}, a_{1:L} \mid q)$ 表示由 policy $g_\phi$ 與 base model $\pi_\theta$ 共同誘導出的 rollout distribution。

核心重點如下：

1. 基本的不一致量是 **state-wise local conditional KL**，而不是 trajectory-wise KL；
2. 這個局部 KL 可以沿著 rollout 訪問到的 masked states 累積，從而 **lift** 成 trajectory cost；
3. 由該 cost 誘導出的 Gibbs target trajectory distribution，其 matching 問題等價於最大化一個 **KL-regularized ideal reward objective**；
4. 實際訓練時通常優化的是可觀測的 surrogate reward，而不是精確的 ideal reward；
5. surrogate 與 ideal 之間的差距具有一個 **精確的分解式**，並且可以進一步導出一個 **surrogate-to-ideal transfer bound**。

---

## Part I. 理想變分橋接

## 1. 記號與局部 conditional KL

令 vocabulary 為 $\mathcal V$，target sequence 長度為 $L$，則 clean sequence space 為

$$
\mathcal X = \mathcal V^L.
$$

對於每個 $n \in \{0,1,\dots,L\}$，令

$$
\mathcal X_n
:=
\{x \in (\mathcal V \cup \{M\})^L : \text{$x$ 恰有 $n$ 個 mask tokens}\},
$$

並以 $x_n \in \mathcal X_n$ 表示一個含有 $n$ 個 masks 的部分遮罩序列。令

$$
\mathcal A_{x_n} = \{a_1[x_n],\dots,a_n[x_n]\}
$$

表示 $x_n$ 中所有 masked positions 的索引集合。為簡潔起見，下文將省略 $[x_n]$，直接寫成 $a_i$。

給定 prompt / context $q \sim \rho_Q$，clean target sequence 服從

$$
x_0 \sim p_{\mathrm{data}}(\cdot \mid q).
$$

依循 MDLM / ReMDM 風格，令

$$
q(x_n \mid x_0)
$$

表示將 $x_0$ forward-corrupt 成 $x_n$ 的 masking kernel。對應地，training-time corruption 所誘導出的 masked-state distribution 可寫為

$$
q_{\mathrm{data}}(x_n \mid q)
:=
\sum_{x_0 \in \mathcal X}
p_{\mathrm{data}}(x_0 \mid q) \, q(x_n \mid x_0).
$$

對於一個給定的 masked state $x_n$，記其 masked coordinates 為 $M(x_n)$，則 oracle data posterior 可寫為

$$
p_{\mathrm{data}}\big(x_0^{M(x_n)} \mid q, x_n\big),
$$

而模型在相同 state 上所誘導出的對應 conditional distribution 記為

$$
\pi_\theta\big(x_0^{M(x_n)} \mid q, x_n\big).
$$

這裡必須強調：

$$
p_{\mathrm{data}}\big(x_0^{M(x_n)} \mid q, x_n\big)
\quad\text{與}\quad
\pi_\theta\big(x_0^{M(x_n)} \mid q, x_n\big)
$$

都不是 trajectory distributions；它們是 **在固定 masked state $x_n$ 下，對被遮住之 clean content 的局部 conditional distributions**。

因此，我們定義 **state-wise posterior inconsistency cost** 為

$$
c_\theta(q, x_n)
:=
D_{\mathrm{KL}}\!\left(
 p_{\mathrm{data}}\big(x_0^{M(x_n)} \mid q, x_n\big)
 \,\|\,
 \pi_\theta\big(x_0^{M(x_n)} \mid q, x_n\big)
\right).
$$

這是本文真正的原始理想量：它是 **局部 conditional KL**，不是 rollout KL，也不是 terminal-output KL。

此外，$x_n$ 可以同時有兩種來源：

1. **training-time states**：由 $p_{\mathrm{data}}(x_0\mid q)$ 與 forward kernel $q(x_n\mid x_0)$ 所產生；
2. **inference-time visited states**：由當前 $(\theta,\phi)$ 所誘導的 rollout 所實際訪問到。

因此，$c_\theta(q,x_n)$ 應被視為一個 **ideal local objective**；在 inference-time visited states 與 training-time corruption states 存在 distribution shift 時，它未必對每一個實際訪問到的 $x_n$ 都能被精確觀測。

---

## 2. rollout dynamics 與 trajectory cost

在 inference 時，我們從 fully masked target $x_L = M^L$ 開始，逐步去遮罩。對於 $n=L,L-1,\dots,1$，在 state $x_n$ 上：

- action space 為 $\mathcal A_{x_n}$；
- policy head $g_\phi(\cdot \mid q, x_n)$ 從其中選擇一個位置 $a_n \in \mathcal A_{x_n}$；
- base MDM $\pi_\theta$ 依據 $(q,x_n,a_n)$ 產生對該位置的 token distribution，並得到下一個 state $x_{n-1}$。

與 UPO 類工作的寫法一致，可將單步 transition 寫成

$$
p_{g_\phi,\theta}(x_{n-1} \mid x_n, q)
=
T_n^{(g_\phi,\theta)}(x_n, x_{n-1} \mid q)
=
 g_\phi(a_n \mid q, x_n)
 \, \pi_\theta(x_{n-1} \mid q, x_n, a_n),
$$

其中 $\pi_\theta(x_{n-1} \mid q, x_n, a_n)$ 在離散 masked diffusion 的 setting 下，表示只在座標 $a_n$ 上將 mask 替換為 sampled token，而其餘座標保持不變。

因此，給定 $q$ 的完整 rollout distribution 可寫成

$$
p_{g_\phi,\theta}(x_{0:L}, a_{1:L} \mid q)
:=
\prod_{n=1}^L
 g_\phi(a_n \mid q, x_n)
 \, \pi_\theta(x_{n-1} \mid q, x_n, a_n).
$$

現在沿著 rollout 將局部 KL 累積，定義 cumulative inconsistency cost：

$$
C_\theta(q, x_{1:L})
:=
\sum_{n=1}^L c_\theta(q, x_n).
$$

注意：$C_\theta$ 仍然是由 **local conditional KL lifted to trajectory level** 所得到的量；它不是直接從某個 trajectory-level data/model KL 定義出來的。

對應的理想 inference-consistency objective 為

$$
J_{\mathrm{IC}}(\theta,\phi)
:=
\mathbb E_{q \sim \rho_Q}
\Big[
\mathbb E_{(x_{0:L},a_{1:L}) \sim p_{g_\phi,\theta}(\cdot \mid q)}
\big[C_\theta(q,x_{1:L})\big]
\Big].
$$

因此，原本的局部 KL objective 自然被 lift 成一個 rollout-level cost minimization 問題。

---

## 3. 參考 rollout distribution 與 Gibbs target

令 $g_{\mathrm{ref}}$ 為 reference unmasking policy，則其與 base model $\pi_\theta$ 共同誘導出的 reference rollout distribution 記為

$$
p_{g_{\mathrm{ref}},\theta}(x_{0:L}, a_{1:L} \mid q).
$$

給定溫度參數 $\beta > 0$，定義 **Gibbs target trajectory distribution**：

$$
p_\beta^\star(x_{0:L}, a_{1:L} \mid q)
:=
\frac{1}{Z_\beta(q)}
\, p_{g_{\mathrm{ref}},\theta}(x_{0:L}, a_{1:L} \mid q)
\exp\!\big(-\beta C_\theta(q,x_{1:L})\big),
$$

其中 partition function 為

$$
Z_\beta(q)
:=
\mathbb E_{(x_{0:L},a_{1:L}) \sim p_{g_{\mathrm{ref}},\theta}(\cdot \mid q)}
\big[
\exp(-\beta C_\theta(q,x_{1:L}))
\big].
$$

這裡的構造意義是：

- reference rollout distribution 保留了原本的 denoising dynamics 與 reference scheduling bias；
- exponential tilt 項 $\exp(-\beta C_\theta)$ 使得 cumulative inconsistency 較小的 trajectories 被賦予更高權重；
- 因而 $p_\beta^\star$ 是一個 **偏好低 local-KL 累積成本的理想 rollout distribution**。

需要注意的是，當 $\theta$ 也在訓練中更新時，$c_\theta$、$C_\theta$、$p_{g_{\mathrm{ref}},\theta}$ 與 $p_\beta^\star$ 都會隨之變動。因此，下述變分等價應理解為：**在固定當前 $\theta$ 的條件下，對應到該時刻所誘導出的理想目標。**

---

## 4. 變分恆等式

對任意 rollout distribution $p(\cdot \mid q)$，都有

$$
D_{\mathrm{KL}}\!\left(
 p(x_{0:L},a_{1:L} \mid q)
 \,\|\,
 p_\beta^\star(x_{0:L},a_{1:L} \mid q)
\right)
=
D_{\mathrm{KL}}\!\left(
 p(x_{0:L},a_{1:L} \mid q)
 \,\|\,
 p_{g_{\mathrm{ref}},\theta}(x_{0:L},a_{1:L} \mid q)
\right)
+ \beta \, \mathbb E_p\big[C_\theta(q,x_{1:L})\big]
+ \log Z_\beta(q).
$$

### 證明

由定義可得

$$
\log p_\beta^\star(x_{0:L},a_{1:L} \mid q)
=
\log p_{g_{\mathrm{ref}},\theta}(x_{0:L},a_{1:L} \mid q)
- \beta C_\theta(q,x_{1:L})
- \log Z_\beta(q).
$$

因此

$$
\begin{aligned}
&D_{\mathrm{KL}}\!\left(
 p(x_{0:L},a_{1:L} \mid q)
 \,\|\,
 p_\beta^\star(x_{0:L},a_{1:L} \mid q)
\right) \\
&= \mathbb E_p\left[
\log \frac{p(x_{0:L},a_{1:L} \mid q)}{p_\beta^\star(x_{0:L},a_{1:L} \mid q)}
\right] \\
&= \mathbb E_p\left[
\log \frac{p(x_{0:L},a_{1:L} \mid q)}{p_{g_{\mathrm{ref}},\theta}(x_{0:L},a_{1:L} \mid q)}
\right]
+ \beta \, \mathbb E_p\big[C_\theta(q,x_{1:L})\big]
+ \log Z_\beta(q) \\
&= D_{\mathrm{KL}}\!\left(
 p(x_{0:L},a_{1:L} \mid q)
 \,\|\,
 p_{g_{\mathrm{ref}},\theta}(x_{0:L},a_{1:L} \mid q)
\right)
+ \beta \, \mathbb E_p\big[C_\theta(q,x_{1:L})\big]
+ \log Z_\beta(q).
\end{aligned}
$$

證畢。

---

## 5. Variational RL 形式與 ideal reward 的識別

將上式套用到 $p = p_{g_\phi,\theta}(\cdot \mid q)$，則在固定 $\theta$ 下，由於 $\log Z_\beta(q)$ 不依賴於 $g_\phi$，最小化

$$
D_{\mathrm{KL}}\!\left(
 p_{g_\phi,\theta}(x_{0:L},a_{1:L} \mid q)
 \,\|\,
 p_\beta^\star(x_{0:L},a_{1:L} \mid q)
\right)
$$

等價於最小化

$$
D_{\mathrm{KL}}\!\left(
 p_{g_\phi,\theta}(x_{0:L},a_{1:L} \mid q)
 \,\|\,
 p_{g_{\mathrm{ref}},\theta}(x_{0:L},a_{1:L} \mid q)
\right)
+
\beta \, \mathbb E_{p_{g_\phi,\theta}}\big[C_\theta(q,x_{1:L})\big].
$$

等價地，也就是最大化

$$
\mathbb E_{p_{g_\phi,\theta}}\big[R_\theta^\star(q,x_{1:L})\big]
-
\frac{1}{\beta}
D_{\mathrm{KL}}\!\left(
 p_{g_\phi,\theta}(x_{0:L},a_{1:L} \mid q)
 \,\|\,
 p_{g_{\mathrm{ref}},\theta}(x_{0:L},a_{1:L} \mid q)
\right),
$$

其中 **ideal trajectory reward** 被識別為

$$
R_\theta^\star(q,x_{1:L})
:=
- C_\theta(q,x_{1:L})
=
- \sum_{n=1}^L c_\theta(q,x_n).
$$

進一步對 $q \sim \rho_Q$ 取期望，得到整體的 ideal KL-regularized objective：

$$
J_{\mathrm{ideal}}(\theta,\phi)
:=
\mathbb E_{q \sim \rho_Q}
\Big[
\mathbb E_{p_{g_\phi,\theta}(\cdot \mid q)}\big[R_\theta^\star(q,x_{1:L})\big]
-
\frac{1}{\beta}
D_{\mathrm{KL}}\!\left(
 p_{g_\phi,\theta}(\cdot \mid q)
 \,\|\,
 p_{g_{\mathrm{ref}},\theta}(\cdot \mid q)
\right)
\Big].
$$

因此，變分推導所真正證明的是：

$$
\boxed{
\text{matching the Gibbs target rollout distribution}
\quad \Longleftrightarrow \quad
\text{maximizing a KL-regularized ideal reward objective}
}
$$

而這個 ideal reward 並不是任意設計的，而是被 **唯一地識別** 為負的 cumulative local-KL cost。

---

## 6. 理想 per-step reward 與 value function

定義 ideal per-step reward 為

$$
r_\theta^\star(q,x_n)
:=
- c_\theta(q,x_n).
$$

則

$$
R_\theta^\star(q,x_{1:L})
=
\sum_{n=1}^L r_\theta^\star(q,x_n).
$$

在固定 $\theta$ 的條件下，對應的 ideal action-value function 可寫為

$$
Q_{\mathrm{IC}}^{g_\phi}(q,x_n,a_n)
:=
\mathbb E_{p_{g_\phi,\theta}}
\Big[
\sum_{k=1}^{n} r_\theta^\star(q,x_k)
\;\Big|\;
q, x_n, a_n
\Big]
=
-
\mathbb E_{p_{g_\phi,\theta}}
\Big[
\sum_{k=1}^{n} c_\theta(q,x_k)
\;\Big|\;
q, x_n, a_n
\Big].
$$

因此，長期 action quality 正是未來 inconsistency cost-to-go 的負值。

---

## Part II. 從 ideal reward 到 practical surrogate

## 7. state-level 的精確分解

對固定的 $(q,x_n)$，有

$$
\begin{aligned}
c_\theta(q,x_n)
&=
D_{\mathrm{KL}}\!\left(
 p_{\mathrm{data}}\big(x_0^{M(x_n)} \mid q, x_n\big)
 \,\|\,
 \pi_\theta\big(x_0^{M(x_n)} \mid q, x_n\big)
\right) \\
&=
\mathbb E_{x_0^{M(x_n)} \sim p_{\mathrm{data}}(\cdot \mid q,x_n)}
\left[
\log \frac{p_{\mathrm{data}}(x_0^{M(x_n)} \mid q,x_n)}{\pi_\theta(x_0^{M(x_n)} \mid q,x_n)}
\right] \\
&=
-
\mathbb E_{x_0^{M(x_n)} \sim p_{\mathrm{data}}(\cdot \mid q,x_n)}
\big[
\log \pi_\theta(x_0^{M(x_n)} \mid q,x_n)
\big]
+
H_{\mathrm{data}}(q,x_n),
\end{aligned}
$$

其中

$$
H_{\mathrm{data}}(q,x_n)
:=
H\!\left(
p_{\mathrm{data}}\big(x_0^{M(x_n)} \mid q, x_n\big)
\right).
$$

因此

$$
-c_\theta(q,x_n)
=
\mathbb E_{x_0^{M(x_n)} \sim p_{\mathrm{data}}(\cdot \mid q,x_n)}
\big[
\log \pi_\theta(x_0^{M(x_n)} \mid q,x_n)
\big]
-
H_{\mathrm{data}}(q,x_n).
$$

定義 state-level 的 expected surrogate reward 為

$$
\bar r_\theta(q,x_n)
:=
\mathbb E_{x_0^{M(x_n)} \sim p_{\mathrm{data}}(\cdot \mid q,x_n)}
\big[
\log \pi_\theta(x_0^{M(x_n)} \mid q,x_n)
\big].
$$

則有精確恆等式

$$
\boxed{
\bar r_\theta(q,x_n)
=
- c_\theta(q,x_n) + H_{\mathrm{data}}(q,x_n)
}
$$

等價地，

$$
\bar r_\theta(q,x_n) - r_\theta^\star(q,x_n)
=
H_{\mathrm{data}}(q,x_n).
$$

因此，surrogate reward 與 ideal per-state reward 的差距，正是一個 **state-dependent entropy residual**。

---

## 8. trajectory-level surrogate 與 objective gap identity

定義 trajectory-level expected surrogate reward 為

$$
\bar R_\theta(q,x_{1:L})
:=
\sum_{n=1}^L \bar r_\theta(q,x_n).
$$

則由 state-level identity 直接可得

$$
\bar R_\theta(q,x_{1:L})
=
R_\theta^\star(q,x_{1:L})
+
\mathcal H(q,x_{1:L}),
$$

其中

$$
\mathcal H(q,x_{1:L})
:=
\sum_{n=1}^L H_{\mathrm{data}}(q,x_n)
$$

表示 cumulative entropy residual。

定義 surrogate KL-regularized objective：

$$
J_{\mathrm{sur}}(\theta,\phi)
:=
\mathbb E_{q \sim \rho_Q}
\Big[
\mathbb E_{p_{g_\phi,\theta}(\cdot \mid q)}\big[\bar R_\theta(q,x_{1:L})\big]
-
\frac{1}{\beta}
D_{\mathrm{KL}}\!\left(
 p_{g_\phi,\theta}(\cdot \mid q)
 \,\|\,
 p_{g_{\mathrm{ref}},\theta}(\cdot \mid q)
\right)
\Big].
$$

則有下述精確恆等式。

### Proposition 1（objective gap identity）

對任意 $(\theta,\phi)$，都有

$$
\boxed{
J_{\mathrm{sur}}(\theta,\phi)
=
J_{\mathrm{ideal}}(\theta,\phi)
+
\mathbb E_{q \sim \rho_Q}
\Big[
\mathbb E_{p_{g_\phi,\theta}(\cdot \mid q)}\big[\mathcal H(q,x_{1:L})\big]
\Big]
}
$$

也就是說，surrogate objective 並不是任意設計的 heuristic quantity；它偏離 ideal objective 的部分，**精確地**等於 visited-state distribution 所誘導出的 expected cumulative entropy residual。

### 證明

將

$$
\bar R_\theta(q,x_{1:L}) = R_\theta^\star(q,x_{1:L}) + \mathcal H(q,x_{1:L})
$$

代入 $J_{\mathrm{sur}}$ 的定義即可。

$$
\begin{aligned}
J_{\mathrm{sur}}(\theta,\phi)
&=
\mathbb E_{q \sim \rho_Q}
\Big[
\mathbb E_{p_{g_\phi,\theta}(\cdot \mid q)}
\big[R_\theta^\star(q,x_{1:L}) + \mathcal H(q,x_{1:L})\big]
-
\frac{1}{\beta}
D_{\mathrm{KL}}\!\left(
 p_{g_\phi,\theta}(\cdot \mid q)
 \,\|\,
 p_{g_{\mathrm{ref}},\theta}(\cdot \mid q)
\right)
\Big] \\
&=
J_{\mathrm{ideal}}(\theta,\phi)
+
\mathbb E_{q \sim \rho_Q}
\Big[
\mathbb E_{p_{g_\phi,\theta}(\cdot \mid q)}\big[\mathcal H(q,x_{1:L})\big]
\Big].
\end{aligned}
$$

證畢。

---

## 9. 一個可控制的 surrogate-to-ideal transfer bound

接下來對上述 residual 在更新前後的變化給出上界。為簡潔起見，定義 joint rollout measure

$$
\widetilde p_{\theta,\phi}(q,x_{0:L},a_{1:L})
:=
\rho_Q(q) \, p_{g_\phi,\theta}(x_{0:L},a_{1:L} \mid q).
$$

假設對所有實際訪問到的 $(q,x_n)$，皆有

$$
0 \le H_{\mathrm{data}}(q,x_n) \le H_{\max} < \infty.
$$

則對所有 trajectories 都有

$$
0 \le \mathcal H(q,x_{1:L}) \le L H_{\max}.
$$

### Theorem 2（controlled transfer bound）

令 $(\theta,\phi)$ 與 $(\theta',\phi')$ 為兩組參數，並分別記

$$
\widetilde p := \widetilde p_{\theta,\phi},
\qquad
\widetilde p' := \widetilde p_{\theta',\phi'}.
$$

定義

$$
\Delta_{\mathrm{sur}}
:=
J_{\mathrm{sur}}(\theta',\phi') - J_{\mathrm{sur}}(\theta,\phi),
\qquad
\Delta_{\mathrm{ideal}}
:=
J_{\mathrm{ideal}}(\theta',\phi') - J_{\mathrm{ideal}}(\theta,\phi).
$$

則有

$$
\boxed{
\big|\Delta_{\mathrm{sur}} - \Delta_{\mathrm{ideal}}\big|
\le
L H_{\max} \, \mathrm{TV}(\widetilde p', \widetilde p)
\le
L H_{\max}\sqrt{\tfrac12 D_{\mathrm{KL}}(\widetilde p'\|\widetilde p)}
}
$$

其中 $\mathrm{TV}(\widetilde p',\widetilde p)$ 為 total variation distance。

特別地，若

$$
\Delta_{\mathrm{sur}}
>
L H_{\max}\sqrt{\tfrac12 D_{\mathrm{KL}}(\widetilde p'\|\widetilde p)},
$$

則必然有

$$
\Delta_{\mathrm{ideal}} > 0.
$$

### 證明

由 Proposition 1 可得

$$
\Delta_{\mathrm{sur}} - \Delta_{\mathrm{ideal}}
=
\mathbb E_{\widetilde p'}[\mathcal H(q,x_{1:L})]
-
\mathbb E_{\widetilde p}[\mathcal H(q,x_{1:L})].
$$

由於 $0 \le \mathcal H(q,x_{1:L}) \le L H_{\max}$，對 bounded test function 的標準不等式給出

$$
\big|
\mathbb E_{\widetilde p'}[\mathcal H]
-
\mathbb E_{\widetilde p}[\mathcal H]
\big|
\le
L H_{\max} \, \mathrm{TV}(\widetilde p',\widetilde p).
$$

再套用 Pinsker inequality，

$$
\mathrm{TV}(\widetilde p',\widetilde p)
\le
\sqrt{\tfrac12 D_{\mathrm{KL}}(\widetilde p'\|\widetilde p)},
$$

即可得證。

---

## 10. sample-based observable surrogate

若對一個給定的 $(q,x_n)$，抽取 oracle masked target

$$
x_0^{\star, M(x_n)}
\sim
p_{\mathrm{data}}(\cdot \mid q, x_n),
$$

並定義可觀測的 per-state reward 為

$$
\hat r_\theta(q,x_n,x_0^\star)
:=
\log \pi_\theta(x_0^{\star, M(x_n)} \mid q,x_n),
$$

則有

$$
\mathbb E_{x_0^{\star, M(x_n)} \sim p_{\mathrm{data}}(\cdot \mid q,x_n)}
\big[
\hat r_\theta(q,x_n,x_0^\star)
\big]
=
\bar r_\theta(q,x_n)
=
- c_\theta(q,x_n) + H_{\mathrm{data}}(q,x_n).
$$

因此，$\hat r_\theta$ 是 $\bar r_\theta(q,x_n)$ 的無偏估計量；而 practical surrogate 與 ideal reward 之間的偏差，仍然由同一個 entropy residual 所支配。

從 rollout level 來看，對應的 observable surrogate reward 可以寫成

$$
\hat R_\theta(q,x_{1:L},x_0^\star)
:=
\sum_{n=1}^L \hat r_\theta(q,x_n,x_0^\star),
$$

它可以被理解為某個帶噪的 rollout-level surrogate realization；然而其相對於 ideal objective 的偏差，並不是任意的，而是透過前述 decomposition 與 transfer bound 被顯式控制。

---

## 11. 從 on-policy local KL 到 terminal correctness

到目前為止，我們已經證明：

1. local posterior inconsistency KL 可被 lift 成 rollout-level cost $C_\theta$；
2. minimizing the corresponding Gibbs-matching objective 等價於 maximizing a KL-regularized ideal reward objective；
3. practical surrogate objective 與 ideal objective 之間的差距可由 entropy residual 與 distribution shift 控制。

然而，上述結論仍然停留在 **consistency / reward / objective** 的層次，尚未直接觸及 task-level correctness。為了將優化目標進一步橋接到 correctness，我們額外引入一個 **local-to-terminal bridge assumption**。

### 11.1 terminal correctness 與 error rate

令 terminal evaluator 為

$$
\mathrm{Corr}(q,x_0) \in \{0,1\},
$$

其中 $\mathrm{Corr}(q,x_0)=1$ 表示最終生成之 clean output $x_0$ 在 prompt $q$ 上被判定為 correct。

定義 terminal accuracy 與 error rate 為

$$
\mathrm{Acc}(\theta,\phi)
:=
\mathbb E_{q \sim \rho_Q}
\Big[
\mathbb E_{(x_{0:L},a_{1:L})\sim p_{g_\phi,\theta}(\cdot\mid q)}
\big[
\mathrm{Corr}(q,x_0)
\big]
\Big],
$$

以及

$$
\mathrm{Err}(\theta,\phi)
:=
1-\mathrm{Acc}(\theta,\phi).
$$

此外，對每個 step $n$，記當前 policy 所誘導出的第 $n$ 步 inference-state visitation distribution 為

$$
d_{\theta,\phi}^{(n)}(x_n\mid q).
$$

### 11.2 Local-to-terminal bridge assumption

我們假設存在常數 $\alpha_1,\dots,\alpha_L \ge 0$ 與不可避免誤差項 $\xi \ge 0$，使得對任意 $(\theta,\phi)$，皆有

$$
\mathrm{Err}(\theta,\phi)
\le
\xi
+
\sum_{n=1}^L
\alpha_n\,
\mathbb E_{q \sim \rho_Q,\; x_n \sim d_{\theta,\phi}^{(n)}(\cdot\mid q)}
\Big[
\mathrm{TV}\!\Big(
p_{\mathrm{data}}(\cdot\mid q,x_n),
\pi_\theta(\cdot\mid q,x_n)
\Big)
\Big].
$$

這個假設的意義是：

> 最終答錯率可以被 learner 在 inference 時實際訪問到的 masked states 上，對 oracle posterior 的局部失配程度所上界。

這一步是本文新增的 task bridge：它不再只談 local KL 本身，而是明確假設 **terminal correctness 對 inference-state local mismatch 敏感**。

### 11.3 correctness bridge theorem

由 Pinsker inequality，

$$
\mathrm{TV}(P,Q)
\le
\sqrt{\tfrac12 D_{\mathrm{KL}}(P\|Q)},
$$

而本文的局部 inconsistency cost 正是

$$
c_\theta(q,x_n)
=
D_{\mathrm{KL}}\!\Big(
p_{\mathrm{data}}(\cdot\mid q,x_n)
\;\|\;
\pi_\theta(\cdot\mid q,x_n)
\Big). \tag{11.1}
$$

因此

$$
\mathrm{Err}(\theta,\phi)
\le
\xi
+
\sum_{n=1}^L
\alpha_n\,
\mathbb E_{q,x_n}
\Big[
\sqrt{\tfrac12 c_\theta(q,x_n)}
\Big].
$$

再由 Jensen 與 Cauchy–Schwarz，不等式可進一步化為

$$
\mathrm{Err}(\theta,\phi)
\le
\xi
+
\frac{\|\alpha\|_2}{\sqrt 2}
\sqrt{
\sum_{n=1}^L
\mathbb E_{q,x_n}[c_\theta(q,x_n)]
},
$$

而由第 2 節的定義，

$$
J_{\mathrm{IC}}(\theta,\phi)
=
\mathbb E_{q}
\Big[
\mathbb E_{p_{g_\phi,\theta}(\cdot\mid q)}
\big[
\sum_{n=1}^L c_\theta(q,x_n)
\big]
\Big]. \tag{11.2}
$$

故得到下述結果。

### Theorem 3（KL-to-correctness bridge）

在上述 local-to-terminal bridge assumption 下，對任意 $(\theta,\phi)$，

$$
\boxed{
\mathrm{Err}(\theta,\phi)
\le
\xi
+
\frac{\|\alpha\|_2}{\sqrt 2}
\sqrt{J_{\mathrm{IC}}(\theta,\phi)}
}
$$

等價地，

$$
\boxed{
\mathrm{Acc}(\theta,\phi)
\ge
1
-
\xi
-
\frac{\|\alpha\|_2}{\sqrt 2}
\sqrt{J_{\mathrm{IC}}(\theta,\phi)}
}
$$

因此，**降低 on-policy cumulative local KL cost，會提升 terminal accuracy 的下界**。

### 證明

由 local-to-terminal bridge assumption 與 Pinsker inequality，

$$
\mathrm{Err}(\theta,\phi)
\le
\xi
+
\sum_{n=1}^L
\alpha_n
\mathbb E_{q,x_n}
\Big[
\sqrt{\tfrac12 c_\theta(q,x_n)}
\Big].
$$

由 Jensen inequality，

$$
\mathbb E[\sqrt{c_\theta}]
\le
\sqrt{\mathbb E[c_\theta]}.
$$

因此

$$
\mathrm{Err}(\theta,\phi)
\le
\xi
+
\frac{1}{\sqrt 2}
\sum_{n=1}^L
\alpha_n
\sqrt{
\mathbb E_{q,x_n}[c_\theta(q,x_n)]
}.
$$

再由 Cauchy–Schwarz，

$$
\sum_{n=1}^L
\alpha_n
\sqrt{\mathbb E[c_\theta(q,x_n)]}
\le
\|\alpha\|_2
\sqrt{
\sum_{n=1}^L \mathbb E[c_\theta(q,x_n)]
}.
$$

結合 $J_{\mathrm{IC}}$ 的定義即可得證。

---

## 12. 從 surrogate optimization 到 correctness lower bound

第 5 節已證明，ideal KL-regularized reward objective 可寫為

$$
J_{\mathrm{ideal}}(\theta,\phi)
=
-
J_{\mathrm{IC}}(\theta,\phi)
-
\mathcal R(\theta,\phi),
$$

其中

$$
\mathcal R(\theta,\phi)
:=
\frac{1}{\beta}
\mathbb E_{q \sim \rho_Q}
\Big[
D_{\mathrm{KL}}\!\big(
p_{g_\phi,\theta}(\cdot\mid q)
\;\|\;
p_{g_{\mathrm{ref}},\theta}(\cdot\mid q)
\big)
\Big].
$$

也就是說，maximizing $J_{\mathrm{ideal}}$ 等價於 minimizing cumulative local KL cost $J_{\mathrm{IC}}$ 與 reference-regularization term $\mathcal R$ 的和。

另一方面，第 8–9 節已證明 practical surrogate objective $J_{\mathrm{sur}}$ 與 $J_{\mathrm{ideal}}$ 的差距可由 entropy residual 與 rollout-measure shift 控制。

因此，只要再控制 regularization term 的變動，便可將 surrogate improvement 進一步推到 correctness lower bound。

### 12.1 objective decomposition

對兩組參數 $(\theta,\phi)$ 與 $(\theta',\phi')$，定義

$$
\Delta_{\mathrm{IC}}
:=
J_{\mathrm{IC}}(\theta',\phi') - J_{\mathrm{IC}}(\theta,\phi),
$$

$$
\Delta_{\mathcal R}
:=
\mathcal R(\theta',\phi') - \mathcal R(\theta,\phi),
$$

$$
\Delta_{\mathrm{ideal}}
:=
J_{\mathrm{ideal}}(\theta',\phi') - J_{\mathrm{ideal}}(\theta,\phi).
$$

則由 $J_{\mathrm{ideal}}=-J_{\mathrm{IC}}-\mathcal R$ 可得

$$
\Delta_{\mathrm{ideal}}
=
-
\Delta_{\mathrm{IC}}
-
\Delta_{\mathcal R}. \tag{12.1}
$$

因此，只要 $\Delta_{\mathcal R}$ 不會上升太多，sufficiently positive 的 $\Delta_{\mathrm{ideal}}$ 就會迫使 $\Delta_{\mathrm{IC}}<0$。

### 12.2 bounded-regularizer assumption

假設在一次局部更新中，reference-regularization term 的增量滿足

$$
\Delta_{\mathcal R} \le \eta
$$

對某個 $\eta \ge 0$。

則由 (12.1) 立即得到：若

$$
\Delta_{\mathrm{ideal}} > \eta,
$$

則必然有

$$
\Delta_{\mathrm{IC}} < 0.
$$

而由第 9 節的 Theorem 2，若

$$
\Delta_{\mathrm{sur}}
>
L H_{\max}\sqrt{\tfrac12 D_{\mathrm{KL}}(\widetilde p'\|\widetilde p)}
+
\eta,
$$

則有 $\Delta_{\mathrm{ideal}}>\eta$，進而得到 $\Delta_{\mathrm{IC}}<0$。

因此可得下述 corollary。

### Corollary 4（surrogate improvement implies accuracy-bound improvement）

在 Theorem 2 與 Theorem 3 的條件下，若一次更新滿足

$$
\Delta_{\mathrm{sur}}
>
L H_{\max}\sqrt{\tfrac12 D_{\mathrm{KL}}(\widetilde p'\|\widetilde p)}
+
\eta,
$$

且 $\Delta_{\mathcal R}\le \eta$，則

$$
J_{\mathrm{IC}}(\theta',\phi')
<
J_{\mathrm{IC}}(\theta,\phi),
$$

從而

$$
\mathrm{Acc}(\theta',\phi')
\ge
1-\xi-\frac{\|\alpha\|_2}{\sqrt 2}\sqrt{J_{\mathrm{IC}}(\theta',\phi')}
>
1-\xi-\frac{\|\alpha\|_2}{\sqrt 2}\sqrt{J_{\mathrm{IC}}(\theta,\phi)}.
$$

亦即，**sufficiently large 的 practical surrogate improvement，在 controlled distribution shift 與 bounded regularizer drift 下，會提升 terminal correctness 的可證下界。**

---

## 13. 詮釋與限制（改寫版）

因此，本文的理論圖像可分為三層。

### 一致性層

$$
\text{local conditional KL}
\;\Longrightarrow\;
\text{trajectory cost } C_\theta
\;\Longrightarrow\;
\text{Gibbs target } p_\beta^\star
\;\Longrightarrow\;
\text{ideal reward } R_\theta^\star=-C_\theta.
$$

這一層說明：若 desideratum 是降低 local posterior inconsistency，則對應的 ideal rollout reward 被唯一識別為負的 cumulative local-KL cost。

### 近似優化層

practical surrogate 並不逐點等於 ideal reward；但其偏差具有精確 decomposition，並可由 entropy residual 與 rollout-measure shift 控制。

### 任務層

在額外的 local-to-terminal bridge assumption 下，on-policy cumulative local KL cost 進一步控制 terminal error 的上界，因此也控制 terminal accuracy 的下界。

因此，本文最終得到的不是

$$
\text{optimize surrogate}
\equiv
\text{optimize correctness}
$$

這種過強命題，而是一個較弱但可防守的結論：

> reducing learner-induced local posterior inconsistency improves a certified lower bound on terminal correctness, provided that terminal error is controlled by the cumulative local mismatch on inference-time visited states.

---

## 14. 一句話總結（改寫版）

$$
\boxed{
\text{local conditional KL}
\;\Longrightarrow\;
\text{ideal rollout reward}
\;\Longrightarrow\;
\text{practical surrogate tracks it within a controlled gap}
\;\Longrightarrow\;
\text{under a local-to-terminal bridge, improved terminal correctness lower bound}
}
$$
