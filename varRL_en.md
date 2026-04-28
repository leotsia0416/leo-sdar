# A Variational RL Bridge from Local Posterior-Inconsistency KL to Rollout Reward (UPO / MDLM-Style Notation)

## Goal

This note develops a mathematically clean bridge, written in notation closer to the discrete masked-diffusion literature, from a **local conditional posterior-inconsistency KL** to an **ideal rollout-level reward objective** with a variational / control-as-inference interpretation. We then further show that, in practice, the gap between the optimized **surrogate reward objective** and the ideal objective admits an explicit decomposition and can be controlled under mild conditions.

We adopt notation closer to UPO-style work, while using probability expressions that resemble common MDLM / ReMDM conventions:

1. $q \sim \rho_Q$ denotes a sampled prompt / context;
2. $x_0 \sim p_{\mathrm{data}}(\cdot \mid q)$ denotes a clean target sequence;
3. $q(x_n \mid x_0)$ denotes a forward masking / corruption kernel, where $x_n \in \mathcal X_n$ is a partially masked sequence with $n$ masks;
4. $\pi_\theta$ denotes the MDM denoiser / reverse conditional;
5. $g_\phi$ denotes the remask / unmask position-selection policy;
6. $p_{g_\phi,\theta}(x_{0:L}, a_{1:L} \mid q)$ denotes the rollout distribution jointly induced by policy $g_\phi$ and base model $\pi_\theta$.

The key ideas are as follows:

1. the primitive inconsistency quantity is a **state-wise local conditional KL**, not a trajectory-wise KL;
2. this local KL can be accumulated along visited masked states in a rollout and thereby **lifted** to a trajectory cost;
3. matching the Gibbs target trajectory distribution induced by this cost is equivalent to maximizing a **KL-regularized ideal reward objective**;
4. practical training usually optimizes an observable surrogate reward rather than the exact ideal reward;
5. the gap between the surrogate and the ideal admits an **exact decomposition**, which in turn yields a **surrogate-to-ideal transfer bound**.

---

## Part I. Ideal Variational Bridge

## 1. Notation and Local Conditional KL

Let the vocabulary be $\mathcal V$, and let the target sequence length be $L$. The clean sequence space is

$$
\mathcal X = \mathcal V^L.
$$

For each $n \in \{0,1,\dots,L\}$, define

$$
\mathcal X_n
:=
\{x \in (\mathcal V \cup \{M\})^L : \text{$x$ contains exactly $n$ mask tokens}\},
$$

and let $x_n \in \mathcal X_n$ denote a partially masked sequence with $n$ masks. Let

$$
\mathcal A_{x_n} = \{a_1[x_n],\dots,a_n[x_n]\}
$$

denote the set of masked positions in $x_n$. For brevity, we suppress $[x_n]$ below and simply write $a_i$.

Given a prompt / context $q \sim \rho_Q$, the clean target sequence is sampled as

$$
x_0 \sim p_{\mathrm{data}}(\cdot \mid q).
$$

Following MDLM / ReMDM-style notation, let

$$
q(x_n \mid x_0)
$$

denote the forward masking kernel that corrupts $x_0$ into $x_n$. The masked-state distribution induced by training-time corruption is then

$$
q_{\mathrm{data}}(x_n \mid q)
:=
\sum_{x_0 \in \mathcal X}
p_{\mathrm{data}}(x_0 \mid q) \, q(x_n \mid x_0).
$$

For a given masked state $x_n$, let $M(x_n)$ denote its masked coordinates. The oracle data posterior is written as

$$
p_{\mathrm{data}}\big(x_0^{M(x_n)} \mid q, x_n\big),
$$

and the corresponding model-induced conditional distribution at the same state is written as

$$
\pi_\theta\big(x_0^{M(x_n)} \mid q, x_n\big).
$$

It is crucial to emphasize that

$$
p_{\mathrm{data}}\big(x_0^{M(x_n)} \mid q, x_n\big)
\quad\text{and}\quad
\pi_\theta\big(x_0^{M(x_n)} \mid q, x_n\big)
$$

are **not** trajectory distributions; they are **local conditional distributions over the masked clean content at a fixed masked state $x_n$**.

We therefore define the **state-wise posterior inconsistency cost** by

$$
c_\theta(q, x_n)
:=
D_{\mathrm{KL}}\!\left(
 p_{\mathrm{data}}\big(x_0^{M(x_n)} \mid q, x_n\big)
 \,\|\,
 \pi_\theta\big(x_0^{M(x_n)} \mid q, x_n\big)
\right).
$$

This is the true primitive ideal quantity in this note: it is a **local conditional KL**, not a rollout KL and not a terminal-output KL.

Moreover, $x_n$ may arise from two distinct sources:

1. **training-time states**: generated from $p_{\mathrm{data}}(x_0\mid q)$ and the forward kernel $q(x_n\mid x_0)$;
2. **inference-time visited states**: actually visited by the rollout induced by the current $(\theta,\phi)$.

Therefore, $c_\theta(q,x_n)$ should be regarded as an **ideal local objective**. When inference-time visited states and training-time corruption states differ substantially, it need not be exactly observable at every visited $x_n$.

---

## 2. Rollout Dynamics and Trajectory Cost

At inference time, we start from the fully masked target $x_L = M^L$ and iteratively unmask. For $n=L,L-1,\dots,1$, at state $x_n$:

- the action space is $\mathcal A_{x_n}$;
- the policy head $g_\phi(\cdot \mid q, x_n)$ selects a position $a_n \in \mathcal A_{x_n}$;
- the base MDM $\pi_\theta$ produces a token distribution conditioned on $(q,x_n,a_n)$ and yields the next state $x_{n-1}$.

In the same style as UPO-like work, the one-step transition can be written as

$$
p_{g_\phi,\theta}(x_{n-1} \mid x_n, q)
=
T_n^{(g_\phi,\theta)}(x_n, x_{n-1} \mid q)
=
 g_\phi(a_n \mid q, x_n)
 \, \pi_\theta(x_{n-1} \mid q, x_n, a_n),
$$

where, in the discrete masked-diffusion setting, $\pi_\theta(x_{n-1} \mid q, x_n, a_n)$ means that only coordinate $a_n$ is replaced by a sampled token, while all other coordinates remain unchanged.

The full rollout distribution conditioned on $q$ can therefore be written as

$$
p_{g_\phi,\theta}(x_{0:L}, a_{1:L} \mid q)
:=
\prod_{n=1}^L
 g_\phi(a_n \mid q, x_n)
 \, \pi_\theta(x_{n-1} \mid q, x_n, a_n).
$$

We now accumulate the local KL along the rollout and define the cumulative inconsistency cost:

$$
C_\theta(q, x_{1:L})
:=
\sum_{n=1}^L c_\theta(q, x_n).
$$

Importantly, $C_\theta$ is obtained by **lifting a local conditional KL to the trajectory level**; it is not introduced as a primitive trajectory-level data/model KL.

The corresponding ideal inference-consistency objective is

$$
J_{\mathrm{IC}}(\theta,\phi)
:=
\mathbb E_{q \sim \rho_Q}
\Big[
\mathbb E_{(x_{0:L},a_{1:L}) \sim p_{g_\phi,\theta}(\cdot \mid q)}
\big[C_\theta(q,x_{1:L})\big]
\Big].
$$

Thus, the original local KL objective is naturally lifted into a rollout-level cost-minimization problem.

---

## 3. Reference Rollout Distribution and Gibbs Target

Let $g_{\mathrm{ref}}$ be a reference unmasking policy, and denote by

$$
p_{g_{\mathrm{ref}},\theta}(x_{0:L}, a_{1:L} \mid q)
$$

the corresponding reference rollout distribution induced jointly with the base model $\pi_\theta$.

For a temperature parameter $\beta > 0$, define the **Gibbs target trajectory distribution**:

$$
p_\beta^\star(x_{0:L}, a_{1:L} \mid q)
:=
\frac{1}{Z_\beta(q)}
\, p_{g_{\mathrm{ref}},\theta}(x_{0:L}, a_{1:L} \mid q)
\exp\!\big(-\beta C_\theta(q,x_{1:L})\big),
$$

where the partition function is

$$
Z_\beta(q)
:=
\mathbb E_{(x_{0:L},a_{1:L}) \sim p_{g_{\mathrm{ref}},\theta}(\cdot \mid q)}
\big[
\exp(-\beta C_\theta(q,x_{1:L}))
\big].
$$

The construction has the following meaning:

- the reference rollout distribution preserves the baseline denoising dynamics and reference scheduling bias;
- the exponential tilt $\exp(-\beta C_\theta)$ upweights trajectories with smaller cumulative inconsistency;
- consequently, $p_\beta^\star$ is an **ideal rollout distribution that prefers trajectories with lower accumulated local-KL cost**.

Note that when $\theta$ is also updated during training, $c_\theta$, $C_\theta$, $p_{g_{\mathrm{ref}},\theta}$, and $p_\beta^\star$ all change accordingly. Hence the variational equivalence below should be interpreted as an ideal statement **conditional on the current fixed $\theta$**.

---

## 4. Variational Identity

For any rollout distribution $p(\cdot \mid q)$, we have

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

### Proof

By definition,

$$
\log p_\beta^\star(x_{0:L},a_{1:L} \mid q)
=
\log p_{g_{\mathrm{ref}},\theta}(x_{0:L},a_{1:L} \mid q)
- \beta C_\theta(q,x_{1:L})
- \log Z_\beta(q).
$$

Therefore,

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

QED.

---

## 5. Variational RL Form and Identification of the Ideal Reward

Applying the above identity to $p = p_{g_\phi,\theta}(\cdot \mid q)$, and fixing $\theta$, we note that $\log Z_\beta(q)$ does not depend on $g_\phi$. Hence minimizing

$$
D_{\mathrm{KL}}\!\left(
 p_{g_\phi,\theta}(x_{0:L},a_{1:L} \mid q)
 \,\|\,
 p_\beta^\star(x_{0:L},a_{1:L} \mid q)
\right)
$$

is equivalent to minimizing

$$
D_{\mathrm{KL}}\!\left(
 p_{g_\phi,\theta}(x_{0:L},a_{1:L} \mid q)
 \,\|\,
 p_{g_{\mathrm{ref}},\theta}(x_{0:L},a_{1:L} \mid q)
\right)
+
\beta \, \mathbb E_{p_{g_\phi,\theta}}\big[C_\theta(q,x_{1:L})\big].
$$

Equivalently, this is the same as maximizing

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

where the **ideal trajectory reward** is identified as

$$
R_\theta^\star(q,x_{1:L})
:=
- C_\theta(q,x_{1:L})
=
- \sum_{n=1}^L c_\theta(q,x_n).
$$

Taking expectation over $q \sim \rho_Q$, we obtain the global ideal KL-regularized objective:

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

Therefore, what the variational derivation truly proves is

$$
\boxed{
\text{matching the Gibbs target rollout distribution}
\quad \Longleftrightarrow \quad
\text{maximizing a KL-regularized ideal reward objective}
}
$$

and this ideal reward is not arbitrary: it is **uniquely identified** as the negative cumulative local-KL cost.

---

## 6. Ideal Per-Step Reward and Value Function

Define the ideal per-step reward by

$$
r_\theta^\star(q,x_n)
:=
- c_\theta(q,x_n).
$$

Then

$$
R_\theta^\star(q,x_{1:L})
=
\sum_{n=1}^L r_\theta^\star(q,x_n).
$$

Under fixed $\theta$, the corresponding ideal action-value function can be written as

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

Thus the long-horizon action quality is precisely the negative future inconsistency cost-to-go.

---

## Part II. From the Ideal Reward to a Practical Surrogate

## 7. Exact State-Level Decomposition

For fixed $(q,x_n)$, we have

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

where

$$
H_{\mathrm{data}}(q,x_n)
:=
H\!\left(
p_{\mathrm{data}}\big(x_0^{M(x_n)} \mid q, x_n\big)
\right).
$$

Therefore,

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

Define the state-level expected surrogate reward by

$$
\bar r_\theta(q,x_n)
:=
\mathbb E_{x_0^{M(x_n)} \sim p_{\mathrm{data}}(\cdot \mid q,x_n)}
\big[
\log \pi_\theta(x_0^{M(x_n)} \mid q,x_n)
\big].
$$

Then we obtain the exact identity

$$
\boxed{
\bar r_\theta(q,x_n)
=
- c_\theta(q,x_n) + H_{\mathrm{data}}(q,x_n)
}
$$

or equivalently,

$$
\bar r_\theta(q,x_n) - r_\theta^\star(q,x_n)
=
H_{\mathrm{data}}(q,x_n).
$$

Hence the gap between the surrogate reward and the ideal per-state reward is exactly a **state-dependent entropy residual**.

---

## 8. Trajectory-Level Surrogate and Objective Gap Identity

Define the trajectory-level expected surrogate reward by

$$
\bar R_\theta(q,x_{1:L})
:=
\sum_{n=1}^L \bar r_\theta(q,x_n).
$$

Then the state-level identity immediately gives

$$
\bar R_\theta(q,x_{1:L})
=
R_\theta^\star(q,x_{1:L})
+
\mathcal H(q,x_{1:L}),
$$

where

$$
\mathcal H(q,x_{1:L})
:=
\sum_{n=1}^L H_{\mathrm{data}}(q,x_n)
$$

denotes the cumulative entropy residual.

Define the surrogate KL-regularized objective by

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

We then have the following exact identity.

### Proposition 1 (Objective Gap Identity)

For any $(\theta,\phi)$,

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

That is, the surrogate objective is not an ad hoc heuristic quantity; its deviation from the ideal objective is **exactly** the expected cumulative entropy residual induced by the visited-state distribution.

### Proof

Substitute

$$
\bar R_\theta(q,x_{1:L}) = R_\theta^\star(q,x_{1:L}) + \mathcal H(q,x_{1:L})
$$

into the definition of $J_{\mathrm{sur}}$:

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

QED.

---

## 9. A Controlled Surrogate-to-Ideal Transfer Bound

We now upper-bound the change of the above residual across updates. For brevity, define the joint rollout measure

$$
\widetilde p_{\theta,\phi}(q,x_{0:L},a_{1:L})
:=
\rho_Q(q) \, p_{g_\phi,\theta}(x_{0:L},a_{1:L} \mid q).
$$

Assume that for all practically visited $(q,x_n)$,

$$
0 \le H_{\mathrm{data}}(q,x_n) \le H_{\max} < \infty.
$$

Then for all trajectories,

$$
0 \le \mathcal H(q,x_{1:L}) \le L H_{\max}.
$$

### Theorem 2 (Controlled Transfer Bound)

Let $(\theta,\phi)$ and $(\theta',\phi')$ be two parameter settings, and define

$$
\widetilde p := \widetilde p_{\theta,\phi},
\qquad
\widetilde p' := \widetilde p_{\theta',\phi'}.
$$

Also define

$$
\Delta_{\mathrm{sur}}
:=
J_{\mathrm{sur}}(\theta',\phi') - J_{\mathrm{sur}}(\theta,\phi),
\qquad
\Delta_{\mathrm{ideal}}
:=
J_{\mathrm{ideal}}(\theta',\phi') - J_{\mathrm{ideal}}(\theta,\phi).
$$

Then

$$
\boxed{
\big|\Delta_{\mathrm{sur}} - \Delta_{\mathrm{ideal}}\big|
\le
L H_{\max} \, \mathrm{TV}(\widetilde p', \widetilde p)
\le
L H_{\max}\sqrt{\tfrac12 D_{\mathrm{KL}}(\widetilde p'\|\widetilde p)}
}
$$

where $\mathrm{TV}(\widetilde p',\widetilde p)$ denotes the total variation distance.

In particular, if

$$
\Delta_{\mathrm{sur}}
>
L H_{\max}\sqrt{\tfrac12 D_{\mathrm{KL}}(\widetilde p'\|\widetilde p)},
$$

then necessarily

$$
\Delta_{\mathrm{ideal}} > 0.
$$

### Proof

By Proposition 1,

$$
\Delta_{\mathrm{sur}} - \Delta_{\mathrm{ideal}}
=
\mathbb E_{\widetilde p'}[\mathcal H(q,x_{1:L})]
-
\mathbb E_{\widetilde p}[\mathcal H(q,x_{1:L})].
$$

Since $0 \le \mathcal H(q,x_{1:L}) \le L H_{\max}$, the standard inequality for bounded test functions gives

$$
\big|
\mathbb E_{\widetilde p'}[\mathcal H]
-
\mathbb E_{\widetilde p}[\mathcal H]
\big|
\le
L H_{\max} \, \mathrm{TV}(\widetilde p',\widetilde p).
$$

Applying Pinsker’s inequality,

$$
\mathrm{TV}(\widetilde p',\widetilde p)
\le
\sqrt{\tfrac12 D_{\mathrm{KL}}(\widetilde p'\|\widetilde p)},
$$

completes the proof.

---

## 10. Sample-Based Observable Surrogate

For a given $(q,x_n)$, if we draw an oracle masked target

$$
x_0^{\star, M(x_n)}
\sim
p_{\mathrm{data}}(\cdot \mid q, x_n),
$$

and define the observable per-state reward by

$$
\hat r_\theta(q,x_n,x_0^\star)
:=
\log \pi_\theta(x_0^{\star, M(x_n)} \mid q,x_n),
$$

then

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

Hence $\hat r_\theta$ is an unbiased estimator of $\bar r_\theta(q,x_n)$; the gap between the practical surrogate and the ideal reward is still governed by the same entropy residual.

At the rollout level, the corresponding observable surrogate reward can be written as

$$
\hat R_\theta(q,x_{1:L},x_0^\star)
:=
\sum_{n=1}^L \hat r_\theta(q,x_n,x_0^\star),
$$

which may be viewed as a noisy rollout-level realization of the surrogate. However, its deviation from the ideal objective is not arbitrary; it is explicitly controlled by the above decomposition and transfer bound.

---

## 11. Interpretation and Limitations

The theoretical picture in this note therefore has two layers.

### Ideal layer

The variational construction is exact:

$$
\text{local conditional KL}
\;\Longrightarrow\;
\text{trajectory cost } C_\theta
\;\Longrightarrow\;
\text{Gibbs target } p_\beta^\star
\;\Longrightarrow\;
\text{ideal reward } R_\theta^\star = - C_\theta.
$$

This layer answers the question: **if the original desideratum is to reduce local posterior inconsistency, what should the corresponding ideal rollout-level reward be?**

### Practical layer

The actually trainable surrogate reward is not exactly the same as the ideal reward. Still, it is not an unprincipled heuristic quantity, because:

1. it admits an **exact state-level decomposition**;
2. it induces an **exact trajectory-level objective gap identity**;
3. improvement transfer from the surrogate objective to the ideal objective is controlled by a visitation-dependent entropy residual, which in turn can be upper-bounded by the distributional shift between rollout measures.

Hence the conclusion obtained here is weaker than

$$
\text{optimize practical surrogate}
\equiv
\text{optimize ideal KL objective},
$$

but also more honest and more suitable for the joint-training setting:

> practical surrogate optimization is not pointwise identical to ideal objective optimization; however, their discrepancy is not arbitrary. It has explicit structure and is controllable by the distributional shift in a local-update regime.

---

## 12. One-Line Summary

$$
\boxed{
\text{local conditional KL}
\;\Longrightarrow\;
\text{via variational lifting yields the ideal rollout reward},
\qquad
\text{practical surrogate}
\;\Longrightarrow\;
\text{corresponds to the ideal objective up to a controllable entropy-residual gap}
}
$$
