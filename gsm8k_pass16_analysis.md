# SDAR-1.7B Base Model 的 GSM8K Pass@16 分析

## 實驗設定

- 模型：`Models/SDAR-1.7B-Chat-`
- 評測集：GSM8K test，共 `1319` 題
- 推論 backend：`lmdeploy`
- 比較基準：
  - greedy / deterministic baseline：`80.06%`
  - sampled `pass@16`：每題跑 `16` 個不同 seed，只要其中一次答對就算通過
- sampled 設定：
  - `temperature=0.8`
  - `top_p=0.95`
  - `top_k=50`
  - seeds：`1..16`

Pass@16 聚合結果輸出在：

- `outputs/gsm8k_pass16_sdar_base_lmdeploy_20260424_194404/gsm8k_pass_at_16.txt`
- `outputs/gsm8k_pass16_sdar_base_lmdeploy_20260424_194404/gsm8k_pass_at_16.json`

## 核心結果

| 指標 | 分數 |
|---|---:|
| greedy baseline | `80.06%` |
| sampled 單次平均 | `75.91%` |
| sampled 單次最佳 | `76.80%` |
| sampled 單次最差 | `74.60%` |
| `Pass@16` | **`96.21%`** |

換成題數來看：

- greedy baseline：約 `1056/1319`
- `Pass@16`：`1269/1319`

也就是說，和 greedy 相比，`pass@16` 額外救回了大量題目。

## 直接解讀

這個結果很關鍵，因為它呈現出一個很鮮明的形狀：

- **單次 sampled pass@1 明顯比 greedy 差**
- **但 oracle `pass@16` 非常高**

這代表：

1. `SDAR-1.7B` base model 並不是「不會做」很多 GSM8K 題目。
2. 問題更像是：**模型常常有正確路徑，但單次抽樣很不穩定。**
3. 因此後續如果有：
   - verifier
   - reranker
   - self-consistency
   - 更準的 remask / local repair

就理論上還有很大的改善空間。

## 16 個 Seed 的單次分數

| seed | accuracy |
|---|---:|
| `1` | `76.12%` |
| `2` | `76.80%` |
| `3` | `76.27%` |
| `4` | `75.21%` |
| `5` | `76.72%` |
| `6` | `75.89%` |
| `7` | `75.89%` |
| `8` | `75.66%` |
| `9` | `76.27%` |
| `10` | `75.97%` |
| `11` | `75.51%` |
| `12` | `75.59%` |
| `13` | `74.60%` |
| `14` | `75.66%` |
| `15` | `75.89%` |
| `16` | `76.42%` |

重點是：

- sampled 單次最好也只有 `76.80%`
- 但 `pass@16` 卻能到 `96.21%`

所以「多樣性」本身非常強，但「單次取樣品質」不夠穩。

## 逐題命中分布

這裡的 `num_correct` 指的是：同一題在 `16` 個 seed 裡，有幾次答對。

| num_correct | 題數 |
|---|---:|
| `0` | `50` |
| `1` | `33` |
| `2` | `36` |
| `3` | `15` |
| `4` | `20` |
| `5` | `29` |
| `6` | `34` |
| `7` | `33` |
| `8` | `44` |
| `9` | `40` |
| `10` | `25` |
| `11` | `53` |
| `12` | `57` |
| `13` | `63` |
| `14` | `118` |
| `15` | `195` |
| `16` | `474` |

把它合併成比較容易理解的區間：

| 區間 | 題數 | 解讀 |
|---|---:|---|
| `0/16` | `50` | 16 次都錯，最硬的一群 |
| `1/16` | `33` | 極不穩，偶爾才會撞對 |
| `2-4/16` | `71` | 不穩，但已有可救訊號 |
| `5-11/16` | `258` | 明顯可救，路徑品質波動大 |
| `12-15/16` | `433` | 幾乎會，但仍偶爾失手 |
| `16/16` | `474` | 非常穩定，所有 seed 都對 |

這個分布說明：

- 真正完全無法解的題目只有 `50` 題左右
- 很多題是「會做，但不穩」
- 這正是 search / rerank / remask 最有價值的地帶

## 和 Greedy 的逐題比較

把 greedy `80.06%` 跟 `pass@16` 對齊後：

| 類型 | 題數 |
|---|---:|
| greedy 錯，`pass@16` 救回來 | `214` |
| greedy 錯，`pass@16` 仍然錯 | `49` |
| greedy 對，但 16 次 sampled 全錯 | `1` |

這代表：

- `pass@16` 把 greedy 原本錯的題裡，大多數都救回來了
- 只剩下 `49` 題是「greedy 也錯，16 次抽樣也全錯」

唯一一題特殊 case 是：

- `gsm8k_test_1288`
- greedy 對
- 但 16 次 sampled 全錯

這也提醒一件事：**直接用 sampling 當部署策略不一定好**，因為它會把一些原本 greedy 很穩的簡單題弄壞。

## 最常錯的題型

下面是 `16` 次都錯的 `50` 題，用關鍵字做的粗分類結果：

| 類型 | 題數 |
|---|---:|
| `time/calendar` | `34` |
| `money/transaction` | `21` |
| `percent/compound` | `14` |
| `ratio/fraction` | `13` |
| `age/timeline` | `7` |
| `state tracking` | `6` |
| `average/target total` | `4` |
| `other` | `4` |
| `probability` | `1` |
| `geometry/layout` | `1` |

如果看 `<=1 hit`，也就是 `16` 次裡最多只對 `1` 次的困難題群：

| 類型 | 題數 |
|---|---:|
| `time/calendar` | `52` |
| `money/transaction` | `38` |
| `ratio/fraction` | `28` |
| `percent/compound` | `19` |
| `state tracking` | `15` |
| `age/timeline` | `13` |
| `other` | `7` |
| `average/target total` | `5` |
| `geometry/layout` | `2` |
| `probability` | `1` |

## 哪些類型是真的偏難

因為 `time/calendar` 和 `money/transaction` 在 GSM8K 裡本來就很多，所以只看絕對題數會有偏差。若改看「類型內的失敗率」，更頑固的是：

| 類型 | 題數 | `all-wrong` 比率 | `<=4 hits` 比率 |
|---|---:|---:|---:|
| `percent/compound` | `188` | `7.4%` | `17.6%` |
| `average/target total` | `55` | `7.3%` | `12.7%` |
| `geometry/layout` | `16` | `6.2%` | `12.5%` |
| `time/calendar` | `643` | `5.3%` | `15.4%` |
| `money/transaction` | `492` | `4.3%` | `13.6%` |
| `age/timeline` | `183` | `3.8%` | `9.8%` |
| `ratio/fraction` | `357` | `3.6%` | `10.9%` |
| `state tracking` | `202` | `3.0%` | `14.9%` |

這裡最值得注意的是：

- `percent/compound` 真的偏難
- `average/target total` 也不太好做
- `time/calendar` 雖然是大宗題型，但也確實偏難

## 最硬的錯題模式

從抽樣 16 次仍全錯的題目來看，最常見的 hard pattern 包括：

### 1. 時間 / 日程串接推理

例如：

- `gsm8k_test_1001`
- 做蛋糕、烘烤、冷卻、上霜，反推最晚幾點開始

這類題的難點是：

- 多步驟時間累加
- 正反向推理混合
- 一步算錯就整題錯

### 2. 百分比 / 複利 / 相對變化

例如：

- `gsm8k_test_1016`
- 每年按原價增加 `5%`

以及：

- `gsm8k_test_1048`
- 每年折舊 `21%`

這類題容易出現：

- 百分比基準搞錯
- 用線性變化錯代複利或反之

### 3. 隱藏初始量的反推

例如：

- `gsm8k_test_1190`
- 先用了多少、又補了多少、最後剩多少，反推最初有多少

這類題需要維持正確的 state tracking。

### 4. 目標平均 / 目標總量

例如：

- `gsm8k_test_205`
- 去掉最低分後，最後一科要考幾分才能維持平均 `93`

這類題常見錯法是：

- 平均和總和轉換時漏項
- 條件變更後沒有重建正確總量

## Search 最能救哪些題

看 `greedy 錯，但 pass@16 救回來` 的 `214` 題，最常見類型是：

| 類型 | 題數 |
|---|---:|
| `time/calendar` | `126` |
| `money/transaction` | `88` |
| `ratio/fraction` | `67` |
| `state tracking` | `43` |
| `percent/compound` | `39` |
| `age/timeline` | `37` |
| `other` | `17` |
| `average/target total` | `9` |
| `geometry/layout` | `4` |
| `probability` | `1` |

這說明 search 最能幫的題型，通常是：

- 正確推理路徑其實存在
- 但單次 rollout 很容易在中間某一步走偏
- 換一條 sample path 就有機會做對

也就是說，這些題和「模型根本不會」是不同性質。

## 對 Remask 的意義

### 有希望的區域

最值得 remask / verifier / reranker 去處理的是：

- `1-4 hits`
- `5-11 hits`

因為這些題已經表明：

- 模型有時候能走到正解
- 只是路徑不穩
- 如果能在中途修正錯步，或從多個候選中挑出對的，就很可能有收益

這類題通常比較像：

- 某一步算錯
- 中途狀態追蹤漂掉
- 後段 boxed answer 跟前文不一致

### 比較沒希望的區域

最後剩下的 `50` 題 `all-wrong`，比較可能是：

- 一開始建模就錯
- 關係理解錯
- 全局推理框架就不對

這些題未必適合靠單純 suffix remask 來救，可能需要：

- 更強的 global reasoning
- 更好的 verifier
- 更明確的 scratchpad / equation supervision

## 關鍵結論

這次 `pass@16` 的訊號很明確：

1. **SDAR-1.7B base model 的潛在上限其實很高。**
2. **問題不在於模型完全不會，而是在於單次 sampled rollout 很不穩。**
3. **如果只是直接 sampling，單次 accuracy 反而會比 greedy 差。**
4. **真正有價值的方向不是單純加大隨機性，而是：**
   - verifier
   - reranker
   - self-consistency
   - 有條件的 remask / local repair

因此，如果你接下來想提高 GSM8K 分數，最值得投資的方向不是「更亂地抽」，而是：

- 如何辨識哪條路徑比較可信
- 如何在中途發現錯誤並局部修正
- 如何把這個很高的 latent ceiling 轉成穩定的單次表現
