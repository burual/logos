# 外部预训练词向量作为 GAT 冷启动特征

> 对应任务：把外部预训练中文词向量接进 Logos GAT，替换随机初始化的节点嵌入，
> 看能不能补上「335 节点 / 694 条语义边」的稀疏缺口。
>
> 结论文件。实验代码 `logos_transformer/train_pretrained_init.py`，
> 向量加载 `logos_transformer/embeddings.py`，人工金标 `residual_gold.json`。

---

## 0 结论先行

| # | 任务书预期 | 实测 | 判定 |
|---|---|---|---|
| 1 | 向量覆盖率 > 90% | **98.8%**（331/335，4 个 OOV） | ✅ 达标 |
| 2 | test MRR 0.3609 → > 0.45 | **0.3687**（冻结）/ 0.3668（dense）/ 0.3556（微调） | ❌ 只涨 0.008，离 0.45 很远 |
| 3 | `鱼 is_a 无脊椎动物` 置信度下降 | **0.8327 → 0.4849**（冻结），六组全部下降 | ✅ 达成，且跌破 0.5 |
| 4 | `猫 is_a 人` 下降 | **0.9915 → 0.9999**（冻结）/ 0.9908（微调） | ❌ 反而升高 |
| 5 | 残留错误变少 | 金标口径错误率 **88.1% → 81.1%**（冻结） | ⚠️ 有改善，但只 7 个百分点 |
| 6 | —（任务书未提） | hard-test MRR **0.6938 → 0.7777**（+0.0839） | ✅ 最大收益在这里 |

三条补充事实，比上面任何一条都重要：

1. **基线（随机初始化）的「正确/错误」分离度只有 0.3814**。也就是说，模型给
   错误三元组的分数**系统性地高于**正确三元组（0.8864 vs 0.8641）。它能把这个
   关系的真边排在假边前面（hard MRR 0.69），但**写进库的那批**恰好是最自信的
   错条。这个缺陷在任务书里完全没有提到，是本次做金标才量出来的。
2. **预训练初始化把分离度从 0.3814 拉到 0.7664，翻了一倍** —— 这是本次唯一
   大幅度、可复现的改善。但它没有转化成更低的写入错误率（见 §6）。
3. **微调预训练嵌入有害，且危害与微调步数成正比**：一直冻结 0.3687 / 0.7664；
   等到 epoch 11 才解冻（实际只微调 3 个 epoch）0.3556 / 0.7644，几乎无损；
   第 4 个 epoch 就解冻（微调 10 个 epoch）掉到 0.3370 / 0.4976，接近随机。
   **不是「能不能微调」的问题，是「微调多久」的问题。**

---

## 1 拿到的是什么向量（先说清楚，否则后面的数字会被误读）

任务书给了三个选项：腾讯 6GB、fastText 3.4GB、或者本地 mini。实际跑了方案 A：

| 检查项 | 实测 |
|---|---|
| `huggingface.co` / `pypi.org` 直连 | 502（隧道拒绝） |
| `pypi.tuna.tsinghua.edu.cn` | 200（可用） |
| `ai.tencent.com` 三个向量 URL | 200，但三份 Content-Length 完全相同 = 错误页，文件不在该路径 |
| `modelscope.cn` API | 404 |
| 本机 `jieba` / `gensim` / `transformers` / `sklearn` / `scipy` | 全部未安装 |
| 本机已有向量缓存 | 无 |
| C 盘剩余 | 18 GB（6GB 文件即便下载也偏紧） |

所以本次跑的是**内置 mini 向量**：一张 56 维的动物学属性矩阵（`有脊柱`、`温血`、
`卵生`、`六足`、`哺乳类`……），**由我按教科书手工填写**，329 词 × 56 维，41 KB，
随仓库交付（`logos_transformer/mini_vectors.txt`，可用 `embeddings.py --build-mini`
从 `MINI_TABLE` 重建）。

### 1.1 这个方法学问题必须写在最前面

> **mini 向量是我手写的，不是学出来的。**
>
> 它是一张人类标注的属性表。用它跑出的任何提升，**不能**当作「预训练向量有效」
> 的证据 —— 它最多是**上界探针**：告诉我「如果初始特征恰好包含了正确的语义维，
> 收益的天花板在哪」。
>
> 真实词向量不会有这么干净的 `鱼: {冷血:1, 有脊柱:1, 有鳞:1}`，它是从共现统计里
> 挤出来的稠密向量，`鱼` 和 `鲸` 的余弦相似度可能比 `鱼` 和 `鲤` 还高。

为了让结论有对照，另外实现了两个基线：

- **`charngram`**：字形三元组哈希，**零外部数据**，只反映「词形相近 ⇒ 向量相近」。
  永不 OOV（覆盖率 100%）。
- **`random`**：完全复现原基线（同 seed、同 split、同负样本）。

判据是：如果 `charngram` 的提升接近 `mini`，说明收益来自「不再让 335 个节点共享
一个随机向量」，与语义无关；如果只有 `mini` 涨，才说明语义内容起了作用。
实测两者差距极大（分离度 0.4151 vs 0.7664），见 §5。

---

## 2 覆盖率与 OOV

```
义项覆盖      : 331/335 = 98.8%
  裸词          :  331
  零向量(OOV)    :    4
  OOV 样例    : 游动/v, 筑造/v, 认知/v, 声呐器官/n
特征矩阵      : (335, 56)  非零行 331/335
```

四个 OOV 都是 mini 表里没收录的动词/复合词。OOV 一律给零向量（`--oov-mode zero`），
没有崩、没有随机填。可选的 `--oov-mode charngram` 会给它们字形哈希向量，
`embeddings.py --neighbour-fill N` 可以用 N 跳邻居均值初始化。

**零向量在 GAT 里不是死路**：`RelationAwareGATLayer` 的注意力是 softmax 归一化的
加权和，只要邻居非零，第一层之后该节点就有非零表示。只是它拿不到语义先验，等价于
退回随机初始化的效果。

---

## 3 两种架构与「写回兼容」

新 checkpoint 必须能被 `infer_and_write.py` 直接加载，否则第 3 条验证做不了。而写回
管线是按 `gat.node_embedding.weight` 的**形状**反推架构的（`inspect_checkpoint`），
所以：

| 架构 | 节点表示 | 参数字典 | 写回可用 |
|---|---|---|---|
| `--arch table`（默认） | 保持 `nn.Embedding(N, hidden)`，只把 `weight` 的初值换成 `proj(词向量)` | 与基线**逐键一致** | ✅ 直接加载 |
| `--arch dense` | `input_proj = Linear(D, hidden)` + ELU，节点表示由稠密向量算出 | 多 `gat.input_proj.*`，少 `gat.node_embedding.weight` | 需折叠 |

dense 训练结束后自动折叠一份 `closure_gat_relaware_best_folded.pt`：

```python
E = ELU(input_proj(input_features))          # eval 模式 dropout 关闭
state["gat.node_embedding.weight"] = E
```

这是**精确等价**而不是近似：`LogosGAT.forward` 的第一步就是
`F.dropout(self.node_embedding.weight, p, training=self.training)`，eval 时 dropout
是恒等映射。

参数量差异：table 261,262 / dense 252,366（少 8,896 = 335×32 − 56×32 − 32）。

---

## 4 一个必须对齐的工程细节：初始化尺度

`LogosGAT.__init__` 用 `normal_(weight, std=0.1)`，每行范数 ≈ `0.1×√32 = 0.5657`。
而 mini 向量一行有 6~8 个非零维（值 1），范数约 2.4 —— 是随机初始化的 **4 倍**。

不管它，`h_s^T W_r h_t` 一上来就接近饱和，sigmoid 全部挤到 0/1 两端，训练不动。
所以 `project_features` 在投影之后统一重标：

```python
rescale_rows(out, target=0.1 * math.sqrt(hidden))
```

首轮实验确实踩了这个坑：dense 分支最初直接把原始 56 维特征喂给 `Linear`，实测
`h0` 行范数 mean=0.874 / max=2.15（应为 0.5657），test MRR 只有 0.3402、分离度
0.6214。修成先 `rescale_rows(features, 0.1*sqrt(hidden))` 再正交投影之后，同一
配置变成 0.3668 / 0.5341、OOD 0.9861。**这部分差异是尺度效应，不是架构效应。**

投影方式默认 `orth`（固定种子的随机正交投影，保范数），另有 `pca`（确定性、无随机）
和 `slice`（截断/补零，用来排除「投影本身有用」）。

---

## 5 六配置消融

同一 seed 42、同一 source-disjoint split、同一负样本采样、同一套指标函数
（全部从 `train_closure_relaware` 直接 import，未重写）。

| 配置 | 覆盖率 | test MRR | test hits@10 | hard-test MRR | hard margin | OOD 拒绝率 | 参数量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 基线（随机初始化） | 0.0% | 0.3609 | 0.702 | 0.6938 | +0.1623 | 0.8611 | 261,262 |
| mini · table · 微调 | 98.8% | 0.3556 | 0.702 | 0.7370 | +0.1861 | 0.8611 | 261,262 |
| **mini · table · 冻结** | 98.8% | **0.3687** | 0.709 | **0.7777** | +0.1535 | 0.9167 | 261,262 |
| mini · table · 只冻 3 epoch | 98.8% | 0.3370 | 0.660 | 0.7242 | +0.1669 | 0.8264 | 261,262 |
| mini · dense · 微调 | 98.8% | 0.3668 | **0.745** | 0.7411 | +0.1336 | **0.9861** | 252,366 |
| 字形哈希 · table · 微调 | 100.0% | 0.3025 | 0.631 | 0.6649 | +0.1043 | 0.8403 | 261,262 |

读法：

- **`--init random` 逐位复现基线**（0.3609 / 0.6938 / 写回 118 行 / best epoch 13），
  同口径成立。这条是前提，不是成绩 —— 复现不了的话后面所有对比都无效。
- **冻结优于微调，但差距不大**：test MRR +0.0131、hard MRR +0.0407、OOD +0.0556，
  唯一反例是 hard margin（+0.1535 vs +0.1861）。原因很实在 —— 微调组的嵌入实际
  只在 epoch 11~13 被更新了 3 步（`best_epoch=13`），等于「几乎没微调」。所以这两行
  的真实含义是「不微调 ≈ 微调 3 步」，真正拉开差距的是下一行的「微调 10 步」。
- **只冻 3 epoch 是最差的一组**。解冻太早，前 10 个 epoch 里 GAT 还冻着、只有嵌入在
  被 830 条正样本拽着走，语义先验被少量数据破坏。这条比「冻结 vs 微调」更值得记：
  **不是「微调有害」，是「微调步数有害」**，3 步就够毁掉先验。
- **dense 的 OOD 最好（0.9861）**，hits@10 也最高（0.745），但 hard MRR 与分离度
  都不如 table-冻结。它的输入经过一层可训练变换，等于多了一道「压缩」，对
  cross-POS 这类明显违反词性的负样本更敏感。
- **charngram 全面低于基线**：test MRR 0.3025、hard MRR 0.6649。字形哈希提供的是
  「词长得像」而不是「意思像」，反而引入了噪声。

---

## 6 金标：58 个字的结论是「预训练不认识对错，只认识像不像」

### 6.1 金标怎么来的

`residual_gold.json` 给 `inferred_relations` 的**全部 118 行**逐条打了判定，
四分类：

| 判定 | 条数 | 含义 |
|---|---:|---|
| `correct` | 13 | 三元组成立 |
| `wrong` | 67 | 事实或类型错误 |
| `wrong_relation` | 29 | src 与 tgt 之间确有语义关系，但关系名用错（`狗 typically_has 有脊柱` → 应 `has_property`） |
| `uncertain` | 9 | 母语直觉也有争议 |

另加 12 条 `relations` 表里的既有断言作对照组（`control: true`）。

判定规则写在 `make_residual_gold.py` 的 `RULES` 与每条 `reason` 里，可审计、可增补。
关键一条：

> `can_do(A,B)` 在库里混了两种读法 —— 动作类→子动作（`运动 can_do 攀爬`）与
> 施事→动作（`猫 can_do 捕猎`）。判 `correct` 当且仅当 (a) A 是施事且确实能执行 B，
> 或 (b) A 是动作类且 B 属于该类。A 是动作类而 B 属于另一个动作类
> （`防御 can_do 潜水`：潜水∈运动，∉防御）判 `wrong`。

**注意口径**：写回是按置信度 top-3 挑的，天然是高分条，所以「正确 vs 错误」只能在
**写回组内部**比较。把对照组（模型没见过的真边，均分 0.54）混进来算会得到
0.28 这种毫无意义的数 —— 首轮就是这么算错的，已修。

### 6.2 分离度（同一组内，correct 的分数 > 错误条分数的配对比）

| 配置 | correct 均分 | 错误均分 | **分离度** |
|---|---:|---:|---:|
| 基线（随机初始化） | 0.8641 | 0.8864 | **0.3814** |
| mini · table · 微调 | 0.9035 | 0.7716 | **0.7644** |
| mini · table · 冻结 | 0.8862 | 0.7475 | **0.7664** |
| mini · table · 只冻 3 epoch | 0.7963 | 0.7619 | 0.4976 |
| mini · dense · 微调 | 0.8235 | 0.7575 | 0.5341 |
| 字形哈希 · table · 微调 | 0.7671 | 0.7919 | 0.4151 |

**基线的 0.3814 是本轮最刺眼的数字**：模型给错条的分数比对的还高。这不是「训练不足」，
是「目标错位」—— 训练目标是同一个关系内把真边排在假边前面（hard MRR 0.69 做到了），
但没有任何一项损失在区分「这个三元组整体该不该存在」。

mini 初始化把 0.3814 推到 0.7664。**但 charngram 只到 0.4151** —— 这说明收益来自
语义内容，不是「不再共享随机向量」。

### 6.3 阈值过滤：看起来聪明，实际只是整体压低

拿 118 条当固定探针集，对每个 checkpoint 重算 prob，统计有多少条仍越过 0.8：

| 配置 | 仍 >0.8 | correct 保留 | wrong 保留 | 保留集合错误率 |
|---|---:|---:|---:|---:|
| 基线（随机初始化） | 118/118 | 13/13 (100%) | 96/96 (100%) | 88.1% |
| mini · table · 微调 | 64/118 | 10/13 (77%) | 53/96 (55%) | 84.1% |
| **mini · table · 冻结** | 54/118 | 10/13 (77%) | 43/96 (45%) | **81.1%** |
| mini · table · 只冻 3 epoch | 59/118 | 3/13 (23%) | 52/96 (54%) | 94.5% |
| mini · dense · 微调 | 70/118 | 7/13 (54%) | 57/96 (59%) | 89.1% |
| 字形哈希 · table · 微调 | 59/118 | 3/13 (23%) | 52/96 (54%) | 94.5% |

冻结组看起来「聪明」：过滤掉 55% 的错误，只丢掉 23% 的正确。但**净效果只有
88.1% → 81.1%**。原因是底数悬殊 —— 错误条 96 条、正确条 13 条，即使错条砍掉一半，
剩下 43 vs 10 的比例仍然是 81%。

要真正降错误率，correct 保留率必须接近 100% 而 wrong 接近 0%。77% vs 45% 的差距
远远不够。**这正是软特征的天花板**：嵌入相似度不知道 `蛇` 是变温动物，只知道
`蛇` 和 `爬行纲` 像。

反过来看 `只冻 3 epoch` 与 `charngram`：correct 保留率只有 23%，把本来对的也砍了，
错误率反而升到 94.5%。**没有正确的先验时，过滤只会随机杀伤。**

---

## 7 写回对比

同一命令、同一组符号守卫、同一阈值 0.8、同一 top-3 规则，只换 checkpoint：

```bash
python logos_transformer/infer_and_write.py \
  --checkpoint <CKPT> --dry-run \
  --hierarchical-guard --source-pos-guard --leaf-target-guard --top-k 0 --seed 42
```

| 配置 | 阈值以上（不加守卫） | 阈值以上（加守卫） | top-3 后 | **实际写入** |
|---|---:|---:|---:|---:|
| 基线（随机初始化） | 267 | 211 | 154 | **118** |
| mini · table · 微调 | 303 | 211 | 217 | 166 |
| mini · table · 冻结 | 418 | 291 | 240 | 183 |
| mini · table · 只冻 3 epoch | 264 | 118 | 183 | 95 |
| mini · dense · 微调 | 167 | 147 | 116 | 96 |
| 字形哈希 · table · 微调 | 169 | 126 | 137 | 98 |

基线那行 154 → 118 与上一轮 `GUARDS_EVALUATION.md` 的记录**逐位一致**，说明对照成立。

**写入行数不是质量指标**。冻结组写 183 行（比基线多 55%），但它的保留集合错误率是
最低的 81.1%；只冻 3 epoch 写 95 行（比基线少 20%），错误率却最高（94.5%）。
上一轮把「写入量 154 → 50~80」当验收准则，从这次的结果看这个指标本身是错的 ——
一个什么都不写的模型，写入量指标完美。

高置信度错条在两个极端都出现：

```
基线                 冻结
猫/n is_a 人/n  0.9915   猫/n is_a 人/n  0.9999
鱼/n is_a 人/n  0.9886   鱼/n is_a 人/n  0.9989
```

`鱼 is_a 人` 在冻结组里升到 0.9989 —— 比基线更自信。原因见 §8。

---

## 8 为什么 `猫 is_a 人` 不降反升

任务书的假设是：

> `猫 is_a 人` should drop (humans and cats are semantically distant in pre-trained
> space despite both being mammals)

**这个假设在两个层面都不成立。**

1. 在真实 word2vec/fastText 空间里，`猫` 和 `人` 都是高频名词，余弦相似度通常在
   0.4~0.5，**不低**。它们共享大量上下文（宠物、家庭、情感、动词搭配）。
2. 在我手写的 mini 向量里，`猫` 与 `人` 共享 6 个维（有毛/温血/胎生/有脊柱/陆生/
   哺乳类），余弦相似度**更高**。冻结组给到 0.9999 是这套特征的直接后果。

更根本的是：`猫 is_a 人` 是**分类学层级**错误，它需要的是「猫科与人科是两个互斥
分支」这条公理，不是「两个词像不像」。嵌入相似度恰好在这一类问题上**逆向指路**：
越像的两个概念，越容易被判成上位关系。

同理 `鱼 is_a 无脊椎动物` 能降下来（0.8327 → 0.4849），不是因为它懂了「鱼有脊柱」，
而是因为 mini 表里 `鱼` 与 `无脊椎动物` 的属性向量差异大（有脊柱 vs 无脊柱），
点积自然变小。**如果换成真实词向量，这个降幅未必保得住** —— 真实空间里
`鱼` 与 `无脊椎动物` 的距离没有手写属性表这么干净。

一句话：**这次降下去的都是「属性差异大」的错条，降不下去的都是「层级错」的错条。**
层级错需要的是公理，不是相似度。

---

## 9 为什么 test MRR 到不了 0.45

test MRR 0.3609 由 `is_a` 主导（test 141 条正样本里它占大头）。`is_a` 的瓶颈是
**缺互斥公理**（上一轮 `GUARDS_EVALUATION.md` §6 已经量过：`鱼/青蛙/鲤科 is_a
无脊椎动物`、`猫 is_a 人` 都是「祖先集是真子集、特异性严格递增」，与真边在结构上
不可区分）。

嵌入能提供的是「两个概念有多像」，而 `is_a` 需要的是「A 的全部外延是否被 B 包含」。
这是集合关系，不是相似度。所以：

- hard MRR（同关系内排序）能涨 —— 相似度确实能改善「哪个候选更像」；
- filtered MRR（跨全库排位）涨不动 —— 真正的上位候选未必是向量最近的那个。

预期 0.3609 → 0.45 的 +0.09 需要的是**互斥公理 + 层级约束**，不是更好的初值。
把嵌入接进来只买到 +0.008。

---

## 10 复现

```bash
# 0) 重建 mini 向量并看覆盖率（不需要联网、不需要任何模型权重）
python logos_transformer/embeddings.py --build-mini --report
#  → 义项覆盖 331/335 = 98.8%

# 1) 同口径基线（必须先跑这条确认 0.3609 / 0.6938）
python logos_transformer/train_pretrained_init.py --init random --epochs 200 \
  --output-dir logos_transformer/checkpoints_pretrained/random

# 2) 六组消融（每组 30~40s，CPU 单核）
GOLD=residual_gold.json
PT='鱼/n|is_a|无脊椎动物/n,猫/n|is_a|人/n,猫/n|is_a|猫科/n,蛇/n|has_property|温血/a'
for spec in "mini_tuned:--init mini" \
            "mini_frozen:--init mini --freeze-embed" \
            "mini_early3:--init mini --freeze-epochs 3" \
            "mini_dense:--init mini --arch dense" \
            "charngram:--init charngram"; do
  name=${spec%%:*}; flags=${spec#*:}
  python logos_transformer/train_pretrained_init.py $flags --epochs 200 \
    --gold $GOLD --probe-triples "$PT" --compare-mode \
    --output-dir logos_transformer/checkpoints_pretrained/$name
done

# 3) 汇总表（文档里的三张表就是这两条命令的输出）
python logos_transformer/ablation_report.py
python logos_transformer/gold_probe.py

# 4) 写回对比：只换 checkpoint，别的参数一个都不动
python logos_transformer/infer_and_write.py \
  --checkpoint logos_transformer/checkpoints_pretrained/mini_frozen/closure_gat_relaware_best.pt \
  --dry-run --hierarchical-guard --source-pos-guard --leaf-target-guard --top-k 0 --seed 42
```

---

## 11 接真实向量只差一个路径

`embeddings.py` 的加载器已经为 6GB 文件准备好了：先 `keep_set_for()` 把 335 个词
及其子串前缀/后缀收成过滤集，再**流式**扫一遍，只解析需要的行，峰值内存
O(|keep|×dim)。不是先读进内存再筛选。

```bash
# 腾讯 AI Lab 200d（6GB）
python logos_transformer/train_pretrained_init.py \
  --init file \
  --vectors /path/to/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt \
  --embedding-dim 200 --epochs 200 \
  --gold residual_gold.json --compare-mode

# fastText 300d（3.4GB，bz2 直接读，不用先解压）
python logos_transformer/train_pretrained_init.py \
  --init file --vectors /path/to/sgns.merge.word.bz2 --embedding-dim 300
```

**接线已验证**：把 `mini_vectors.txt` 当外部文件喂给 `--init file --vectors`，
结果与 `--init mini` **逐位相同**（test MRR 0.3556 / hard MRR 0.7370 / margin
+0.1861）。说明流式过滤、维度推断、覆盖率统计、行序对齐这几段都是通的，换成真文件
只需要改路径。

`--init file` 的自动行为：

- `dim` 缺省从文件首行推断；显式给错维度会**报错**而不是静默截断；
- `word_pos`（如 `猫_n`）优先，回退到裸词，再回退到复合词拆分（`鲸类头部` →
  `鲸类` + `头部`），最后 `zero` 或 `charngram`；
- 行序按 `graph.node_ids` 重排并打印是否发生过重排（`build_node_features` 按
  `sense_id` 升序，两者未必一致）。

跑完真文件之后要重看的是 §8：如果 `鱼 is_a 无脊椎动物` 的降幅消失、`猫 is_a 人`
仍然接近 1，那就证实了「这次收益来自手写属性表的干净性，不是来自预训练」。
**这两条是判断 mini 结论能不能外推的关键实验。**

---

## 12 口径与坑

1. **`--init random` 必须逐位复现基线**。为此训练脚本复制了 `train_closure_relaware`
   main 的训练段（含 `init_attention` / `init_loss` 这两次前向 —— 它们在 `train()`
   模式下会消耗 dropout 随机数，删掉就复现不了）。数据切分、负样本、全部指标函数
   则是**直接 import**，一行都没有重写。

2. **写回集合内部比较，不要跨组**。写回条是按置信度 top-3 挑的，天然高分；对照组
   （未见的真边，均分 0.54）混进来会把分离度算成 0.28。首轮就是这么错的。

3. **`--freeze-gat-epochs` 会顺带冻住 `node_embedding`**。因为它就在 `model.gat` 里。
   所以 `--freeze-epochs 10`（默认 GAT 也冻 10）与不冻**完全一样** —— 首轮跑出两组
   逐位相同的结果才发现。现在 `--freeze-epochs` 的赋值写在 GAT 赋值**之后**并且只在
   显式给出时才覆盖。

4. **`charngram` 必须强制 `--oov-mode charngram`**。它的加载器是空的（没有任何真词），
   如果 OOV 回退还是 `zero`，每个义项都拿到零向量，指标掉到随机水平（test MRR
   0.0128）而**不会报错**。这是静默失败。

5. **dense 的输入要先重标范数**。见 §4，不改的话测到的是尺度效应不是架构效应。

6. **`total_changes` 不可信（上一轮遗留）**。`senses` 上有 FTS 同步触发器
   `tr_senses_fts_au`，单行 UPDATE 会让 `conn.total_changes` 虚报 9~14 倍。

7. **`typically_has` 的目标词性分布被闭包污染**。`load_relation_target_pos` 从
   `closure` 统计，`relations` 表里 `typically_has` 的目标是 146 n / 3 a（a 占 2.0%，
   低于 5% 门槛），但 `closure` 里是 208 n / 18 a（8.0%），于是 a 被放行 —— 这就是
   118 条里 31 条 `typically_has → 形容词` 的来源。**这是上一轮的设计缺陷**，本轮
   没有改（改了会让既有的 118 行失效），但记录在此：目标词性应当从 `relations`
   的 asserted 边统计，而不是从 `closure`。

---

## 13 结论

预训练初始化是**有用的**，但用处和任务书设想的不同：

- 它改善的是**同关系内的排序质量**（hard MRR +0.0839）和**正确/错误分离度**
  （0.3814 → 0.7664），这两项都是实打实的。
- 它**不改善**跨库排位的 filtered MRR（+0.008），因为那需要互斥公理而不是相似度。
- 它**不改善**写回的绝对质量（88.1% → 81.1%），因为过滤是近似均匀的。
- **冻结 ≥ 微调 3 步 ≫ 微调 10 步**（0.3687 / 0.3556 / 0.3370）。用少量数据去拽一个
  本来正确的先验，拽得越久越糟 —— 这条比「冻结 vs 微调」的二分法更有用。

要把 118 条里的错误真正压下去，缺的仍然不是更好的初值，而是三样东西：
**互斥公理**（`鱼` 与 `无脊椎动物` 不能同时成立）、**层级约束的传递性检查**
（`猫 is_a 人` 需要知道人科与猫科互斥）、以及**一个真正在区分「该不该存在」的
损失项**（现在的 BCE 只区分「比另一个候选好」）。

上一轮 `GUARDS_EVALUATION.md` §6 已经说过「假阳靠校准与符号守卫都解决不了」。
这一轮的金标给了那个判断一个具体数字：**基线分离度 0.3814**。预训练把它推到 0.7664，
是迄今最大的一次改善 —— 但 88.1% → 81.1% 的错误率说明，**问题不在特征质量，在目标函数**。
