<a id="reasoning"></a>

## 1. Mirage 的 reasoning 是什么，怎么生成？

`reasoning` 是 **同一个 8 层语义解码器生成的一段短文本**。它和场景、物体名称、属性与关系等语义信息共用视觉条件；公开计算图没有显示一个额外的 Reason 网络、工具搜索或多轮推理模块。它是一次模型预测，不是把注意力、隐藏状态或经过验证的内部思考日志导出来。

![视觉 Keys 和 reasoning 的模型结构](../assets/reasoning-model.svg)

1. **准备视觉条件**：共享编码器输出粗图 E（16×16×512）和细图 D（64×64×128）。检测头从 D 取得候选区域 B。把 E 的图像格与已构造的区域槽放进视觉记忆 M。区域槽也是 512 维，表示该框的中心细特征、坐标编码和框内粗特征的组合。
2. **解码器逐步读图**：当前位置的文本状态通过交叉注意力读取 M。当前位置的文本状态随已生成的前缀变动，视觉记忆不随当前 token 重算。
3. **语义文字 → JSON**：语义解码器逐 token 生成模型所用的紧凑语义表示。运行时将语义结果与检测框、掩码和独立 OCR 字符串组合。公开 API 可返回 `reasoning` 文本字段；坐标和掩码不是从这段文字反解析出来的。

对生成整条语义序列 Y，可以写成：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>p</mi><mo>(</mo><mi>Y</mi><mo>∣</mo><mi>M</mi><mo>)</mo><mo>=</mo><munderover><mo>∏</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mi>p</mi><mo>(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>∣</mo><msub><mi>y</mi><mrow><mo>&lt;</mo><mi>t</mi></mrow></msub><mo>,</mo><mi>M</mi><mo>)</mo></math>

这是序列生成公式，不是整份返回 JSON 的统一概率。官方公开资料未给出 `reasoning` 的原生段落分隔符、精确生成顺序、训练样本或独立损失，也不能从 Python wrapper 推断出它们。

<a id="keys"></a>

## 2. Keys 是什么？它和 Values 有什么不同？

这里是**解码器交叉注意力的视觉 Keys**。`key` 指给一条视觉记忆生成的匹配向量，不是加密密钥，也不是 JSON 里的字段名。以一个逻辑注意力头为例：

| 对象 | 从哪里得来 | 每头／组向量宽度 | 用途 |
|---|---|---:|---|
| Query（Q） | 当前文本状态 xₜ | 48 | 这一步找什么 |
| Key（K） | 视觉记忆第 j 行 Mⱼ（512 维）做 K 投影、归一化 | 48 | 用于与 Q 匹配 |
| Value（V） | 同一行 Mⱼ 做另一套 V 投影 | 64 | 读出的内容，被匹配权重加权 |

同一个 Mⱼ 给 K 和 V 两条**不同**的投影，不是先算 K 再由 K 生成 V。同一请求内，固定图片的视觉 K/V 可按解码层预计算或复用；每个生成步重算的是随文本状态改变的 Q。不同解码层使用本层的投影；并不等于把每层 K/V 都混成一个，也不能据此推出原生缓存的实际内存布局。

设视觉记忆的行数为 L。已公开的权重清单表明，**每层视觉交叉注意力各有 8 组 K/V**：K 投影权重为 512×384（384＝8×48），V 投影权重为 512×512（512＝8×64）。对 L×512 的 M 逐行投影并拆成分组，逻辑 Keys 形状是 **L×8×48**，Values 是 **L×8×64**；上图只画其中一组。L 不要写死成 320：图像 E 的逻辑网格为 256 格，区域槽数量和 padding 有效性另计。

别把三套 Keys 混在一起：此处是解码器的**视觉交叉注意力**；同一解码器的**文本自注意力**从文本状态生成 K/V，每层只有 2 个 KV 头，还要遵守因果约束；编码器的**空间注意力**从视觉网格与寄存器取 K/V，Q/K 每头宽 32。它们读的对象、投影矩阵和分组数不同。

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>α</mi><mi>j</mi></msub><mo>=</mo><msub><mi mathvariant="normal">softmax</mi><mi>j</mi></msub><mrow><mo>(</mo><mfrac><mrow><msubsup><mi>q</mi><mi>t</mi><mi mathvariant="normal">T</mi></msubsup><msub><mi>k</mi><mi>j</mi></msub></mrow><msqrt><mn>48</mn></msqrt></mfrac><mo>+</mo><msub><mi>Mask</mi><mi>j</mi></msub><mo>)</mo></mrow><mo>;</mo><msub><mi>o</mi><mi>t</mi></msub><mo>=</mo><munder><mo>∑</mo><mi>j</mi></munder><msub><mi>α</mi><mi>j</mi></msub><msub><mi>v</mi><mi>j</mi></msub></math>

**手算例（教学，不是官方权重）**：两个区域的缩放后匹配分数是 2 和 0，softmax 约为 0.881、0.119。如果对应 Values 简化成 [3, 0] 和 [0, 4]，则读出约为 [2.643, 0.476]。Q 和 K 决定读哪一行更重；真正被混合的是 V。这个权重不等于“识别正确的概率”。

<a id="reasoning-ocr"></a>

## 3. reasoning、OCR 和表单为什么会不同？

| 输出场景 | 哪一路产生 | 不代表什么 |
|---|---|---|
| `reasoning` | 共享语义解码器生成短文本 | 不是 OCR 的复述，不是内部计算日志 |
| `text` 内的识别字符串 | 独立 OCR 从原图文字框裁图读取 | 不回流语义解码器 |
| `form` | 同模型的独立约束请求模式；枚举按各候选完整 token 路径得到分数 | 不是把 `reasoning` 再交给另一模型总结；该模式不运行 OCR reader |

所以同一张图可以出现 `text` 正确读到设备编号，但 `reasoning` 没提或写错。遇到需要精确原文的问题，分别检查 OCR 字符串、框与原图。如果要把 JSON 交给另一个模型，建议先只给结构化字段，再单独比较加上 trace 的收益。官方用 Qwen3-0.6B 在 128 张图的 256 道 VQAv2 题做过这一消融：不传 trace 33.1，传完整 trace 31.3（每百得分；配对差 −1.8，95% 区间 −5.0 至 1.2），没有观察到可靠增益。这是特定小样本，不是所有任务上 trace 都有害。

## 4. 用 API 看这个字段

在已经安装兼容 SDK、持有本地权重和运行库的环境中，先用普通描述请求检查它是否存在。下例只读取 JSON 字段，不声称能控制模型内部的“思考预算”。

```python
from cactus import Mirage

vision = Mirage(weights="mirage.cact")
result = vision.describe("photo.jpg")
trace = result.get("reasoning")
if isinstance(trace, str):
    print(trace)
```

这段代码的 `"reasoning"` 是 **JSON 字段名（字典的 key）**，和上面的注意力 Keys 没有计算关系。不要把 `describe(..., reasoning=True)` 当成已公开验证的开关。这里只展示接口读法，未在本次站点更新中以官方权重复跑结果。

公开依据：Cactus《Mirage》发布页的 What comes back 和 Maths；《Structured Visuo-Semantic Perception》的计算图、区域槽、Compact semantics and bounded decisions 及 VQAv2 对照；Python SDK 快照里的 `Mirage.describe` 调用边界。

继续读：[模型结构](01-architecture.md) · [实现与数学](02-implementation.md) · [API](04-api.md)。
