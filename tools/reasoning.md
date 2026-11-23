<a id="keys"></a>

## 1. `describe` 里说的 key 是什么？

**这里是 SDK 参数，不是注意力的 K。** Mirage Python SDK 的 `describe` 没有显式的单数 `key` 参数；它有一个专门的 **`keywords`** 参数。另外，`options` 是请求字典，里面也有键，例如 `sections`；返回结果同样是字典，`reasoning` 就是一个返回字段名。这三处不能互换。

![Mirage SDK 的 keywords、options 键和返回字段如何传递](../assets/reasoning-model.svg)

本项目所附 SDK 快照的签名：

```python
def describe(
    self, image, *, options=None, language=None, keywords=None,
    width=None, height=None, **settings
):
    ...
```

| 写法 | 放在哪里 | 实际作用／边界 |
|---|---|---|
| `keywords=["helmet", "vest"]` | 顶层显式参数 | 包装层合并成 UTF-8 文本，经独立参数送入原生引擎。不应当作硬筛选条件。 |
| `options={"sections": ["objects"]}` | 请求字典 | 与不同名的 `**settings` 合并，编码成请求 JSON；`sections` 是其中一个键。 |
| `result.get("reasoning")` | 返回字典 | 取模型给出的文本字段；不存在时返回 `None`。 |
| `key="helmet"` 或 `keys=...` | 无显式参数 | Python 会收进 `**settings` 并编码成请求 JSON 的同名键；不等于 `keywords`，没有公开支持的用法。 |

<a id="keywords"></a>

## 2. `keywords` 到底传了什么？

```python
from cactus import Mirage

vision = Mirage(weights="mirage.cact")  # 本机已有兼容运行库和权重
result = vision.describe(
    "photo.jpg",
    options={"sections": ["objects", "text"]},
    conf=0.35,
    language="en",
    keywords=["helmet", "vest"],
)
print(result.keys())  # 看这一版本、这次调用实际返回了哪些字段
```

这次调用在 C ABI 入口处分成不同的参数（JSON 仅为可读格式）：

```text
options  = {"sections":["objects","text"],"conf":0.35}
language = b"en"
keywords = b"helmet\nvest"  # \n 表示实际换行，不是字面反斜杠
```

Python SDK 由 `_options(language, keywords)` 处理后两项：

- 传字符串：保留内容原样，仅做 UTF-8 编码。例如 `"helmet, vest"` 中的逗号不会被 Python 拆成两个词。
- 传多个字符串：用换行连接，再做 UTF-8 编码；不要混入数字等非字符串元素。
- 传 `None`、`""` 或空列表：没有关键词字节，传给原生接口的是 `NULL`。
- 使用显式的 `keywords=...`。写成 `options={"keywords": ...}` 只会进 JSON，不能保证与独立的 C 参数等效。不要同时传同名 `options` 键和 `**settings` 键：例如两处都给 `conf` 会抛出 `ValueError`，不是覆盖。

常用且已有公开示例或说明的选项键：`sections`、`fields` 选输出；`conf`、`text_conf` 控制候选阈值；`max_objects`、`max_text_lines`、`max_tokens` 分别约束数量或长度；`mask_format` 选掩码表示；`form` 定义表单。此处不推定默认值。

**能确认的是传输协议，不是关键词对预测的具体影响。** 公开 wrapper 直接调用 `cactus_describe`；不能据此断言原生引擎一定把词拼进提示、加到 logits、用于筛框或者强制输出。需要只在固定标签中选择时，应使用已公开的 `extract` 表单枚举，而不是用 `keywords` 冒充白名单。调试 `keywords` 是否影响实际输出时，固定图像、权重、SDK、引擎和其他参数后做有／无关键词对照；本次用替身原生接口核验了包装层的参数与字节，不报告模型实测效果。

<a id="reasoning"></a>

## 3. `reasoning` 是返回字段，不是输入开关

根据 Cactus 公开结构，普通描述请求中的 `reasoning` 是 **同一套语义解码器生成的短文本**。解码器读取由粗图特征与候选框区域槽组成的视觉记忆，逐 token 生成语义内容；运行时再把它与检测框、掩码、独立 OCR 等结果一起组装为 JSON。公开材料未给出 trace 的分隔符、确切生成顺序、训练数据或独立损失。它可能漏看或猜错，并不是经验证的内部计算日志。

```python
description = vision.describe("photo.jpg")
trace = description.get("reasoning")
if isinstance(trace, str):
    print(trace)
```

只有解码器内部的**语义 token 序列**适用下式：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>p</mi><mo>(</mo><mi>Y</mi><mo>∣</mo><mi>M</mi><mo>)</mo><mo>=</mo><munderover><mo>∏</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mi>p</mi><mo>(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>∣</mo><msub><mi>y</mi><mrow><mo>&lt;</mo><mi>t</mi></mrow></msub><mo>,</mo><mi>M</mi><mo>)</mo></math>

这里 M 是视觉记忆，Y 是 N 个生成 token 组成的序列，`y_<t` 是第 t 步之前已生成的 token。公式说的是解码器逐步预测下一个 token，**不是**整份 JSON 的概率、不是单独给 `reasoning` 打分，也不能用它推断 `keywords` 在原生引擎中怎样起作用。

读取字典字段不等于启用某个模型模式。没有核实到可依赖的 `reasoning=True` 开关；分区裁剪和本地引擎版本也可能影响返回字段，实际接入先检查 `result.keys()`，不要假设总会返回 `reasoning`。OCR 的识别字符串不回流这套语义解码器；需要核对图片上的精确文字时，优先回看 OCR 的字符串、框和原图。

继续读：[04 API](04-api.md) · [实现与数学](02-implementation.md) · [模型结构](01-architecture.md)。

核对依据：随包 SDK 源码快照的 `cactus/agent/mirage.py`、`cactus/agent/audio.py`（共用参数编码）和 `cactus/agent/runtime.py`（C ABI 绑定）；Cactus Mirage 发布页及其 Structured Visuo-Semantic Perception 说明。
