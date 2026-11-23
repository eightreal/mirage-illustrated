从[安全帽完整数据流](#data-flow)入手，再看[模型如何生成 Reasoning](#reasoning)、[哪些字段是语义输出](#semantic-output)，或直接跳到[SDK Describe 的 key](#keys)及 [keywords 的传递方式](#keywords)。

<a id="data-flow"></a>

## 先用同一张安全帽照片走完整条路径

**教学例，不是 Mirage 实测结果**：照片里有一顶黄色安全帽，帽壳贴纸印着 `A13`。下面始终用这一张图，区分各张量和最后返回的字段。

![Mirage 模型结构：Stem 旁路、Body 与带 4 个 register 的 Attention、E／D 多尺度特征，以及语义、检测、分割、OCR 输出](../assets/mirage-model-architecture.svg)

*图按网络模块和张量分区，不按调用顺序排成一列。点图可查看完整尺寸；下文继续用同一张安全帽照片沿箭头走一遍。*

1. **原图 → 卷积 Stem。** 等比例缩放并补到 512×512。五级卷积让网格从 512² 依次降到 256²、128²、64²、32²、16²，通道增加到 512。末级 F5（16×16×512）走主干，S4/S8/S16 是保留给细节分支的中间特征。这时没有“安全帽”这个名称。
2. **F5 → Body → Attention → 粗图 E。** 单个 Body 保持 16×16×512：先在各通道混合相邻网格，再在同一位置混合通道，两段分别加回各自输入。16 块串联得到的还是主干中间张量，不是 E；展平加寄存器、经过 Attention 和按层融合，才得到 16×16×512 的 E。
3. **E + Stem 旁路 → FPN → 细图 D。** 先将 E 投影到 128 通道、上采样至 32²，与投影后的 S16 相加，再过 1 个局部残差块；然后上采样到 64²，与投影后的 S8 相加，过 3 个局部残差块，再做 RMSNorm 和 SiLU 得到 D（64×64×128）。D 可以给出更细的位置特征，不是物体列表，也不是另一套编码器。
4. **D → 检测候选 B。** 检测头从 D 找物体框候选（这里对应帽子）和文字框候选（对应贴纸），给各自的位置、大小和检测分数。它还没有证明物体名字或贴纸字符。
5. **E、D、B → 视觉记忆 M → 解码器。** 以帽子框为例，从 D 取中心特征、从 B 编码坐标、从 E 平均池化框内有效格，合成 512 维区域槽。全图 E 网格与区域槽共同组成 M；8 层语义解码器读取 M 和此前 token，逐步写出 `helmet`、`yellow` 等语义及短 `reasoning`。候选框的数值不是解码器一个字符个字符写出来的。
6. **空间和文字支路单独算 → 最后组装 JSON。** 检测给 `bbox`／`score`；分割利用 D、S4 与对象框算 `mask`；OCR 用文字框从**原照片**裁图读取 `A13`，之后不把识别字符串喂回语义解码器。运行时才按区域引用把输出组装到一起。

一条简化的结果可以是这样：对象名和颜色来自语义，框来自检测，`text[0].string` 来自 OCR。下面坐标和文字都是为了说明字段来源而自编的示意。

```json
{
  "objects": [{
    "name": "helmet",
    "description": {"color": "yellow"},
    "bbox": [120, 200, 360, 510]
  }],
  "text": [{"string": "A13"}],
  "reasoning": "A yellow helmet is visible."
}
```

[继续看 Reasoning 的实现](#reasoning) · [按字段核对语义输出](#semantic-output) · [查 SDK Describe 的 key](#keys)。


<a id="reasoning"></a>

## Reasoning：模型怎么写出这段解释？

**Mirage 的 `reasoning` 是同一个 8 层语义解码器根据视觉条件生成的短文本。** 它和场景描述、物体名称、属性及关系共用解码器。检测框、掩码和 OCR 另有计算分支，运行时最后才组合 JSON。没有证据表明这里另接了一个推理网络，或返回了可验证的内部计算日志。

![Reasoning 模型结构：视觉记忆到语义解码，空间与 OCR 分支旁路](../assets/reasoning-architecture.svg)

<a id="semantic-output"></a>

### 图中深橙色：哪些是语义输出？

**语义输出是解码器生成的场景、物体名称和属性、关系、事件及 `reasoning` 等内容。** 框、检测分数、掩码和 OCR 字符串来自别的分支。运行时会把两个来源的字段写进同一个 JSON；不能仅看字段在 `objects` 里面，就把整个对象当成解码器输出。

| 返回字段示例 | 来源 |
|---|---|
| `scene.setting`、`objects[].name`、`objects[].description`、`objects[].position`、`relations[].predicate`、`events`、`reasoning` | **语义输出**：模型对场景、实体、属性、关系／事件的描述和短解释。 |
| `objects[].bbox`、`objects[].score`；若请求返回则也有 `objects[].mask` | **非语义解码输出**：框和分数来自检测，掩码来自实例分割，运行时按区域引用挂回对象。检测分数不是名字正确的概率。 |
| `text[].string`、`text[].bbox`、`text[].score` | **非语义解码输出**：字符串由 OCR reader 读取；框和分数来自文字检测。OCR 字符串虽然也是文本，但不是语义解码器生成的。 |

`id` 和关系里的实体引用用于运行时关联记录，不是检测框索引，也不能据此判断语义解码器是否直接生成了这个整数。

下面同一条对象记录里，`name` 和 `description` 是语义，`bbox` 和 `score` 不是。示例是自编截取，非真实运行记录或完整返回结构。

```json
{
  "objects": [{
    "name": "helmet",
    "description": {"color": "yellow"},
    "bbox": [120, 200, 360, 510],
    "score": 0.72
  }],
  "text": [{"string": "A13"}],
  "reasoning": "A yellow helmet is visible."
}
```

### 1. 从 E、D、B 造视觉记忆 M

| 张量或数据 | 由谁产生 | 如何进入语义分支 |
|---|---|---|
| E：16×16×512 | 共享编码器的粗图输出 | 256 个空间格，加上经门控缩放的二维位置编码，按有效区域掩码读取。 |
| D：64×64×128 | FPN 的细图输出 | 检测头从这里产生 B；区域槽在这里取框中心特征。 |
| B：候选框坐标 | 检测头 | 提供区域位置和引用基础，不负责给物体命名或写理由。 |
| 区域槽：每槽 512 维 | 在 E、D 上按 B 取样并投影 | 与 E 的图像格组成视觉记忆 M。 |

取第 k 个框：从 D 的框中心取 fₖ（128 维）；框的 4 个归一化坐标各用频率 2ʳπ（r＝0…7）做 8 个正弦和 8 个余弦编码，得 φ(bₖ)（64 维）；从 E 平均池化框内有效粗图格（包括框中心格）得 ēₖ（512 维）。将前两者**拼接**成 192 维后投影到 512，ēₖ 独立做 512→512 投影；这两个 512 维结果**逐元素相加**。

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>z</mi><mi>k</mi></msub><mo>=</mo><msub><mi>W</mi><mi>s</mi></msub><mo>[</mo><msub><mi>f</mi><mi>k</mi></msub><mo>;</mo><mi>φ</mi><mo>(</mo><msub><mi>b</mi><mi>k</mi></msub><mo>)</mo><mo>]</mo><mo>+</mo><msub><mi>W</mi><mi>p</mi></msub><msub><mover><mi>e</mi><mo>¯</mo></mover><mi>k</mi></msub></math>

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>m</mi><mi>k</mi></msub><mo>=</mo><mi mathvariant="normal">RMS</mi><mo>(</mo><msub><mi>z</mi><mi>k</mi></msub><mo>)</mo><mo>+</mo><msqrt><mn>512</mn></msqrt><msub><mi>e</mi><msub><mi>r</mi><mi>k</mi></msub></msub></math>

最后一项是按 √512 缩放的 512 维区域引用嵌入。没有候选框支持的引用被屏蔽；有候选也仍可能被模型用错。M 的逻辑组成是 256 个粗图格加区域槽，不要把有效位置数写死成 320。

### 2. 从视觉条件到下一 token

解码器用文本自注意力读**此前已生成的 token**，再通过视觉交叉注意力读 M。8 层完成后，将四条 mHC 残差支路（每条 512 维）的状态取平均并归一化，得到 512 维 h。共享的词嵌入矩阵 T（公开权重清单为 8299×512）将它映射到词表分数，`logits = T · h`。公开数学面板显示用词表掩码限制可选 token，并在有效候选中取最高分；选中的 token 成为下一步前缀。

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>p</mi><mo>(</mo><mi>Y</mi><mo>∣</mo><mi>M</mi><mo>)</mo><mo>=</mo><munderover><mo>∏</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mi>p</mi><mo>(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>∣</mo><msub><mi>y</mi><mrow><mo>&lt;</mo><mi>t</mi></mrow></msub><mo>,</mo><mi>M</mi><mo>)</mo></math>

Y 是全部 N 个内部语义 token，`y_<t` 是第 t 步之前的 token。公式只描述语义序列，**不是**整份 JSON 的概率或单独给 `reasoning` 打分，也不是 SDK 关键词影响解码的公式。

### 3. 区分生成文字、OCR 与约束表单

| 要看什么 | 走哪条分支 | 应如何解读 |
|---|---|---|
| 普通描述中的解释 | `vision.describe(image).get("reasoning")` | 生成文本，可能缺失或误判；没有核实到可依赖的 `reasoning=True` 开关。 |
| 图上的精确文字 | 独立 OCR 从原图的文字框裁图，组装到 `text` | 已公开计算图中，OCR 字符串不回流语义解码器。所以 trace 与 OCR 可能不同。 |
| 在固定选项中选答案 | `vision.extract(image, form)`，同模型的约束请求模式 | 枚举按候选完整 token 路径（含终止标记）评分，不把 trace 再交给另一模型；该模式不运行 OCR reader。 |

**教学示例（不是实测返回）**：同一张图，`reasoning` 可以写“看起来像编号 A18”，但 OCR 的 `text` 读为 `A13`。要取图上的编号，应以核对 OCR 字符串及原图为主，而不是抄解释。官方 VQAv2 实验使用固定 SDK 4.0.1／引擎 4.0.0；归档的 128 条图像记录都含非空字符串 `reasoning`，不代表所有选项组合或版本都返回它。

公开材料未披露 trace 的原生段落分隔符、确切生成顺序、训练样本或独立损失。给下游模型是否有帮助也要实测：官方让 Qwen3‑0.6B 读取 Mirage 的同图结构化结果，在 VQAv2 的 128 张图、256 道题上，不传 trace 得 33.1，传完整 trace 得 31.3（每百得分；配对差 −1.8，95% 区间 −5.0 至 1.2）。该实验没发现可靠增益，不能推广成所有任务上 trace 都无用。

依据：Cactus 发布页 What comes back、Maths，及 Structured Visuo-Semantic Perception 的结构计算图、区域槽、约束语义和 VQAv2 实验。


<a id="keys"></a>

## SDK： `describe` 里说的 key 是什么？

**这里讨论的是 SDK 参数。** Mirage Python SDK 的 `describe` 没有显式的单数 `key` 参数；它有一个专门的 **`keywords`** 参数。另外，`options` 是请求字典，里面也有键，例如 `sections`；返回结果同样是字典，`reasoning` 就是一个返回字段名。这三处不能互换。


<a id="weights-key"></a>

### 源码里的 `key`：用权重路径复用 SDK 对象

如果你问的是模块级快捷函数 `cactus.describe(image, *, weights=None, **options)` 里的字面变量 **`key`**，它是 Python 字典 `_shared` 的内部索引：

```python
key = os.fspath(weights) if weights is not None else None
if key not in _shared:
    _shared[key] = Mirage(weights=weights)
return _shared[key].describe(image, **options)
```

同一个权重路径值再次调用，复用同一个 Python `Mirage` 包装对象。比如两次都传 `weights="mirage.cact"`，就查同一条 `_shared["mirage.cact"]`；`weights=None` 统一用字典键 `None`，由构造函数取得默认权重。这里的 `os.fspath` 只取路径表示，不做绝对路径转换、文件身份或内容比较；传入两个字符串 `"mirage.cact"` 和 `"./mirage.cact"` 会得到两个不同的键。

`_shared[key]` 缓存的是**包装对象**；`_loaded` 是另一处状态，记录 Python 绑定器当前认可的已加载权重路径，用于决定是否再调用 `cactus_load`。如果同一路径已经被绑定器认定为已加载，**原地覆盖该权重文件并不保证重新加载**；要切换版本，使用不同的权重路径或重新启动进程。调用实例方法 `vision.describe(...)` 不经过这个模块级 `_shared[key]` 查找。`key` 不是要传给 SDK 的参数，不是 API 密钥、token，也不是 `keywords`。

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

### `keywords` 到底传了什么？

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

继续读：[04 API](04-api.md) · [实现与数学](02-implementation.md) · [模型结构](01-architecture.md)。

核对依据：随包 SDK 源码快照的 `cactus/agent/mirage.py`、`cactus/agent/audio.py`（共用参数编码）和 `cactus/agent/runtime.py`（C ABI 绑定）；Cactus Mirage 发布页及其 Structured Visuo-Semantic Perception 说明。
