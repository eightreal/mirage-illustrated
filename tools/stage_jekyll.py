#!/usr/bin/env python3
"""Stage Mirage as real Jekyll Markdown from the prior verified offline ZIP.

Usage:
    python3 stage_jekyll.py --zip ../Mirage_GitHub_Pages.zip --output site
    python3 stage_jekyll.py --zip ../Mirage_GitHub_Pages.zip --output site --layout doc_layout.html

Publish the generated site root via the official actions/jekyll-build-pages action.
Only the prior rendered <math> nodes are read. No prior HTML page is an input page.
The script is stdlib-only. Local Jekyll build validation is a separate step.
"""
from __future__ import annotations

import argparse
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import posixpath as pp
import re
import shutil
from tempfile import TemporaryDirectory
from urllib.parse import quote, unquote, urlsplit
from zipfile import ZipFile

# (source Markdown in the offline ZIP, preserved HTML permalink, key, nav label).
PAGES = [
    ("README.md", "index.html", "overview", "首页"),
    ("docs/01-architecture.md", "chapters/01-architecture.html", "architecture", "01 模型结构"),
    ("docs/02-implementation.md", "chapters/02-implementation.html", "implementation", "02 实现与数学"),
    ("docs/03-training.md", "chapters/03-training.html", "training", "03 训练"),
    ("docs/04-api.md", "chapters/04-api.html", "api", "04 API"),
    ("docs/05-model-files.md", "chapters/05-model-files.html", "model-files", "05 模型文件"),
    ("code/README.md", "code/README.html", "code", "参考代码"),
    ("upstream/README.md", "upstream/README.html", "upstream", "官方 SDK"),
]
REASON_PAGE = ("docs/06-reasoning.md", "chapters/06-reasoning.html", "reasoning", "06 Describe 参数与 Reasoning")
REASON_FIGURE = "assets/reasoning-model.svg"
REASON_ARCHITECTURE = "assets/reasoning-architecture.svg"
HELMET_FLOW = "assets/mirage-model-architecture.svg"
TRICK_FIGURES = {f"assets/tricks-{name}.svg" for name in ("encoder", "mixer", "heads", "deployment")} | {
    "assets/level-gate-example.svg", "assets/fpn-grid-example.svg", "assets/hconv-channel-example.svg",
    "assets/hconv-principle.svg", "assets/register-two-layers.svg", "assets/register-tensor-shapes.svg",
    "assets/hconv-walsh-matrices.svg", "assets/hconv-glu-matrices.svg",
}
META = ("config.json", "model_files.json", "tensor_manifest.json", "tensor_summary.csv",
        "toy_memory.json", "toy_model.npz", "verify_examples.py")
CODE = ("mirage_reference.py", "decoder_ops.py", "train_toy.py", "test_reference.py",
        "native_ctypes_demo.py", "official_api_demo.py", "requirements.txt")
SDK = ("LICENSE-Apache-2.0.txt", "cactus-sdk-bb822a3.tar.gz",
       "source/LICENSE", "source/cactus/agent/audio.py", "source/cactus/agent/mirage.py",
       "source/cactus/agent/runtime.py", "source/cactus/cli/parser.py",
       "source/cactus/cli/vision.py", "source/tests/test_mirage.py")
ROUTES = {source: url for source, url, *_ in [*PAGES, REASON_PAGE]}
STYLE_SNIP = re.compile(r"<style[^>]*>(.*?)</style>", re.S)
MATH = re.compile(r"<math\b[^>]*>.*?</math>", re.S)
ANNOTATION = re.compile(r'<annotation encoding="application/x-tex">(.*?)</annotation>', re.S)
TOKEN = re.compile(
    r"(?P<fence>^[ ]{0,3}(?P<fmark>`{3,}|~{3,})[^\n]*\n.*?"
    r"^[ ]{0,3}(?P=fmark)[ \t]*(?:\n|$))"
    r"|(?P<code>(?P<cmark>`+)[^\n`]*?(?P=cmark))"
    r"|(?P<display>(?<!\\)\$\$(?P<dtex>.*?)\$\$)"
    r"|(?P<inline>(?<![\\$])\$(?![ \t\n$])(?P<itex>(?:\\.|[^$\n])+?)(?<![ \t])\$(?![\d$]))",
    re.M | re.S,
)
LINK = re.compile(r"(?P<img>!?)\[(?P<label>[^\[\]\n]+)\]\((?P<url>[^()\n\s]+)\)")
STATIC_FOLDERS = {"assets", "code", "upstream", "model-metadata"}
BODY_VECTOR = "assets/impl-05-body.svg"
BODY_PREVIEW = "assets/impl-05-body-preview.svg"
BODY_CHANNEL_PREVIEW = "assets/impl-05-body-channel.svg"
BODY_VIEWS = {
    BODY_VECTOR: (636, 1673, "一个 Body Block：两个残差与 HConvGLU"),
    BODY_PREVIEW: (638, 638, "第一个残差：门控空间混合"),
    BODY_CHANNEL_PREVIEW: (1275, 486, "第二个残差：通道混合和 GRN"),
}
HOME_VECTORS = {"assets/overview-encoder.svg", "assets/overview-evidence.svg"}
SVG_OVERLAY = HOME_VECTORS | TRICK_FIGURES | {REASON_FIGURE, REASON_ARCHITECTURE, HELMET_FLOW,
    BODY_VECTOR,
    "assets/15_block_dimensions.svg",
    "assets/16_stem_backbone.svg",
    "assets/17_d_fusion.svg",
    "assets/18_mask_fusion.svg",
}
CONFIG = """title: "Mirage 图解技术文档"
lang: zh-CN
encoding: utf-8
markdown: kramdown
kramdown:
  input: GFM
  math_engine: null
  syntax_highlighter: rouge
strict_front_matter: true
exclude:
  - README.md
  - Gemfile
  - Gemfile.lock
  - vendor
"""
EXTRA_CSS = """
.site-nav {display:flex;flex-wrap:wrap;gap:.45rem .9rem;padding:0 0 1.1rem;
  font-size:.88rem;line-height:1.4;}
.site-nav a[aria-current="page"] { font-weight:750; text-decoration:none; }
h1 {margin:.8rem 0 1.2rem}
pre code { background:none; }
picture,img { max-width:100%;height:auto; }
:is([href$=".png"],[href$=".svg"]) > img { cursor:zoom-in; display:block; }
math { max-width:100%; }
math[display="block"] { overflow-x:auto; padding-block:.35rem; }
@media(max-width:600px) {body {padding:1.25rem 1rem; font-size:16px}}
"""


def must(condition, message):
    if not condition:
        raise ValueError(message)


def clean_path(name):
    p = PurePosixPath(name)
    must(not p.is_absolute() and ".." not in p.parts and name == p.as_posix()
         and "\\" not in name, f"Unsafe path {name!r}")
    return name


def write(base, name, data):
    dest = base / clean_path(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    dest.write_bytes(data)


def dump(base, name, data):
    write(base, name, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def norm_tex(x):
    return re.sub(r"\s+", "", x)


def protect_math(markdown, old_html, key):
    """Replace math source only, in order, with pure text markers for Kramdown."""
    rendered_tags = iter(MATH.findall(old_html))
    equations = []
    md_fences = []

    def replace(m):
        if m.group("fence"):
            md_fences.append(m.group(0))
            return m.group(0)
        if m.group("code"):
            return m.group(0)
        is_display = m.group("display") is not None
        tex = m.group("dtex") if is_display else m.group("itex")
        rendered = next(rendered_tags, None)
        must(rendered, f"{key}: missing prior MathML for {tex.strip()!r}")
        tag = ANNOTATION.search(rendered)
        must(tag, f"{key}: missing MathML TeX annotation")
        must(norm_tex(tex) == norm_tex(unescape(tag.group(1))),
             f"{key}: prior MathML and Markdown diverged for {tex.strip()!r}")
        must(('display="block"' in rendered) == is_display,
             f"{key}: display/inline mismatch for {tex.strip()!r}")
        must(re.search(r"</?(?:script|iframe|object)\b|\bon\w+\s*=", rendered, re.I) is None,
             f"{key}: unexpected active MathML")
        marker = f"MIRAGEMATH{key.replace('-', '').upper()}Q{len(equations)+1:04d}END"
        must(marker not in markdown, f"{key}: marker collided with input")
        equations.append({"marker": marker, "mathml": rendered})
        if is_display:
            # A standalone Markdown paragraph, avoiding juxtaposing consecutive displays.
            return f"\n\n{marker}\n\n"
        return marker

    protected = TOKEN.sub(replace, markdown)
    must(next(rendered_tags, None) is None, f"{key}: unused prior MathML")
    return protected, equations, md_fences


def update_markdown_links(md, old_path, url, found):
    def repl(m):
        raw = m.group("url")
        u = urlsplit(raw)
        must(not u.scheme and not u.netloc and not u.path.startswith("/"),
             f"External or root-absolute URL unexpected: {raw!r}")
        if not u.path:
            if u.fragment:
                found.append((url, url, unquote(u.fragment)))
            return m.group(0)
        old_target = pp.normpath(pp.join(pp.dirname(old_path), unquote(u.path)))
        clean_path(old_target)
        if old_target in ROUTES:
            new_target = ROUTES[old_target]
        else:
            new_target = old_target
            must(new_target.split("/", 1)[0] in STATIC_FOLDERS,
                 f"{old_path}: unsupported target {new_target}")
        if m.group("img") and new_target.endswith(".png") and new_target[:-4]+".svg" in SVG_OVERLAY:
            new_target = new_target[:-4]+".svg"
        found.append((url, new_target, unquote(u.fragment)))
        relative = quote(pp.relpath(new_target, start=pp.dirname(url) or "."), safe="/")
        relative += ("?" + u.query if u.query else "") + ("#" + u.fragment if u.fragment else "")
        label = m.group("label")
        if m.group("img"):
            must(new_target.startswith("assets/")
                 and (new_target.endswith(".png") or new_target in SVG_OVERLAY),
                 f"{old_path}: nonlocal or unapproved visual type")
            if new_target == BODY_VECTOR:
                def panel(asset, alt):
                    found.append((url, asset, ""))
                    preview = quote(pp.relpath(asset, start=pp.dirname(url) or "."), safe="/")
                    return f'[![{alt}]({preview})]({relative} "打开单个 Body Block 完整图")'
                spatial = panel(BODY_PREVIEW, "单个 Body 的第一段：空间门控与第一次残差")
                channel = panel(BODY_CHANNEL_PREVIEW, "同一个 Body 的第二段：通道混合与第二次残差")
                return spatial + "\n\n第一段产生的中间张量 **X̃** 直接进入下面的第二段。两张图连起来才是一个 Body Block。\n\n" + channel
            return f'[![{label}]({relative})]({relative} "打开原尺寸图片")'
        return f"[{label}]({relative})"
    # Existing fences and code spans must not be touched.
    gap = 0
    parts = []
    for match in TOKEN.finditer(md):
        if match.group("fence") is None and match.group("code") is None:
            continue
        parts += [LINK.sub(repl, md[gap:match.start()]), match.group(0)]
        gap = match.end()
    parts.append(LINK.sub(repl, md[gap:]))
    return "".join(parts)


def add_downloads(md, key, bundled, url, found):
    def links(folder, names, heading):
        mdlinks = [f"- [`{n}`]({quote(pp.relpath(folder+'/'+n, pp.dirname(url) or '.'),safe='/')})"
                   for n in names]
        found.extend((url, folder+"/"+n, "") for n in names)
        return f"\n\n## {heading}\n\n" + "\n".join(mdlinks) + "\n"

    if key == "overview":
        md = md.replace(
            "**在 GitHub 上从下方目录阅读；离线时，解压后打开 `打开这里.html`。**",
            "**本网页由 GitHub Pages 的 Jekyll 从项目 Markdown 构建。**",
        )
        md += links("model-metadata", META, "下载模型元数据和教学样例")
        if bundled:
            md += ("\n\n## 下载完整项目\n\n"
                   "[完整离线项目 ZIP](downloads/Mirage_GitHub_Pages.zip)。"
                   "其中保留可编辑的 Markdown、SVG、参考代码和源码说明。\n")
            found.append((url, "downloads/Mirage_GitHub_Pages.zip", ""))
    elif key == "code":
        md += links("code", CODE, "下载教学代码")
    elif key == "upstream":
        md += links("upstream", SDK[:2], "下载官方 SDK 归档和许可证")
    return md


def stage(zip_path, output, layout_path, bundled=False, baseurl=""):
    layout = layout_path.read_text(encoding="utf-8")
    must(all(word in layout for word in ("content", "relative_url", "mirage_equations",
         "eq.marker", "eq.mathml")),
         "Layout must render content, local equation data and relative URLs")
    must(not baseurl or re.fullmatch(r"/[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*", baseurl),
         "Baseurl must be empty or a path like /mirage-illustrated without a trailing slash")
    if baseurl:
        clean_path(baseurl.lstrip("/"))
    must(not output.exists(), f"Output already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    found = []
    math_data = {}
    page_contents = {}
    with ZipFile(zip_path) as archive, TemporaryDirectory(prefix="mirage-jekyll-",
                                                          dir=output.parent) as td:
        stage_root = Path(td) / "site"
        stage_root.mkdir()
        must(len(archive.namelist()) == len(set(archive.namelist())), "Duplicate ZIP members")
        for source, public, key, _ in PAGES:
            md = archive.read("source/" + source).decode("utf-8")
            old_html = archive.read("docs/" + public).decode("utf-8")
            heading, nl, content = md.partition("\n")
            must(heading.startswith("# ") and nl, f"{source}: missing title")
            must("{%" not in content and "{{" not in content,
                 f"{source}: unreviewed Liquid syntax")
            raw, eqs, fences = protect_math(content, old_html, key)
            if key == "architecture":
                next_image = "![D由三路特征融合得到]"
                must(raw.count(next_image) == 1, "Cannot locate Body insertion point")
                body_section = (
                    '<a id="body-structure"></a>\n\n'
                    '### 一个 Body Block 在做什么\n\n'
                    '一个 Body Block 输入和输出都是 **16×16×512**。它先在每个通道内混合邻近网格位置，'
                    '再逐位置混合 512 个通道，最后用 GRN 根据有效图像位置的响应调整通道强度。'
                    '这两段各有一条旁路，分别加回本段输入。图中白框是张量，彩色框是运算。\n\n'
                    '![单个 Body Block 剖面](../assets/impl-05-body.png)\n\n'
                    '公式见[实现章第 5 节](02-implementation.md#section-5)。\n\n'
                )
                raw = raw.replace(next_image, body_section + next_image, 1)
            if key == "implementation":
                raw = raw.replace("十六个空间 body blocks", "一个 Body Block 的内部结构", 2)
                raw = raw.replace(
                    "stem 后有16个卷积主体块；每块包括空间门控混合与 HConvGLU 通道混合。",
                    "单块由两段带残差的运算组成：先做空间门控混合，再做 HConvGLU 通道混合及 GRN。",
                    1,
                )
            if key in {"architecture", "implementation", "api"}:
                raw = (
                    "> 专题图解：[Reasoning 如何生成](06-reasoning.md#reasoning)"
                    " · [Describe 的 key 是什么](06-reasoning.md#keys)\n\n" + raw
                )
            converted = update_markdown_links(raw, source, public, found)
            converted = add_downloads(converted, key, bundled, public, found)
            current_fences = [m.group(0) for m in TOKEN.finditer(converted) if m.group("fence")]
            must(current_fences == fences, f"{source}: changed fenced code contents")
            math_data[key] = eqs
            fm = ("---\nlayout: default\n"
                  f"title: {json.dumps(heading[2:], ensure_ascii=False)}\n"
                  f"permalink: /{public}\nequation_key: {key}\n---\n")
            body = fm + heading + "\n\n" + converted.lstrip("\n")
            page_contents[public] = body
            # Avoid the repository README exclude matching a nested input in some Jekyll versions.
            input_path = public.removesuffix(".html") + ".md"
            if public.endswith("/README.html"):
                input_path = pp.dirname(public) + "/page.md"
            write(stage_root, input_path, body)
        reason_text = Path(__file__).with_name("reasoning.md").read_text(encoding="utf-8")
        must("{%" not in reason_text and "{{" not in reason_text, "Reason page has Liquid syntax")
        reason_source, reason_public, reason_key, _ = REASON_PAGE
        reason_text = update_markdown_links(reason_text, reason_source, reason_public, found)
        reason_title = "Mirage：Reasoning 原理与 SDK Describe 的 key"
        reason_body = ("---\nlayout: default\n"
                       f"title: {json.dumps(reason_title, ensure_ascii=False)}\n"
                       f"permalink: /{reason_public}\nequation_key: {reason_key}\n---\n"
                       f"# {reason_title}\n\n" + reason_text)
        page_contents[reason_public] = reason_body
        write(stage_root, reason_public.removesuffix(".html") + ".md", reason_body)
        urls = set(page_contents)
        static = {target for _,target,_ in found if target not in urls}
        must(all(not anchor or (target in urls and
                     (f'id="{anchor}"' in page_contents[target] or
                      f"{{#{anchor}}}" in page_contents[target]))
                 for _,target,anchor in found),
             "A Markdown fragment has no explicit anchor in the output")
        home = json.loads(Path(__file__).with_name("overview_assets.json").read_text(encoding="utf-8"))
        must(set(home) == HOME_VECTORS, "Overview SVG manifest mismatch")
        for name in sorted(static):
            if name == "downloads/Mirage_GitHub_Pages.zip":
                (stage_root / "downloads").mkdir()
                shutil.copyfile(zip_path, stage_root / name)
                continue
            if name in BODY_VIEWS:
                raw = Path(__file__).with_name("body_block.svg").read_bytes()
                old_view = b'height="2390" viewBox="0 0 720 2390"'
                must(old_view in raw, "Body artwork dimensions changed")
                y, height, title = BODY_VIEWS[name]
                raw = raw.replace(old_view, f'height="{height}" viewBox="0 {y} 720 {height}"'.encode(), 1)
                raw = re.sub(rb"<title>.*?</title>", ("<title>" + title + "</title>").encode("utf-8"), raw, count=1)
            elif name in ({REASON_FIGURE, REASON_ARCHITECTURE, HELMET_FLOW} | TRICK_FIGURES):
                raw = Path(__file__).with_name(PurePosixPath(name).name).read_bytes()
                must(raw.startswith(b"<svg") and b"<script" not in raw, "Invalid Reason SVG")
            elif name in HOME_VECTORS:
                raw = home[name].encode("utf-8")
            else:
                raw = archive.read(("source/" if name in SVG_OVERLAY else "docs/") + clean_path(name))
            if name in SVG_OVERLAY or name in BODY_VIEWS:
                must(b"<svg" in raw, f"Invalid SVG {name}")
            if name.endswith(".png"):
                must(raw.startswith(b"\x89PNG\r\n\x1a\n"), f"Invalid PNG {name}")
            write(stage_root, name, raw)
        dump(stage_root, "_data/mirage_equations.json", math_data)
        dump(stage_root, "_data/mirage_navigation.json",
             [{"url": "/"+public, "label": label} for _,public,_,label in [*PAGES, REASON_PAGE]])
        write(stage_root, "_config.yml", CONFIG + f"baseurl: {json.dumps(baseurl)}\n")
        write(stage_root, "_layouts/default.html", layout)
        css = archive.read("source/tools/offline_style.html").decode("utf-8")
        snippet = STYLE_SNIP.search(css)
        must(snippet, "No CSS in prior style fragment")
        write(stage_root, "site.css", snippet.group(1).strip()+"\n"+EXTRA_CSS)
        eq_count = sum(map(len, math_data.values()))
        must(eq_count and len(urls) == 9, "Missing pages or math")
        for key, entries in math_data.items():
            source = next(public for _,public,k,_ in PAGES if k == key)
            for record in entries:
                must(page_contents[source].count(record["marker"]) == 1,
                     f"{source}: marker missing or duplicated")
        no_pre_rendered = [p for p in stage_root.rglob("*.html") if "_layouts" not in p.parts]
        must(not no_pre_rendered, f"Unexpected static HTML page: {no_pre_rendered}")
        stage_root.rename(output)
    print(f"{output}: {len(urls)} Jekyll pages, {eq_count} MathML nodes, "
          f"{sum(n.endswith('.png') for n in static)} PNGs, {len(static & SVG_OVERLAY)} overlay-ready SVGs, "
          f"{len(static)} static resources.")


class BuiltPage(HTMLParser):
    def __init__(self, text):
        self.ids = []
        self.hrefs = []
        self.images = set()
        self.math = 0
        self.headings = 0
        self.code = 0
        super().__init__()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        self.math += tag == "math"
        self.headings += tag == "h1"
        self.code += tag == "pre"
        for attr in ("href", "src", "poster"):
            if values.get(attr):
                self.hrefs.append(values[attr])
        if tag == "img":
            must(values.get("alt"), "Built figure lost alt text")
            self.images.add(values["src"])


def check_built(output, baseurl):
    """Inspect the output of official Jekyll build, not the staged Markdown."""
    from urllib.parse import urljoin
    prefix = "https://pages-test.invalid" + baseurl + "/"
    pages = {}
    for _, url, _, _ in [*PAGES, REASON_PAGE]:
        path = output / url
        text = path.read_text(encoding="utf-8")
        must(not re.search(r"MIRAGEMATH[A-Z]+Q\d+END|\{%|\{\{ content", text),
             f"{path}: unresolved Liquid or MathML marker")
        page = BuiltPage(text)
        must(page.headings == 1 and len(page.ids) == len(set(page.ids)), f"{path}: heading/ID issue")
        pages[url] = page
    architecture = pages["chapters/01-architecture.html"]
    must("body-structure" in architecture.ids and any(link.endswith("/impl-05-body-preview.svg") for link in architecture.images)
         and any(link.endswith("/impl-05-body-channel.svg") for link in architecture.images)
         and any(link.endswith("/impl-05-body.svg") for link in architecture.hrefs),
         "Missing Body architecture anchor or figure")
    implementation = pages["chapters/02-implementation.html"]
    must(any(link.endswith("/impl-05-body-preview.svg") for link in implementation.images)
         and any(link.endswith("/impl-05-body-channel.svg") for link in implementation.images)
         and any(link.endswith("/impl-05-body.svg") for link in implementation.hrefs), "Missing Body in implementation")
    must(implementation.math == 218 and implementation.code >= 4, "Lost MathML or code fences")
    must(all(f"section-{i}" in implementation.ids for i in range(1,22)), "Missing implementation anchors")
    reason = pages[REASON_PAGE[1]]
    must({"data-flow", "tricks", "level-gates", "fpn-detail", "hconv-detail", "hconv-principle", "hconv-matrix", "register-detail", "reasoning", "semantic-output", "keys", "keywords", "weights-key"} <= set(reason.ids) and reason.math == 3
         and reason.code >= 3 and all(any(link.endswith("/"+PurePosixPath(asset).name)
                 for link in reason.images) for asset in {REASON_FIGURE, REASON_ARCHITECTURE, HELMET_FLOW} | TRICK_FIGURES),
         "Missing Reasoning/SDK anchors, semantic MathML, examples or a required figure")
    for _, url, _, _ in [*PAGES, REASON_PAGE]:
        must(all(any(link.endswith("06-reasoning.html" + anchor) for link in pages[url].hrefs)
                 for anchor in ("#reasoning", "#keys")), f"{url}: missing Reasoning/SDK Keys navigation")
    images = set()
    for url, page in pages.items():
        for link in page.hrefs:
            resolved = urljoin(prefix + url, link)
            must(resolved.startswith(prefix), f"{url}: external or prefix-escaping URL {link}")
            u = urlsplit(resolved)
            target = unquote(u.path[len(baseurl)+1:]) or "index.html"
            if target.endswith("/"):
                target += "index.html"
            must((output / clean_path(target)).is_file(), f"{url}: missing {target}")
            if link in page.images:
                images.add(target)
            if u.fragment and target in pages:
                must(unquote(u.fragment) in pages[target].ids, f"{url}: broken anchor {link}")
    must(((SVG_OVERLAY-{BODY_VECTOR}) | {BODY_PREVIEW, BODY_CHANNEL_PREVIEW, REASON_FIGURE}) <= images and len(images) == 87,
         "Missing figures or SVG overlays")
    print(f"Built Jekyll site check passed: {len(pages)} pages, {len(images)} figures, prefix {baseurl or '/'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="Prior verified Mirage_GitHub_Pages.zip")
    parser.add_argument("--output", type=Path, help="New directory for genuine Jekyll sources")
    parser.add_argument("--check-built", type=Path, help="Check generated HTML after official Jekyll build")
    parser.add_argument("--layout", type=Path, default=Path(__file__).with_name("doc_layout.html"),
                        help="Liquid layout; defaults to doc_layout.html beside this script")
    parser.add_argument("--baseurl", default="", help="Project URL prefix, e.g. /mirage-illustrated")
    parser.add_argument("--bundle-download", action="store_true", help="Also copy/link the full prior ZIP")
    args = parser.parse_args()
    if args.check_built:
        check_built(args.check_built.resolve(), args.baseurl)
    elif args.zip and args.output:
        stage(args.zip.resolve(), args.output.resolve(), args.layout.resolve(), args.bundle_download, args.baseurl)
    else:
        parser.error("use --zip and --output, or use --check-built after the Jekyll build")


if __name__ == "__main__":
    main()
