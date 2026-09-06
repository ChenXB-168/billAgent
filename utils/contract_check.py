# -*- coding: utf-8 -*-
"""契约自动核对（contract_check）——文档 ↔ 代码漂移检测。

背景
----
`设计_从零系列/09_模块契约手册.md` 的原则是"接口名与签名以代码为准"，但历史上
靠人工 review 发现漂移（v2.1→v2.2 十几处勘误）。本脚本让漂移在**开工 preflight /
测试阶段**被机器抓住：

- 检查 A（符号存在性，FAIL 级）：文档接口表里写的符号（函数 / 类 / 类方法 /
  顶层常量），在文档指定的代码文件里必须存在。文档写了但代码已删/改名 → FAIL。
- 检查 B（行号引用有效性，FAIL 级）：文档里带目录路径的 `xxx.py L123` / `xxx.py:123`
  引用，文件必须存在、行号不得超出文件行数。
- 检查 C（签名参数核对，WARN 级）：接口表签名列里出现的参数名，代码定义里应有
  （抓"死参数 / 错参数名"类漂移，如 M2 删 `send_result` 死参数）。因文档签名是
  非结构化文本，仅告警、不阻塞。

用法
----
    python utils/contract_check.py                 # 全量检查（默认读从零系列 09）
    python utils/contract_check.py --full          # 打印全部 PASS/WARN，而非只打印问题
    python utils/contract_check.py --check-params  # 额外做检查 C（默认关闭）

退出码：0 = 无 FAIL；1 = 存在 FAIL（开工前必须处理）。

接入方式：`tests/unit/test_contract_check.py` 直接 import 本模块的 `check_all`，
保证 L1 全量跑时漂移即被抓。
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# ★v3.0 基线：接口契约的**唯一权威**改为从零系列 09（旧 `设计/10` 降级为改造期记录，
#   不再参与校验，避免"两份权威互相打架"）。
DEFAULT_DOC = REPO_ROOT / "设计_从零系列" / "09_模块契约手册.md"

# 顶层包目录（文档里相对路径不含顶级包名时，用后缀匹配在这些目录内查找）
SKIP_DIRS = {"venv", ".venv", "__pycache__", ".git", ".codebuddy", "node_modules", "generated-images"}

# 接口表数据行级跳过词（这些行表示"还没写代码 / 非代码对象"）
ROW_SKIP_WORDS = ("To-Be", "未落地", "0B", "占位")

# 小节/标题级跳过词（整个小节不核对）
SECTION_SKIP_WORDS = ("To-Be", "未落地", "占位", "0B")

# 签名/方法文本提取时的停用词（类型、语法、装饰器词，不是符号/参数名）
STOP_WORDS = {
    "self", "cls",
    # 类型词
    "str", "int", "float", "bool", "list", "dict", "tuple", "set", "bytes", "None",
    "Any", "Optional", "Union", "List", "Dict", "Tuple", "Set", "Callable",
    "Awaitable", "Literal", "TypedDict", "Iterable", "Mapping", "Type", "Final",
    # 语法词
    "async", "await", "def", "class", "return", "raise", "yield", "lambda",
    "if", "elif", "else", "for", "while", "with", "import", "from", "in", "is",
    "not", "and", "or", "True", "False", "pass", "global", "del", "match",
    # 装饰器 / 框架词
    "contextmanager", "asynccontextmanager", "staticmethod", "classmethod",
    "property", "cached_property", "dataclass", "final", "overload", "override",
    "pytest", "fixture", "mark", "parametrize",
}

# ---------------------------------------------------------------- 解析工具

# ★v3.0：不再硬编码 §5.6 编号——新 09 采用 `## N. 主题` / `### N.N 小节` 的编号体系，
#   这里统一按"任意层级编号标题"识别作用域（同时兼容旧的 5.6.x 写法）。
_HEAD_RE = re.compile(r"^(?:#{2,4})\s*\d+\.\d+(?:\.\d+)?\b")
_BOLD_RE = re.compile(r"^\*\*\s*5\.6\.(\d+)(?:\.(\d+))?")           # 旧格式兼容，v3.0 起不再使用
_BOLD_PATH_RE = re.compile(r"^\*\*[^*]*")                            # **...** 任意粗体行（含路径）

_CODE_TAG_RE = re.compile(r"`([^`]+)`")
_ROW_REF_RE = re.compile(r"^\s*\|")
_SEP_ROW_RE = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
# 匹配 `name(`：前导不能是字母/下划线/点（排除 `asyncio.Lock(` 里的 Lock）
_METHOD_HINT_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(")

# 行号引用：`path/xxx.py` 后跟 `L123` / `:123` / `L30-40`（反引号与空格可夹在中间）
_LINE_REF_RE = re.compile(
    r"([A-Za-z0-9_./\\-]+\.py)\s*[` ]{0,2}\s*(?:L|:|：)\s*(\d{2,5})(?:[-–](\d{2,5}))?\+?"
)


def _iter_py_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(REPO_ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        files.append(p)
    return files


def _module_paths() -> set[str]:
    """仓库内全部 .py 的相对路径（正斜杠）。"""
    return {p.relative_to(REPO_ROOT).as_posix() for p in _iter_py_files(REPO_ROOT)}


def locate_file(ref: str, index: set[str]) -> tuple[Path | None, str | None]:
    """按文档引用定位仓库文件。返回 (文件, 状态)。ref 为文档里写的路径。"""
    norm = ref.replace("\\", "/").lstrip("./")
    if norm in index:
        return REPO_ROOT / norm, "ok"
    # 后缀匹配：只接受唯一命中，避免 nodes.py 之类多义
    hits = sorted(i for i in index if i == norm or i.endswith("/" + norm))
    if len(hits) == 1:
        return REPO_ROOT / hits[0], "ok"
    if not hits:
        return None, "missing"
    return None, "ambiguous"


@dataclass
class CodeSymbols:
    """一个文件（或目录并集）的 AST 符号集。"""
    top_level: set[str] = field(default_factory=set)   # 顶层 def/class/赋值常量
    classes: set[str] = field(default_factory=set)     # 顶层 class 名
    members: dict[str, set[str]] = field(default_factory=dict)  # class -> {方法/嵌套类/类属性}
    func_params: dict[str, set[str]] = field(default_factory=dict)  # sym -> 参数名集

    def wide(self) -> set[str]:
        """宽符号集：顶层 + 全部类成员。目录并集模式用（容忍文件归属不精确）。"""
        s = set(self.top_level)
        for m in self.members.values():
            s |= m
        return s


def _collect_code_symbols(path: Path) -> CodeSymbols:
    """AST 解析单个文件。解析失败时抛 SyntaxError 由调用方处理。"""
    syms = CodeSymbols()
    try:
        # 代码里历史遗留的无效转义（如 long_memory.py 的 "\d"）会打 SyntaxWarning，
        # 与核对无关——静默解析，避免污染输出。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        raise
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            syms.top_level.add(node.name)
            _index_params(syms.func_params, node)
            _index_nested(syms, node, node.name)   # ★v3.0：闭包内定义的嵌套函数也算可核对符号
        elif isinstance(node, ast.ClassDef):
            syms.top_level.add(node.name)
            syms.classes.add(node.name)
            members = syms.members.setdefault(node.name, set())
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(sub.name)
                    _index_params(syms.func_params, sub, f"{node.name}.{sub.name}")
                elif isinstance(sub, ast.ClassDef):
                    members.add(sub.name)
                    for subsub in sub.body:
                        if isinstance(subsub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            members.add(f"{sub.name}.{subsub.name}")
                            _index_params(syms.func_params, subsub,
                                          f"{node.name}.{sub.name}.{subsub.name}")
                else:
                    # dataclass 字段 / 枚举成员等类属性（AnnAssign / Assign）
                    if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                        members.add(sub.target.id)
                    elif isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if isinstance(t, ast.Name):
                                members.add(t.id)
                        # ★v3.0：`__slots__ = ("a","b")` 声明的字段也是可核对契约
                        #   （如 `memory/user_docs.py` 的 `_DocState`），否则文档写字段即假 FAIL
                        if (len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name)
                                and sub.targets[0].id == "__slots__"
                                and isinstance(sub.value, (ast.Tuple, ast.List))):
                            for el in sub.value.elts:
                                if isinstance(el, ast.Constant) and isinstance(el.value, str):
                                    members.add(el.value)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    syms.top_level.add(t.id)
    return syms


def _index_nested(syms: "CodeSymbols", node: ast.AST, parent: str) -> None:
    """★v3.0：递归登记嵌套 def（闭包/内部函数）为父函数的成员，供 wide() 核对。

    典型场景：`parse_json_output` 内部定义的 `try_extract`。这类函数虽非顶层，
    但文档会把它写进接口表（它是真实存在的可核对对象），不登记会产生假 FAIL。
    """
    for sub in ast.walk(node):
        if sub is node or not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        syms.members.setdefault(parent, set()).add(sub.name)
        _index_params(syms.func_params, sub, f"{parent}.{sub.name}")


def _index_params(store: dict[str, set[str]], node: ast.FunctionDef | ast.AsyncFunctionDef,
                  prefix: str = "") -> None:
    args = node.args
    names = [a.arg for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    store[prefix + node.name] = {n for n in names if n}


@dataclass
class Issue:
    level: str            # FAIL / WARN
    kind: str             # symbol / line_ref / param
    where: str            # 文档行号或位置
    file: str             # 代码文件（尽量）
    symbol: str
    msg: str


def _clean_tokens(cell: str) -> list[str]:
    """从接口列单元格提取纯符号候选。

    规则：
    - 反引号内容优先（用户特意标记的代码符号）；无反引号时按 `/`、`、` 拆分。
    - 剥掉尾随参数括号 `name(...)` → `name`（方法写成带括号是文档惯例）。
    - 只接受"纯符号形态"（`name` 或 `Class.member`），其余（常量组大括号、
      类型词、中文）一律丢弃，避免把参数名/描述词误当接口符号。
    """
    tagged = _CODE_TAG_RE.findall(cell)
    pieces = tagged if tagged else re.split(r"\s*[/、]\s*", cell)
    out = []
    for tok in pieces:
        tok = tok.strip().strip("`").strip()
        # ★v3.0：接口列逐字抄 def 行（"async def foo(x: int) -> bool"）→ 剥前缀/返回类型/参数
        tok = re.sub(r"^(?:async\s+)?(?:def|class)\s+", "", tok)
        tok = re.sub(r"\s*->.*$", "", tok)
        tok = re.sub(r"\(.*\)$", "", tok).strip()          # 剥尾括号参数
        if tok.endswith(".py"):
            continue
        # ★v3.0：接口列末尾的出处行号（`L41` / `L294-311` / `L55-188`）不是符号，必须丢弃
        if re.fullmatch(r"L?\d+(?:-\d+)?", tok):
            continue
        if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?", tok) and tok not in STOP_WORDS:
            out.append(tok)
    return out


def _symbol_exists(syms: CodeSymbols, sym: str) -> bool:
    if "." in sym:
        cls, _, mem = sym.partition(".")
        if cls in syms.classes:
            return mem in syms.members.get(cls, set())
        # module.func / 目录并集宽匹配
        return mem in syms.wide()
    return sym in syms.wide() or sym in syms.top_level


# ★v3.0：接口列常把多个函数写在一格（"def a(x: int)、def b(y: str)"）。整格提参数会把
#   a 的参数算到 b 头上 → 成片噪音 WARN。这里按 `def name(...)` 分段，参数核对精确到函数。
_DEF_SEG_RE = re.compile(r"(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(([^()]*)\)")


def _segment_params(text: str) -> dict[str, set[str]]:
    """把一段文本里的每个 `def name(args)` 单独解析出参数名集合。"""
    return {m.group(1): _extract_sig_params("(" + m.group(2))
            for m in _DEF_SEG_RE.finditer(text)}


def _extract_sig_params(sig: str) -> set[str]:
    """启发式从文档签名文本提取参数名。只用于 WARN，不许 FAIL。"""
    params = set(re.findall(r"[\s,(]([A-Za-z_]\w*)\s*(?=[:=])", sig))
    return {p for p in params if p not in STOP_WORDS and p not in ("self", "cls")}


def _method_hints(text: str) -> set[str]:
    out = set(_METHOD_HINT_RE.findall(text))
    return {t for t in out if t not in STOP_WORDS}


def _check_params(syms: CodeSymbols, sym: str, doc_params: set[str]) -> list[str]:
    """检查 C：文档参数名 vs 代码参数名。返回不一致的文档参数名列表。"""
    if not doc_params:
        return []
    code_params = syms.func_params.get(sym)
    if code_params is None:
        # 仅当符号是纯函数/方法且已定位到时才核对；否则放弃
        return []
    return sorted(d for d in doc_params if d not in code_params)


# ---------------------------------------------------------------- 主检查

def check_all(doc_path: Path | None = None, repo_root: Path | None = None,
              check_params: bool = False) -> tuple[list[Issue], dict]:
    """核对文档与代码，返回 (问题列表, 统计信息)。"""
    doc_path = Path(doc_path) if doc_path else DEFAULT_DOC
    repo_root = Path(repo_root) if repo_root else REPO_ROOT
    text = doc_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    issues: list[Issue] = []
    stats = {"symbols_checked": 0, "line_refs": 0}

    index = _module_paths()
    cache: dict[Path, CodeSymbols | None] = {}
    # 已登记过的符号，避免同文件重复报（报告按行给出仍可查）
    seen: set[tuple[str, str]] = set()

    # ---------- 作用域：整篇文档（v3.0 起不再依赖 §5.6 编号定位） ----------
    cur_binding: list[Path] = []          # 当前作用域的代码文件/目录
    section_skip = False
    in_interface_table = False
    header_cells: list[str] = []

    def update_binding(path_tokens: list[str], lineno: int) -> None:
        nonlocal cur_binding
        cur_binding = []
        for tok in path_tokens:
            tok = tok.strip().strip("`")
            if any(ch in tok for ch in "*{}?"):     # glob 写法（如 agents/{a,b}_agent/）不是真实路径
                continue
            if tok.endswith(".py"):
                f, st = locate_file(tok, index)
                if f is not None:
                    cur_binding.append(f)
                elif st == "missing":
                    issues.append(Issue("FAIL", "file_ref", str(lineno), tok, "",
                                        f"文档引用的代码文件不存在（路径已删或改名？）"))
            elif tok.endswith("/"):
                d = repo_root / tok.rstrip("/")
                if d.is_dir():
                    cur_binding.extend(sorted(p for p in d.rglob("*.py")
                                              if not any(part in SKIP_DIRS for part in p.relative_to(REPO_ROOT).parts)))
                else:
                    issues.append(Issue("FAIL", "dir_ref", str(lineno), tok, "",
                                        f"文档引用的目录不存在：{tok}"))

    def load_symbols(paths: list[Path]) -> CodeSymbols | None:
        """合并多个文件的符号；目录并集时取宽集。返回 None 表示无有效代码。"""
        merged = CodeSymbols()
        for p in paths:
            if p in cache:
                syms = cache[p]
            else:
                try:
                    syms = _collect_code_symbols(p)
                except (SyntaxError, OSError):
                    issues.append(Issue("WARN", "parse", "", p.as_posix(), "",
                                        "代码文件无法解析（语法错误/编码）"))
                    syms = None
                cache[p] = syms
            if syms is not None:
                merged.top_level |= syms.top_level
                merged.classes |= syms.classes
                for k, v in syms.members.items():
                    merged.members.setdefault(k, set()).update(v)
                merged.func_params.update(syms.func_params)
        return merged

    for lineno in range(len(lines)):
        raw = lines[lineno]
        line = raw.strip()

        # --- 编号标题行（## 1. … / ### 4.2 …）：更新绑定 / 跳过标记 ---
        if _HEAD_RE.match(line):
            section_skip = any(w in line for w in SECTION_SKIP_WORDS)
            in_interface_table = False
            if section_skip:
                continue
            path_tokens = _CODE_TAG_RE.findall(line)
            update_binding(path_tokens, lineno)
            continue

        # --- 独立粗体定位行（如 **`startup/bootstrap.py`**）：更新绑定 ---
        # 只认"整行粗体 + 含 .py 反引号 + 非表格行"；表格单元格里的 .py 是上下游引用，不算绑定。
        if (line.startswith("**") and "|" not in line and "`" in line
                and not _SEP_ROW_RE.match(line)):
            path_tokens = [t for t in _CODE_TAG_RE.findall(line)
                           if t.endswith(".py") or t.endswith("/")]
            if path_tokens and not section_skip:
                update_binding(path_tokens, lineno)

        # --- 表头行：识别"接口"表（接口列 + 可选签名列） ---
        if _ROW_REF_RE.match(line) and not _SEP_ROW_RE.match(line):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if any(c.startswith("接口") for c in cells) or any(c.startswith("签名") for c in cells):
                header_cells = cells
                in_interface_table = True
                continue

        # 非接口表的内容（其它表头 / 说明文本）：直接跳过
        if not in_interface_table:
            continue

        # --- 分隔行（|---|）：跳过但不断表 ---
        if _SEP_ROW_RE.match(line):
            continue

        # --- 非表格行 → 表结束 ---
        if not _ROW_REF_RE.match(line):
            in_interface_table = False
            continue

        # --- 数据行 ---
        cells = [c.strip() for c in line.strip("|").split("|")]
        # v3.0：表头是"接口（逐字）"/"远端签名（逐字）"这类带修饰的列 → 前缀匹配
        iface_idx = next((i for i, c in enumerate(header_cells) if c.startswith("接口")), -1)
        if iface_idx < 0:
            continue          # 无接口列的表（仅签名）忽略
        if iface_idx < 0 or iface_idx >= len(cells):
            continue
        iface_cell = cells[iface_idx]
        if any(w in line for w in ROW_SKIP_WORDS):
            continue

        # 绑定：接口列内出现的 .py（如 `EmbeddingService`（`embedding_loader.py`））优先覆盖小节绑定。
        # 注意只看接口列：连接/签名列里的 .py 是上下游引用，不得覆盖（否则会张冠李戴）。
        row_path_tokens = [t for t in _CODE_TAG_RE.findall(iface_cell) if t.endswith(".py")]
        bind_paths: list[Path]
        if row_path_tokens:
            bind_paths = []
            for tok in row_path_tokens:
                f, st = locate_file(tok, index)
                if f is not None:
                    bind_paths.append(f)
            if not bind_paths:
                continue
        else:
            bind_paths = cur_binding
        if not bind_paths:
            continue

        syms = load_symbols(bind_paths)
        if syms is None:
            continue

        symbols = _clean_tokens(iface_cell)
        sig_col = next((i for i, c in enumerate(header_cells) if c.startswith("签名")), -1)
        # v3.0：多数表没有独立签名列——签名就在接口列里（逐字抄的 def 行），参数核对回退到接口列
        param_source = cells[sig_col] if 0 <= sig_col < len(cells) else iface_cell
        seg_params = _segment_params(param_source) if check_params else {}

        for sym in symbols:
            # 模块限定引用（asyncio.run / client.call_bill_sql / a2a_bus.send_task）：
            # 前缀既不是本文件的类也不是顶层符号 → 属"文中引用另一模块"，不在本文件核对
            if "." in sym and (sym.split(".")[0] not in syms.classes
                               and sym.split(".")[0] not in syms.top_level):
                continue
            key = (bind_paths[0].as_posix(), sym)
            if key in seen:
                continue
            seen.add(key)
            stats["symbols_checked"] += 1
            if not _symbol_exists(syms, sym):
                issues.append(Issue(
                    "FAIL", "symbol", str(lineno + 1),
                    ", ".join(p.as_posix() for p in bind_paths),
                    sym,
                    "文档接口表中列出的符号在代码中不存在（已删/改名？还是文件绑定错了？）"))
                continue
            # 检查 C：签名参数核对（无独立签名列时取接口列内的 def 行；一格多函数时按段取）
            if check_params:
                # 有独立分段就用分段（哪怕参数为空——无类型标注的签名不该拿别人的参数来核对）；
                # 只有"该符号没有 def 分段"时才回退整格
                doc_params = seg_params.get(sym)
                if doc_params is None:
                    doc_params = _extract_sig_params(param_source)
                if doc_params:
                    for d in _check_params(syms, sym, doc_params):
                        issues.append(Issue(
                            "WARN", "param", str(lineno + 1),
                            ", ".join(p.as_posix() for p in bind_paths),
                            f"{sym} 参数 {d}",
                            "文档签名中的参数名在代码定义里找不到（死参数/错名/解析误差）"))

        # 方法提示（签名列文本里的 method(...)）：宽符号找不到只 WARN
        if sig_col >= 0 and sig_col < len(cells):
            hints = _method_hints(cells[sig_col])
            wide = syms.wide()
            for h in hints:
                if h not in wide:
                    key = (bind_paths[0].as_posix(), f"?{h}")
                    if key in seen:
                        continue
                    seen.add(key)
                    issues.append(Issue(
                        "WARN", "symbol", str(lineno + 1),
                        ", ".join(p.as_posix() for p in bind_paths), h,
                        "接口描述中提到的符号在代码中未找到（已删/改名，或仅是文中用语）"))

    # ---------- 检查 B：带行号引用（全文档范围） ----------
    for lineno, ln in enumerate(lines, 1):
        for m in _LINE_REF_RE.finditer(ln):
            stats["line_refs"] += 1
            ref = m.group(1)
            f, st = locate_file(ref, index)
            if st == "missing":
                issues.append(Issue("FAIL", "line_ref", str(lineno), ref, "",
                                    "文档引用的 .py 文件不存在"))
                continue
            if st == "ambiguous":
                continue  # 纯文件名多义（nodes.py 等），不自动判定
            assert f is not None
            try:
                n = len(f.read_text(encoding="utf-8", errors="replace").splitlines())
            except OSError:
                continue
            s, e = int(m.group(2)), int(m.group(3) or m.group(2))
            if s > n or e > n:
                issues.append(Issue(
                    "FAIL", "line_ref", str(lineno), ref, f"L{s}",
                    f"行号超出文件实际行数（文件共 {n} 行）——行号已漂移，须实测后修正"))

    return issues, stats


# ---------------------------------------------------------------- 报告

def _summarize(issues: list[Issue], full: bool, stats: dict | None = None) -> int:
    if not issues:
        print("契约核对通过：未发现文档-代码漂移。")
        if stats:
            print(f"覆盖：接口符号 {stats['symbols_checked']} 个 / 行号引用 {stats['line_refs']} 处均已核对。")
        return 0
    fails = [i for i in issues if i.level == "FAIL"]
    warns = [i for i in issues if i.level == "WARN"]
    by_level = {"FAIL": fails, "WARN": warns}
    for level, lst in by_level.items():
        print(f"\n{'=' * 70}\n{level}（{len(lst)} 条）\n{'=' * 70}")
        if not full and level == "WARN":
            print(f"（WARN 不阻塞，共 {len(warns)} 条；用 --full 查看全部）")
            continue
        for it in lst:
            loc = f"文档第 {it.where} 行" if it.where else ""
            print(f"- [{it.kind}] {loc}  {it.file}  {it.symbol or ''}")
            print(f"    {it.msg}")
    if stats:
        print(f"覆盖：接口符号 {stats['symbols_checked']} 个 / 行号引用 {stats['line_refs']} 处。")
    print(f"小结：FAIL {len(fails)}（须处理）/ WARN {len(warns)}（建议核对）。")
    return 1 if fails else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="契约自动核对：文档接口 ↔ 代码漂移检测")
    ap.add_argument("--doc", default=str(DEFAULT_DOC),
                    help="要核对的文档（默认 设计_从零系列/09_模块契约手册.md）")
    ap.add_argument("--check-params", action="store_true", help="开启签名参数名核对（WARN）")
    ap.add_argument("--full", action="store_true", help="打印全部 WARN 与详情")
    args = ap.parse_args(argv)

    issues, stats = check_all(Path(args.doc), check_params=args.check_params)
    return _summarize(issues, full=args.full, stats=stats)


if __name__ == "__main__":
    sys.exit(main())
