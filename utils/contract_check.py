# -*- coding: utf-8 -*-
"""契约自动核对（contract_check）——文档 ↔ 代码漂移检测。

背景
----
`设计_从零系列/09_模块契约手册.md` 的原则是"接口名与签名以代码为准"，但历史上
靠人工 review 发现漂移（v2.1→v2.2 十几处勘误）。本脚本让漂移在**开工 preflight /
测试阶段**被机器抓住：

- 检查 A（符号存在性，FAIL 级）：文档接口表里写的符号（函数 / 类 / 类方法 /
  顶层常量 / 函数内嵌套函数），在文档指定的代码文件里必须存在。文档写了但代码已删/改名 → FAIL。
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
    """
    函数功能与逻辑描述：
        递归收集 root 下的全部 .py 文件，并按 SKIP_DIRS 过滤掉虚拟环境、字节码缓存、
        版本库与依赖等非项目源码目录；过滤依据是相对 REPO_ROOT 的路径分段，
        因此 root 必须落在 REPO_ROOT 之内，否则 relative_to 会抛 ValueError（不在本函数兜底）。
    入参说明：
        root (Path)：遍历起点目录，须位于 REPO_ROOT 之内。
    返回值说明：
        list[Path]：过滤后保留的 .py 绝对路径列表；无命中时返回空列表。
    """
    files = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(REPO_ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        files.append(p)
    return files


def _module_paths() -> set[str]:
    """
    函数功能与逻辑描述：
        构建"仓库内全部可核对 Python 文件"的相对路径索引，供 locate_file 做精确/后缀匹配；
        路径统一转成 POSIX 正斜杠形式，保证跨平台（Windows 反斜杠）比对一致。
        数据源为 _iter_py_files(REPO_ROOT)，因此同样受 SKIP_DIRS 过滤。
    入参说明：
        无。
    返回值说明：
        set[str]：形如 "utils/contract_check.py" 的相对路径集合；仓库无 .py 时为空集。
    """
    return {p.relative_to(REPO_ROOT).as_posix() for p in _iter_py_files(REPO_ROOT)}


def locate_file(ref: str, index: set[str]) -> tuple[Path | None, str | None]:
    """
    函数功能与逻辑描述：
        把文档里写的文件引用（可能省略顶级包前缀、可能用反斜杠）解析成仓库内真实文件路径。
        策略：先规范化分隔符与开头的 "./"，命中索引则直接返回；否则退化为"后缀唯一匹配"
        （如文档写 nodes.py 而仓库有多个同名文件时判为 ambiguous，避免张冠李戴）；
        只接受唯一命中，多义与未命中都不定位。
    入参说明：
        ref (str)：文档里的路径引用原文，允许含反斜杠与 "./" 前缀，可为完整相对路径或纯文件名。
        index (set[str])：仓库相对路径索引（通常由 _module_paths() 生成）。
    返回值说明：
        tuple[Path | None, str | None]：二元组 (文件绝对路径, 状态码)
            - 状态 "ok"：命中唯一文件，路径非 None；
            - 状态 "missing"：完全未命中，路径为 None；
            - 状态 "ambiguous"：纯文件名等多义引用，命中多个，路径为 None。
    """
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
    """
    函数/类功能与逻辑描述：
        单个代码文件（或目录并集）经 AST 解析后得到的符号集合，是检查 A/B/C 的核对依据；
        既登记顶层符号，也按父符号分组登记类成员、类属性与函数内嵌套函数，
        并为函数/方法登记参数名，供"文档接口表 ↔ 代码定义"逐项比对。
    构造入参说明：
        top_level (set[str])：顶层 def/class/赋值常量名；默认空集。
        classes (set[str])：顶层 class 名；默认空集。
        members (dict[str, set[str]])：父符号名（类名或函数名）→ 其成员名集合（方法/嵌套类/
            类属性/__slots__ 声明的字段/函数内嵌套函数）；默认空字典。
        func_params (dict[str, set[str]])：函数限定名（如 "Class.method" / "func.nested" / "func"）
            → 参数名集合；默认空字典。
    返回值说明：
        构造返回 CodeSymbols 实例；符号查询见 `wide`。
    """
    top_level: set[str] = field(default_factory=set)   # 顶层 def/class/赋值常量
    classes: set[str] = field(default_factory=set)     # 顶层 class 名
    members: dict[str, set[str]] = field(default_factory=dict)  # 父符号名(类/函数) -> {方法/嵌套类/类属性/__slots__字段/嵌套函数}
    func_params: dict[str, set[str]] = field(default_factory=dict)  # 限定名(Class.method/func.nested/func) -> 参数名集

    def wide(self) -> set[str]:
        """
        函数功能与逻辑描述：
            把顶层符号与全部父符号下的成员合并成一个"宽符号集"，用于目录并集绑定与容错匹配
            （文件归属不精确时只要求符号在并集中出现即可，不苛求精确到某个文件）。
        入参说明：
            无。
        返回值说明：
            set[str]：顶层符号 ∪ 全部 members 成员（含类方法、类属性、嵌套函数）的并集；
                实例为空时返回空集。
        """
        s = set(self.top_level)
        for m in self.members.values():
            s |= m
        return s


def _collect_code_symbols(path: Path) -> CodeSymbols:
    """
    函数功能与逻辑描述：
        用 ast 解析单个 .py 文件并抽取可核对契约：顶层函数/类/赋值常量、类的方法与类属性
        （含 dataclass 字段、`__slots__` 声明的字段）、类内嵌套类的方法，以及函数内嵌套函数；
        同时登记各函数的参数名。解析前临时屏蔽 SyntaxWarning（历史遗留无效转义与核对无关），
        避免污染输出。
    入参说明：
        path (Path)：待解析的 Python 文件路径，须存在且可读。
    返回值说明：
        CodeSymbols：该文件的符号集；顶层为空时返回空 CodeSymbols 实例。
        异常：源码语法错误会原样抛 SyntaxError，文件读取失败会抛 OSError，均由调用方
            （load_symbols）捕获后降级为 WARN，不在本函数内吞掉。
    """
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
    """
    函数功能与逻辑描述：
        ★v3.0：递归登记嵌套 def（闭包/内部函数）为父函数的成员，并同时登记其参数，
        供 wide() 与检查 C 核对。这类函数虽非顶层，但文档会把它写进接口表（是真实存在的
        可核对对象），不登记会产生假 FAIL。典型场景：`parse_json_output` 内部定义的
        `try_extract`。逐层用 f"{parent}.{子函数名}" 拼接限定名，与 _index_params 的口径一致。
    入参说明：
        syms (CodeSymbols)：待写入的符号集容器，就地修改（副作用）。
        node (ast.AST)：父函数节点，walk 时跳过自身、只取函数定义。
        parent (str)：父函数的限定名前缀，用于生成 "父.子" 形式的成员与参数键。
    返回值说明：
        无（仅修改 syms.members 与 syms.func_params 的副作用）。
    """
    for sub in ast.walk(node):
        if sub is node or not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        syms.members.setdefault(parent, set()).add(sub.name)
        _index_params(syms.func_params, sub, f"{parent}.{sub.name}")


def _index_params(store: dict[str, set[str]], node: ast.FunctionDef | ast.AsyncFunctionDef,
                  prefix: str = "") -> None:
    """
    函数功能与逻辑描述：
        从一个 ast 函数节点抽取其全部形参名（位置限定参数、普通位置参数、仅关键字参数，
        以及 *args / **kwargs 的名字），以 `prefix + 函数名` 为键写入 store，供检查 C 核对
        文档签名中的参数名是否真实存在。同名键会被整体覆盖（不做并集），故同一符号重复登记
        以最后一次为准。
    入参说明：
        store (dict[str, set[str]])：参数名登记表，就地写入（副作用）。
        node (ast.FunctionDef | ast.AsyncFunctionDef)：待抽取的函数节点。
        prefix (str)：限定名前缀，类方法传 "类名."、嵌套函数传 "父函数."，顶层函数留空。
    返回值说明：
        无（仅修改 store 的副作用）。
    """
    args = node.args
    names = [a.arg for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    store[prefix + node.name] = {n for n in names if n}


@dataclass
class Issue:
    """
    函数/类功能与逻辑描述：
        一条文档-代码漂移问题记录，统一承载检查 A/B/C 的发现，由 _summarize 按级别分组打印；
        本身无校验逻辑，仅作数据载体，failure 是否阻塞由 level 决定（FAIL 阻塞、WARN 仅提示）。
    构造入参说明：
        level (str)：严重级别，取值 "FAIL" 或 "WARN"。
        kind (str)：问题类别，取值 "symbol"（符号缺失）/ "line_ref"（行号越界）/
            "param"（参数名不符）/ "file_ref"（文件缺失）/ "dir_ref"（目录缺失）/ "parse"（代码解析失败）。
        where (str)：文档中的行号或位置描述；无文档位置时为空串。
        file (str)：涉及的代码文件（相对路径，尽量给出）；无归属时为空串。
        symbol (str)：涉及的符号名；无单一符号时为空串。
        msg (str)：面向人的中文说明。
    返回值说明：
        构造返回 Issue 实例。
    """
    level: str            # FAIL / WARN
    kind: str             # symbol / line_ref / param / file_ref / dir_ref / parse
    where: str            # 文档行号或位置
    file: str             # 代码文件（尽量）
    symbol: str
    msg: str


def _clean_tokens(cell: str) -> list[str]:
    """
    函数功能与逻辑描述：
        从接口列表格单元格中提取"纯符号候选"，作为检查 A 的待核对清单。清洗规则：
        反引号内容优先（用户特意标记的代码符号），无反引号时按 `/`、`、` 拆分；
        逐 token 剥掉 `async def`/`def`/`class` 前缀、`-> 返回类型`、尾随参数括号 `name(...)`；
        再丢弃 .py 文件路径与 `L41`/`L294-311` 形式的出处行号（非符号）；
        最终只保留 `name` 或 `Class.member` 形态且不在 STOP_WORDS 中的项，
        其余（常量组大括号、类型词、中文等）一律丢弃，避免把参数名/描述词误当接口符号。
    入参说明：
        cell (str)：接口列表格的单元格原文，可含反引号、多个用 `/` 或 `、` 分隔的符号。
    返回值说明：
        list[str]：清洗后剩余的符号候选（保持出现顺序）；无可识别符号时返回空列表。
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
    """
    函数功能与逻辑描述：
        判断文档中出现的符号名在代码侧是否真实存在，是「符号存在性」检查的核心判据。
        三段判定：① 含点号的符号先按 `类.成员` 拆分，若左侧是已知类则在该类的成员集合里查，
        否则退化为「模块.函数」语义、在整个宽集里查成员名；② 不含点号的符号在宽集
        （成员 + 顶层）里查，或退而查顶层集合。
        宽集匹配意味着**跨文件同名即视为存在**，这是有意为之的宽匹配策略：
        文档常以目录/模块并集描述接口，若只做精确归属会产出大量假 FAIL。
    入参说明：
        syms (CodeSymbols)：代码符号集合（含 top_level / classes / members 及 wide() 宽集）。
        sym (str)：待核对的符号名，形如 "func" 或 "ClassName.member"。
    返回值说明：
        bool：True 表示符号在代码侧可定位；False 表示未找到（调用方据此报 FAIL）。
    """
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
    """
    函数功能与逻辑描述：
        把一段文档文本里的每个 `def name(args)` 片段单独解析出来，返回
        {函数名: 参数名集合}。存在意义是让参数核对**精确到函数**：接口列常把多个函数写在
        同一格（"def a(x: int)、def b(y: str)"），若整格提参数会把 a 的参数算到 b 头上，
        产生成片噪音 WARN。正则只匹配一层括号内的内容，因此签名中出现嵌套括号
        （如默认值为元组/嵌套调用）时该片段会被跳过而非误解析。
    入参说明：
        text (str)：待解析的文档片段（通常是表格单元格或整行文本）。
    返回值说明：
        dict[str, set[str]]：函数名 → 该函数文档中声明的参数名集合；
            文本中无 `def` 片段时返回空字典；同名函数多次出现时后者覆盖前者。
    """
    return {m.group(1): _extract_sig_params("(" + m.group(2))
            for m in _DEF_SEG_RE.finditer(text)}


def _extract_sig_params(sig: str) -> set[str]:
    """
    函数功能与逻辑描述：
        从文档中的函数签名字符串里启发式提取参数名：用正则查找紧跟 `:` 或 `=` 的标识符
        （即带类型标注或默认值的形参），再过滤停用词与 self/cls。
        之所以是「启发式」：文档签名格式不完全统一，本函数只服务于 WARN 级提示，
        其结论**不允许**升级为 FAIL，避免因文档书写风格差异误判。
    入参说明：
        sig (str)：函数签名文本（通常含括号与参数列表）。
    返回值说明：
        set[str]：提取到的参数名集合（已剔除 STOP_WORDS 与 self / cls）；无匹配时返回空集合。
    """
    params = set(re.findall(r"[\s,(]([A-Za-z_]\w*)\s*(?=[:=])", sig))
    return {p for p in params if p not in STOP_WORDS and p not in ("self", "cls")}


def _method_hints(text: str) -> set[str]:
    """
    函数功能与逻辑描述：
        用 _METHOD_HINT_RE 从文档文本中找出「方法调用/方法名」线索，并过滤停用词，
        用于在接口表未给出明确签名时辅助推断涉及的符号名。
    入参说明：
        text (str)：待扫描的文档文本。
    返回值说明：
        set[str]：命中的方法名集合（已剔除 STOP_WORDS）；无命中时返回空集合。
    """
    out = set(_METHOD_HINT_RE.findall(text))
    return {t for t in out if t not in STOP_WORDS}


def _check_params(syms: CodeSymbols, sym: str, doc_params: set[str]) -> list[str]:
    """
    函数功能与逻辑描述：
        执行「检查 C」：核对文档声明的参数名与代码实际形参是否一致，返回**文档侧多出**的参数名
        （即文档里有、代码里没有的，通常意味着文档过期或参数改名）。
        采用「只报多不报少」的单向判定：代码新增参数不一定需要文档同步，故不报缺项。
        当符号未在代码侧定位到（func_params 无该键）时放弃核对并返回空列表，
        避免把「符号本身不存在」的问题重复报成参数不一致。
    入参说明：
        syms (CodeSymbols)：代码符号集合，需含 func_params（符号 → 形参集合）。
        sym (str)：待核对的符号名。
        doc_params (set[str])：文档声明的参数名集合。
    返回值说明：
        list[str]：文档中多出的参数名，按字母序排序；完全一致或无入参可核对时返回 []。
    """
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
    """
    函数功能与逻辑描述：
        契约核对主流程：逐行扫描设计文档，把文档中引用的代码文件/目录/符号/行号与仓库真实代码比对，
        产出结构化问题列表。核心机制三点：
        ① 作用域绑定（v3.0 起不再依赖 §5.6 编号定位）——遇到含路径的行就把 cur_binding
        更新为对应文件或目录（目录展开为其下全部 .py），后续符号与行号检查都在该作用域内进行；
        ② 符号去重——同一 (文件, 符号) 只报一次，避免同文件重复登记产生刷屏；
        ③ 分级——结构性问题（文件不存在、行号越界）记 FAIL，风格性问题（签名参数名不一致）记 WARN，
        只有 FAIL 才会让退出码非 0（见 _summarize）。
        两个嵌套辅助函数 update_binding / load_symbols 分别负责作用域更新与符号合并（含文件级缓存）。
    入参说明：
        doc_path (Path | None)：要核对的文档路径；默认 None 表示用 DEFAULT_DOC
            （设计_从零系列/09_模块契约手册.md）。
        repo_root (Path | None)：仓库根目录；默认 None 表示用 REPO_ROOT。仅影响目录型引用的展开。
        check_params (bool)：是否开启「检查 C」签名参数名核对，默认 False
            （默认关闭因其结论仅为 WARN 级启发式结果）。
    返回值说明：
        tuple[list[Issue], dict]：二元组 (问题列表, 统计信息)。问题列表含全部 FAIL 与 WARN；
            统计信息字典固定含 symbols_checked（已核对接口符号数）与 line_refs（已核对行号引用数）两项。
    """
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
        """
        函数功能与逻辑描述：
            更新当前检查作用域（闭包内 nonlocal cur_binding）：先清空旧绑定，再逐个处理路径 token。
            三类 token 分别处理：① 含 `*{}?` 的 glob 写法（如 agents/{a,b}_agent/）不是真实路径，
            直接跳过；② `.py` 结尾的文件走 locate_file 定位，定位失败且状态为 missing 时记 FAIL；
            ③ `/` 结尾的目录按相对路径展开为其下全部 .py（并剔除 SKIP_DIRS），目录不存在则记 FAIL。
            未识别的 token（既非文件也非目录）被静默忽略，不做报错。
        入参说明：
            path_tokens (list[str])：从文档行中切分出的路径候选 token 列表。
            lineno (int)：产生这些 token 的文档行号（1 基），用于问题定位。
        返回值说明：
            无（副作用：改写闭包变量 cur_binding，并可能向 issues 追加 FAIL 条目）。
        """
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
        """
        函数功能与逻辑描述：
            把作用域内多个代码文件的符号合并成一个 CodeSymbols（目录并集语义）：
            top_level 与 classes 取并集，members 按类名做 set 合并，func_params 直接 update。
            逐文件结果缓存在闭包字典 cache 中（含 None），因此同一文件在整篇文档中只解析一次 AST。
            解析失败（SyntaxError / OSError）时记一条 WARN 并把该文件缓存为 None，
            从而跳过其符号、不产生连锁假 FAIL。
        入参说明：
            paths (list[Path])：作用域内的代码文件路径列表。
        返回值说明：
            CodeSymbols | None：合并后的符号集合；paths 为空或全部文件解析失败时
                返回一个**空的** CodeSymbols（注意：函数签名虽允许 None，但当前实现恒不返回 None）。
        """
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
    """
    函数功能与逻辑描述：
        打印核对报告并返回**进程退出码**，是 CLI 的裁决点：先按 FAIL / WARN 两级分组输出，
        再打印覆盖率统计与小结。退出码规则为「有 FAIL 时返回 1，否则返回 0」——
        即只有结构性漂移（文件缺失、行号越界）会阻塞，WARN 级问题不阻塞。
        无问题时直接打印通过提示并返回 0。
        输出量与 full 相关：full=False 时 WARN 区段只打印条数摘要、不逐条展开，
        避免大量风格性提示淹没真正要紧的 FAIL；full=True 时全量打印。
    入参说明：
        issues (list[Issue])：check_all 产出的全部问题条目。
        full (bool)：是否打印全部 WARN 详情（对应 CLI 的 --full）。
        stats (dict | None)：覆盖率统计（symbols_checked / line_refs）；默认 None 表示不打印统计行。
    返回值说明：
        int：进程退出码。0 表示无 FAIL（通过）；1 表示存在 FAIL（须处理）。
    """
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
    """
    函数功能与逻辑描述：
        CLI 入口：解析三个命令行参数后调用 check_all 执行核对，再把结果交给 _summarize 输出报告。
        参数为 --doc（被核对文档，默认 DEFAULT_DOC）、--check-params（开启签名参数名核对，WARN 级）、
        --full（打印全部 WARN 与详情）。argv 不传时 argparse 自动读取 sys.argv[1:]，
        因此既支持 `python -m utils.contract_check` 也支持测试中直接注入参数列表调用。
    入参说明：
        argv (list[str] | None)：命令行参数列表（不含程序名）；默认 None 表示取 sys.argv[1:]。
    返回值说明：
        int：进程退出码，直接透传 _summarize 的结果（0 = 无 FAIL，1 = 存在 FAIL）。
    """
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
