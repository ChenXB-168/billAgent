# -*- coding: utf-8 -*-
"""契约自动核对接入 L1。

职责：
1. 真实仓库：`设计_从零系列/09_模块契约手册.md` 的接口符号 / 行号引用与代码核对，
   出现 FAIL（文档写了但代码已删/改名、文件缺失、行号超界）即测试红——
   让"文档 ↔ 代码漂移"不再靠人工 review 才被发现。
2. 合成负例：确保核对脚本自身真能抓漂移（防"空转"）。

注：本项目 tests/conftest.py 有 session 级 autouse fixture（拉起 LLM MCP server），
本测试不依赖它，但会复用其进程生命周期，开销已在同层其余测试中发生。
"""

import pytest

from utils import contract_check as cc
from utils.contract_check import CodeSymbols


def _doc_for(tmp_path, body: str, mini_py: str = "") -> tuple:
    """
    函数功能与逻辑描述：
        构造一个最小可核对仓库，供负例用例注入：在 tmp_path 下写入 mini.py 源码，
        并生成一份位于 "设计_从零系列/09_模块契约手册.md" 的文档（正文含编号小节标题，
        用于触发 contract_check 的作用域绑定与接口表解析）。
    入参说明：
        tmp_path：pytest 内置 fixture 提供的临时目录，充当合成仓库根。
        body (str)：契约手册正文（含编号标题行、接口表、可选行号引用）。
        mini_py (str)：mini.py 的源码内容，默认空串（即仓库中不存在任何可核对 Python 符号）。
    返回值说明：
        tuple：二元组 (repo, doc)，repo 为仓库根 Path，doc 为生成的契约手册 Path。
    """
    repo = tmp_path
    (repo / "mini.py").write_text(mini_py, encoding="utf-8")
    doc = repo / "设计_从零系列" / "09_模块契约手册.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(body, encoding="utf-8")
    return repo, doc


# ---------------------------------------------------------------- 真实仓库漂移检测

def test_repo_contract_no_drift():
    """
    函数功能与逻辑描述：
        对真实仓库做全量契约核对：调用 cc.check_all() 扫描默认文档
        （设计_从零系列/09_模块契约手册.md），断言两件事——(1) stats["symbols_checked"] >= 100，
        即解析覆盖率不得异常偏低（防脚本"空转"，覆盖率过低会掩盖真实漂移）；
        (2) FAIL 级问题为空（FAIL = 符号缺失 / 文件缺失 / 行号超界；WARN 级不阻塞）。
        覆盖场景：文档 ↔ 代码漂移的真实回归门禁。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    issues, stats = cc.check_all()
    fails = [i for i in issues if i.level == "FAIL"]
    assert stats["symbols_checked"] >= 100, "解析覆盖率异常低，脚本可能在空转"
    assert fails == [], "\n".join(f"{i.file} {i.symbol}: {i.msg}" for i in fails)


# ---------------------------------------------------------------- 负例：能抓到漂移

def test_detects_missing_symbol(tmp_path, monkeypatch):
    """
    函数功能与逻辑描述：
        合成负例（确保核对脚本真能抓到漂移）：构造只含 foo / A.bar 的 mini.py 与一份接口表，
        表中同时列入真实存在的 `foo` 与不存在的 `missing_func`；用例把 cc.REPO_ROOT
        monkeypatch 为合成仓库（使文件索引只覆盖该临时仓库），再以显式 doc_path / repo_root
        调用 cc.check_all，断言产出 kind="symbol" 且 level="FAIL" 的 missing_func 记录，
        同时不得把存在的 `foo` 误报为 FAIL。
    入参说明：
        tmp_path：pytest 内置 fixture 提供的临时目录，充当合成仓库根。
        monkeypatch：pytest 内置 fixture，用于把 cc.REPO_ROOT 替换为合成仓库，用例结束自动回滚。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mini = "def foo():\n    pass\n\nclass A:\n    def bar(self):\n        pass\n"
    doc_body = (
        "#### 5.6.1 测试模块\n"
        "**5.6.1.1 `mini.py`**\n"
        "\n"
        "| 接口 | 签名 | 职责 | 连接 |\n"
        "|---|---|---|---|\n"
        "| `foo` | `()` | 存在 | x |\n"
        "| `missing_func` | `()` | 已删 | x |\n"
        "\n"
        "## 6. 模块边界\n"
    )
    repo, doc = _doc_for(tmp_path, doc_body, mini)
    monkeypatch.setattr(cc, "REPO_ROOT", repo)
    issues, _ = cc.check_all(doc_path=doc, repo_root=repo)
    sym_fails = [i for i in issues if i.kind == "symbol" and i.level == "FAIL"]
    assert any(i.symbol == "missing_func" for i in sym_fails), "应抓到缺失符号 missing_func"
    assert not any(i.symbol == "foo" for i in sym_fails), "不应误报存在的 foo"


def test_detects_stale_line_ref(tmp_path, monkeypatch):
    """
    函数功能与逻辑描述：
        合成负例（行号漂移）：构造只含 "x = 1" 的 mini.py，并在文档最前面写入一行
        "`mini.py` L999 已漂移"（检查 B 扫全文档，故行号引用放在任意位置都会触发）；
        再把 cc.REPO_ROOT monkeypatch 为合成仓库，断言 cc.check_all 产出
        kind="line_ref" 且 level="FAIL" 的记录。
        覆盖场景：文档行号引用超出文件实际行数的检测。
    入参说明：
        tmp_path：pytest 内置 fixture 提供的临时目录，充当合成仓库根。
        monkeypatch：pytest 内置 fixture，用于把 cc.REPO_ROOT 替换为合成仓库，用例结束自动回滚。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mini = "x = 1\n"
    doc_body = (
        "#### 5.6.1 测试模块\n"
        "\n"
        "## 6. 模块边界\n"
    )
    # B 检查扫全文档：把超界行号引用放在最前面即可触发
    doc = (tmp_path / "设计_从零系列" / "09_模块契约手册.md")
    doc.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "mini.py").write_text(mini, encoding="utf-8")
    doc.write_text("`mini.py` L999 已漂移\n" + doc_body, encoding="utf-8")
    monkeypatch.setattr(cc, "REPO_ROOT", tmp_path)
    issues, _ = cc.check_all(doc_path=doc, repo_root=tmp_path)
    lr = [i for i in issues if i.kind == "line_ref" and i.level == "FAIL"]
    assert lr, "应抓到行号超界引用"


# ---------------------------------------------------------------- 解析器核心单元

def test_clean_tokens_strips_call_parentheses():
    """
    函数功能与逻辑描述：
        验证 cc._clean_tokens（检查 A 的待核对符号提取器）的清洗规则：反引号内容优先并剥掉
        尾随参数括号（`calc_premium_evaluate(...)` → calc_premium_evaluate）、一格多符号按
        "/" 或 "、" 拆分（`pre_rule_check(text)` / `rule_parse(text)` → 两个符号）、
        点号形式 `ToolRegistry.register` 整体保留；而无反引号的签名式文本同样按分隔符拆分并剥括号；
        大括号常量组 `PREMIUM_THRESHOLD_{LOW,NORMAL,HIGH}` 不是符号，必须被丢弃（返回 []）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert cc._clean_tokens("`calc_premium_evaluate(...)`") == ["calc_premium_evaluate"]
    assert cc._clean_tokens("`pre_rule_check(text)` / `rule_parse(text)`") == [
        "pre_rule_check", "rule_parse",
    ]
    assert cc._clean_tokens("`ToolRegistry.register`") == ["ToolRegistry.register"]
    # 无反引号的接口格：按分隔符拆 + 剥括号
    assert cc._clean_tokens("add_msg(role) / get_history(limit=3)") == ["add_msg", "get_history"]
    # 大括号常量组 / 中文说明不是符号，必须丢弃
    assert cc._clean_tokens("`PREMIUM_THRESHOLD_{LOW,NORMAL,HIGH}`") == []


def test_symbol_exists_semantics():
    """
    函数功能与逻辑描述：
        验证 cc._symbol_exists 的存在性判定语义：顶层函数 / 常量 / 类名（foo、CONST、A）
        均判定存在；类成员以 "类.成员" 形式给出（A.bar）判定存在，而 A.missing 与完全未知的
        gone 判定为不存在——覆盖 top_level / classes / members 三类符号与"宽集匹配"策略。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    syms = CodeSymbols(top_level={"foo", "CONST", "A"},
                      classes={"A"},
                      members={"A": {"bar", "baz"}})
    assert cc._symbol_exists(syms, "foo")
    assert cc._symbol_exists(syms, "CONST")
    assert cc._symbol_exists(syms, "A")
    assert cc._symbol_exists(syms, "A.bar")
    assert not cc._symbol_exists(syms, "A.missing")
    assert not cc._symbol_exists(syms, "gone")
