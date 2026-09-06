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
    """构造一个最小仓库：mini.py + 一份能触发编号小节解析的文档。"""
    repo = tmp_path
    (repo / "mini.py").write_text(mini_py, encoding="utf-8")
    doc = repo / "设计_从零系列" / "09_模块契约手册.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(body, encoding="utf-8")
    return repo, doc


# ---------------------------------------------------------------- 真实仓库漂移检测

def test_repo_contract_no_drift():
    """从零系列 09 的接口契约与代码无 FAIL 级漂移（符号缺失 / 文件缺失 / 行号超界）。"""
    issues, stats = cc.check_all()
    fails = [i for i in issues if i.level == "FAIL"]
    assert stats["symbols_checked"] >= 100, "解析覆盖率异常低，脚本可能在空转"
    assert fails == [], "\n".join(f"{i.file} {i.symbol}: {i.msg}" for i in fails)


# ---------------------------------------------------------------- 负例：能抓到漂移

def test_detects_missing_symbol(tmp_path, monkeypatch):
    """文档接口表写了个代码里不存在的符号 → 必须 FAIL。"""
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
    """文档里的行号引用超出文件实际行数 → 必须 FAIL（行号漂移）。"""
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
    syms = CodeSymbols(top_level={"foo", "CONST", "A"},
                      classes={"A"},
                      members={"A": {"bar", "baz"}})
    assert cc._symbol_exists(syms, "foo")
    assert cc._symbol_exists(syms, "CONST")
    assert cc._symbol_exists(syms, "A")
    assert cc._symbol_exists(syms, "A.bar")
    assert not cc._symbol_exists(syms, "A.missing")
    assert not cc._symbol_exists(syms, "gone")
