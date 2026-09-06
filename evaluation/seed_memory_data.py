# -*- coding: utf-8 -*-
"""
写入模拟月度消费摘要到长期记忆，供 memory_recall_eval.py 评测使用。

为什么需要这个脚本：长期记忆是"运行时逐步积累"的，当前文件里 0 条数据。
评测召回率的前提是先有数据，本脚本用特征分明的模拟摘要把写入链路跑通并落盘。

用法：
    python evaluation/seed_memory_data.py            # 数据已存在则跳过
    python evaluation/seed_memory_data.py --force   # 清空后重新写入
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ===================== 模拟月度消费摘要 =====================
# 格式严格对齐 agents/orchestrator/nodes.py 的 _build_monthly_summary 真实输出，
# 只有 5 个固定字段，不允许额外加戏：
#   【{time_start}~{time_end} 消费摘要】; 总支出: {total}元; 分类: {cat}: {v}元, ...; 共{N}笔; 单笔最大: {amount}元({category})
# 各条区分度仅靠真实字段的值（月份/金额/分类构成），与真实运行时形态一致。
# 注意：这些原文就是评测的"期望记忆"，修改后需同步修改 memory_recall_eval.py 的期望引用
MOCK_SUMMARIES = [
    # 2026-01：总支出5000，餐饮800/交通200/房租3000/购物500/娱乐500，45笔，单笔最大600(娱乐)
    "【2026-01-01~2026-01-31 消费摘要】; 总支出: 5000元; 分类: 餐饮: 800元, 交通: 200元, 房租: 3000元, 购物: 500元, 娱乐: 500元; 共45笔; 单笔最大: 600元(娱乐)",
    # 2026-02：总支出6200，餐饮1500(最高)/交通150/房租3000/购物800/娱乐750，52笔，单笔最大900(购物)
    "【2026-02-01~2026-02-28 消费摘要】; 总支出: 6200元; 分类: 餐饮: 1500元, 交通: 150元, 房租: 3000元, 购物: 800元, 娱乐: 750元; 共52笔; 单笔最大: 900元(购物)",
    # 2026-03：总支出3600，餐饮650/交通90(最低)/房租3000/购物200/娱乐100，38笔，单笔最大300(餐饮)
    "【2026-03-01~2026-03-31 消费摘要】; 总支出: 3600元; 分类: 餐饮: 650元, 交通: 90元, 房租: 3000元, 购物: 200元, 娱乐: 100元; 共38笔; 单笔最大: 300元(餐饮)",
    # 2026-04：总支出4300，餐饮900/交通180/房租3000/购物150/娱乐70，41笔，单笔最大3000(房租)
    "【2026-04-01~2026-04-30 消费摘要】; 总支出: 4300元; 分类: 餐饮: 900元, 交通: 180元, 房租: 3000元, 购物: 150元, 娱乐: 70元; 共41笔; 单笔最大: 3000元(房租)",
    # 2025-12：总支出8800，餐饮1200/交通260/房租3000/购物3200(最高)/娱乐1100，60笔，单笔最大1500(购物)
    "【2025-12-01~2025-12-31 消费摘要】; 总支出: 8800元; 分类: 餐饮: 1200元, 交通: 260元, 房租: 3000元, 购物: 3200元, 娱乐: 1100元; 共60笔; 单笔最大: 1500元(购物)",
    # 2026-05：总支出2800(最低)，餐饮600/交通120/房租1500/购物380/娱乐200，30笔，单笔最大500(餐饮)
    "【2026-05-01~2026-05-31 消费摘要】; 总支出: 2800元; 分类: 餐饮: 600元, 交通: 120元, 房租: 1500元, 购物: 380元, 娱乐: 200元; 共30笔; 单笔最大: 500元(餐饮)",
    # 2026-06：总支出5100，餐饮1000/交通300(最高)/房租3000/购物600/娱乐200，48笔，单笔最大800(交通)
    "【2026-06-01~2026-06-30 消费摘要】; 总支出: 5100元; 分类: 餐饮: 1000元, 交通: 300元, 房租: 3000元, 购物: 600元, 娱乐: 200元; 共48笔; 单笔最大: 800元(交通)",
    # 2026-07：总支出4700，餐饮950/交通150/房租3000/购物400/娱乐200，44笔，单笔最大700(娱乐)
    "【2026-07-01~2026-07-31 消费摘要】; 总支出: 4700元; 分类: 餐饮: 950元, 交通: 150元, 房租: 3000元, 购物: 400元, 娱乐: 200元; 共44笔; 单笔最大: 700元(娱乐)",
]


def seed(force: bool = False):
    from memory.long_memory import LONG_MEM_PATH, clear_all_long_mem, save_consume_memory

    if os.path.exists(LONG_MEM_PATH) and not force:
        # 已写入过则校验条数
        with open(LONG_MEM_PATH, "rb") as f:
            import pickle
            existing = pickle.load(f)
        if len(existing) >= len(MOCK_SUMMARIES):
            print(f"长期记忆已有 {len(existing)} 条，跳过写入（用 --force 可清空重写）")
            return
        print(f"已有 {len(existing)} 条，少于目标 {len(MOCK_SUMMARIES)} 条，继续补写...")

    if force:
        clear_all_long_mem()
        print("已清空全部长期记忆")

    for i, s in enumerate(MOCK_SUMMARIES, 1):
        save_consume_memory(s)
        print(f"[{i}/{len(MOCK_SUMMARIES)}] 已写入: {s[:30]}...")

    with open(LONG_MEM_PATH, "rb") as f:
        import pickle
        total = pickle.load(f)
    print(f"\n写入完成，持久化文件: {LONG_MEM_PATH}")
    print(f"总条数: {len(total)} | 文件大小: {os.path.getsize(LONG_MEM_PATH)} 字节")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="写入模拟月度消费摘要到长期记忆")
    parser.add_argument("--force", action="store_true", help="清空后重新写入")
    args = parser.parse_args()
    seed(args.force)
