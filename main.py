# -*- coding: utf-8 -*-
"""迷你知识库问答：字符 2-gram TF-IDF 检索 + 阈值拒答（纯 Python 标准库）。

用法：
    python main.py --eval                离线评测：阈值网格搜索 + 全部指标 + badcase
    python main.py --ask "怎么报名训练营"   单条提问：输出 Top-3 检索结果或拒答

算法：
    1. 把每条知识的 question 和 answer 拼成文档；
    2. 文本转小写、去标点后切成字符 2-gram；
    3. 用 log((N+1)/(df+1))+1 计算 idf，tf*idf 后做 L2 归一化，得到文档向量；
    4. 评测问题同样向量化（词表外 bigram 直接丢弃），用余弦相似度排序；
    5. 取 Top-3，若 Top-1 相似度低于拒答阈值，则回答「没有找到相关答案」。
"""

import sys

# Windows 控制台默认可能是 GBK，直接输出中文会报 UnicodeEncodeError。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import csv
import math
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "data", "kb.csv")
EVAL_PATH = os.path.join(BASE_DIR, "data", "eval_questions.csv")

NONE = "NONE"                 # 库外问题在评测集里的 expected_id 标记
TOP_K = 3
THRESHOLD_MIN = 0.15
THRESHOLD_MAX = 0.45
THRESHOLD_STEP = 0.01
SCORE_DIGITS = 6              # 综合分比较时保留的小数位，避免浮点误差
REFUSAL_TEXT = "没有找到相关答案。"


# ---------------------------------------------------------------- 数据读取

def _read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_kb(path=KB_PATH):
    rows = []
    for row in _read_csv(path):
        rows.append({
            "id": str(row["id"]).strip(),
            "question": str(row["question"]).strip(),
            "answer": str(row["answer"]).strip(),
        })
    if not rows:
        raise SystemExit("知识库为空：%s" % path)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("知识库 id 有重复：%s" % path)
    return rows


def load_eval(path=EVAL_PATH):
    rows = []
    for row in _read_csv(path):
        rows.append({
            "question": str(row["question"]).strip(),
            "expected_id": str(row["expected_id"]).strip(),
        })
    if not rows:
        raise SystemExit("评测集为空：%s" % path)
    return rows


# ---------------------------------------------------------------- 检索模型

def char_bigrams(text):
    """转小写、只保留文字/数字字符后，切相邻两字符的 2-gram。"""
    chars = [ch for ch in text.lower() if ch.isalnum()]
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def _to_unit_vector(tokens, idf):
    """tf * idf 后做 L2 归一化；不是词表里的 bigram 直接被丢弃。"""
    term_freq = Counter(tokens)
    weights = {}
    for term, count in term_freq.items():
        base = idf.get(term)
        if base is not None:
            weights[term] = count * base
    norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    if norm == 0.0:
        return {}
    return {term: weight / norm for term, weight in weights.items()}


def build_index(kb):
    """在知识库上统计 df / idf，并预先算好每条知识的单位向量。"""
    tokenized_docs = [char_bigrams(item["question"] + item["answer"]) for item in kb]
    n_docs = len(kb)
    df = Counter()
    for tokens in tokenized_docs:
        df.update(set(tokens))
    idf = {
        term: math.log((n_docs + 1) / (count + 1)) + 1.0
        for term, count in df.items()
    }
    doc_vectors = [_to_unit_vector(tokens, idf) for tokens in tokenized_docs]
    return {
        "kb": kb,
        "idf": idf,
        "doc_vectors": doc_vectors,
        "vocab_size": len(idf),
        "avg_doc_len": sum(len(tokens) for tokens in tokenized_docs) / n_docs,
    }


def cosine(vec_a, vec_b):
    """两个已归一化向量的余弦相似度，等于点积。"""
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(weight * vec_b.get(term, 0.0) for term, weight in vec_a.items())


def search(index, query, k=TOP_K):
    """返回 [(相似度, 知识库下标), ...]，按相似度从高到低排。"""
    query_vec = _to_unit_vector(char_bigrams(query), index["idf"])
    scored = [(cosine(query_vec, doc_vec), i) for i, doc_vec in enumerate(index["doc_vectors"])]
    scored.sort(key=lambda pair: (-pair[0], index["kb"][pair[1]]["id"]))
    return scored[:k]


# ---------------------------------------------------------------- 评测

def _threshold_candidates():
    steps = int(round((THRESHOLD_MAX - THRESHOLD_MIN) / THRESHOLD_STEP))
    return [round(THRESHOLD_MIN + i * THRESHOLD_STEP, 2) for i in range(steps + 1)]


def analyze(index, eval_rows):
    """跑完整个评测，返回所有指标、阈值搜索结果和 badcase。"""
    in_rows = [row for row in eval_rows if row["expected_id"] != NONE]
    out_rows = [row for row in eval_rows if row["expected_id"] == NONE]
    n_in, n_out = len(in_rows), len(out_rows)

    results = []
    for row in eval_rows:
        results.append({"row": row, "ranked": search(index, row["question"])})

    sweep = []
    for threshold in _threshold_candidates():
        hit1 = hit3 = rejected = 0
        for item in results:
            expected = item["row"]["expected_id"]
            top1_sim = item["ranked"][0][0]
            if expected == NONE:
                if top1_sim < threshold:
                    rejected += 1
                continue
            if top1_sim < threshold:      # 库内题被拒答，算未命中
                continue
            top1_id = index["kb"][item["ranked"][0][1]]["id"]
            if top1_id == expected:
                hit1 += 1
            if any(index["kb"][idx]["id"] == expected for _, idx in item["ranked"]):
                hit3 += 1
        score = hit1 / n_in + rejected / n_out
        sweep.append({
            "threshold": threshold,
            "hit1": hit1,
            "hit3": hit3,
            "rejected": rejected,
            "score": score,
        })

    # 选择标准：1) 综合分（Top-1 命中率 + 拒答正确率）最高；
    #          2) 同分时拒答正确率更高的更安全；
    #          3) 仍同分时取较小阈值（对库内问题更友好，避免过度拒答）。
    best = max(
        sweep,
        key=lambda item: (
            round(item["score"], SCORE_DIGITS),
            item["rejected"],
            -item["threshold"],
        ),
    )
    threshold = best["threshold"]

    # ---- 阈值下的逐题判定 & badcase 收集
    badcases = []
    in_rejected = 0
    for item in results:
        expected = item["row"]["expected_id"]
        top1_sim, top1_idx = item["ranked"][0]
        top1 = index["kb"][top1_idx]
        correct_rank = None
        if expected != NONE:
            for rank, (_, idx) in enumerate(item["ranked"], start=1):
                if index["kb"][idx]["id"] == expected:
                    correct_rank = rank
                    break
        if expected == NONE:
            if top1_sim >= threshold:
                badcases.append({
                    "type": "库外误答",
                    "question": item["row"]["question"],
                    "expected": NONE,
                    "top1": top1,
                    "sim": top1_sim,
                    "correct_rank": None,
                })
        else:
            if top1_sim < threshold:
                in_rejected += 1
                badcases.append({
                    "type": "库内被拒答",
                    "question": item["row"]["question"],
                    "expected": expected,
                    "top1": top1,
                    "sim": top1_sim,
                    "correct_rank": correct_rank,
                })
            elif top1["id"] != expected:
                badcases.append({
                    "type": "库内Top-1错误",
                    "question": item["row"]["question"],
                    "expected": expected,
                    "top1": top1,
                    "sim": top1_sim,
                    "correct_rank": correct_rank,
                })

    # ---- baseline：不设拒答，永远返回 Top-1
    base_top1 = sum(
        1 for item in results
        if item["row"]["expected_id"] != NONE
        and index["kb"][item["ranked"][0][1]]["id"] == item["row"]["expected_id"]
    )
    base_top3 = sum(
        1 for item in results
        if item["row"]["expected_id"] != NONE
        and any(index["kb"][idx]["id"] == item["row"]["expected_id"] for _, idx in item["ranked"])
    )

    return {
        "n_in": n_in,
        "n_out": n_out,
        "results": results,
        "sweep": sweep,
        "best": best,
        "threshold": threshold,
        "hit1": best["hit1"] / n_in,
        "hit3": best["hit3"] / n_in,
        "reject_acc": best["rejected"] / n_out,
        "wrong_rate": 1.0 - best["rejected"] / n_out,
        "in_rejected": in_rejected,
        "badcases": badcases,
        "baseline": {
            "hit1": base_top1 / n_in,
            "hit3": base_top3 / n_in,
            "wrong_rate": 1.0,
        },
    }


# ---------------------------------------------------------------- 输出

def _print_header(index, eval_rows):
    print("=" * 62)
    print("迷你知识库问答 · 离线评测")
    print("=" * 62)
    print("数据规模   ：知识库 %d 条 / 评测问题 %d 条（库内 %d + 库外 %d）" % (
        len(index["kb"]),
        len(eval_rows),
        sum(1 for row in eval_rows if row["expected_id"] != NONE),
        sum(1 for row in eval_rows if row["expected_id"] == NONE),
    ))
    print("检索方法   ：字符 2-gram TF-IDF + 余弦相似度，取 Top-%d" % TOP_K)
    print("词表规模   ：%d 个 bigram，文档平均长度 %.1f 个 bigram" % (
        index["vocab_size"], index["avg_doc_len"]))
    print("idf 公式   ：log((N+1)/(df+1)) + 1（N = 知识库条数）")
    print("阈值网格   ：%.2f ~ %.2f，步长 %.2f，共 %d 个候选" % (
        THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP, len(_threshold_candidates())))


def print_eval_report(index, eval_rows):
    report = analyze(index, eval_rows)
    n_in, n_out = report["n_in"], report["n_out"]
    threshold = report["threshold"]
    best = report["best"]

    _print_header(index, eval_rows)

    print("")
    print("[1] 拒答阈值网格搜索（* 为最终选择）")
    print("阈值    Top-1命中    Top-3命中    拒答正确   综合分")
    for row in report["sweep"]:
        mark = "*" if row["threshold"] == threshold else " "
        print("%s %.2f   %.3f(%2d/%d)   %.3f(%2d/%d)   %.3f(%2d/%d)   %.4f" % (
            mark,
            row["threshold"],
            row["hit1"] / n_in, row["hit1"], n_in,
            row["hit3"] / n_in, row["hit3"], n_in,
            row["rejected"] / n_out, row["rejected"], n_out,
            row["score"],
        ))
    print("选择标准：最大化「库内 Top-1 命中率 + 库外拒答正确率」；")
    print("          同分时优先拒答正确率更高（更安全），再取较小阈值（避免过度拒答）。")
    print("=> 最优阈值 = %.2f，综合分 = %.4f" % (threshold, best["score"]))

    print("")
    print("[2] 最终指标（阈值 = %.2f，库内题被拒答计为未命中）" % threshold)
    print("库内 Top-1 命中率 : %.3f  (%d/%d)" % (report["hit1"], best["hit1"], n_in))
    print("库内 Top-3 命中率 : %.3f  (%d/%d)" % (report["hit3"], best["hit3"], n_in))
    print("库外 拒答正确率   : %.3f  (%d/%d)" % (
        report["reject_acc"], best["rejected"], n_out))
    print("库外 误答率       : %.3f  (%d/%d)" % (
        report["wrong_rate"], n_out - best["rejected"], n_out))
    print("库内题被误拒      : %d/%d" % (report["in_rejected"], n_in))

    baseline = report["baseline"]
    print("")
    print("[3] 对照 baseline：不设拒答，永远返回相似度最高的 Top-1")
    print("库内 Top-1 命中率 : %.3f  (%d/%d)" % (
        baseline["hit1"], round(baseline["hit1"] * n_in), n_in))
    print("库内 Top-3 命中率 : %.3f  (%d/%d)" % (
        baseline["hit3"], round(baseline["hit3"] * n_in), n_in))
    print("库外 误答率       : %.3f  (%d/%d)  <- 库外问题全部被硬答" % (
        baseline["wrong_rate"], n_out, n_out))

    badcases = report["badcases"]
    shown = badcases[:10]
    print("")
    print("[4] badcase 示例（共 %d 条，展示 %d 条）" % (len(badcases), len(shown)))
    if not shown:
        print("（没有 badcase）")
    for order, case in enumerate(shown, start=1):
        top1 = case["top1"]
        if case["type"] == "库外误答":
            print("%2d) [库外误答] 问：%s" % (order, case["question"]))
            print("    系统硬答：#%s %s（相似度 %.4f，阈值 %.2f）" % (
                top1["id"], top1["question"], case["sim"], threshold))
            continue
        expected_question = next(
            item["question"] for item in index["kb"] if item["id"] == case["expected"])
        rank_text = "未进 Top-3" if case["correct_rank"] is None else "第 %d 名" % case["correct_rank"]
        print("%2d) [%s] 问：%s" % (order, case["type"], case["question"]))
        print("    期望：#%s %s；实际 Top-1：#%s %s（相似度 %.4f）" % (
            case["expected"], expected_question, top1["id"], top1["question"], case["sim"]))
        print("    正确条目排名：%s" % rank_text)
    return report


def print_ask(index, query):
    report = analyze(index, load_eval())
    threshold = report["threshold"]
    ranked = search(index, query)
    top1_sim = ranked[0][0]

    print("问题：%s" % query)
    print("拒答阈值：%.2f（由 --eval 的网格搜索得到）" % threshold)
    print("")
    print("Top-%d 检索结果：" % len(ranked))
    for rank, (sim, idx) in enumerate(ranked, start=1):
        item = index["kb"][idx]
        print("%d. [相似度 %.4f] %s（#%s）" % (rank, sim, item["question"], item["id"]))
        print("   答案：%s" % item["answer"])
    print("")
    if top1_sim >= threshold:
        print("结论：Top-1 相似度 %.4f >= 阈值 %.2f，采用检索结果回答：" % (top1_sim, threshold))
        print("  %s" % index["kb"][ranked[0][1]]["answer"])
    else:
        print("结论：Top-1 相似度 %.4f < 阈值 %.2f，判定为库外问题，拒答。" % (
            top1_sim, threshold))
        print("回答：%s" % REFUSAL_TEXT)


# ---------------------------------------------------------------- 入口

def main():
    parser = argparse.ArgumentParser(
        description="迷你知识库问答：字符 2-gram TF-IDF 检索 + 阈值拒答（仅标准库）")
    parser.add_argument("--eval", action="store_true", help="运行离线评测，打印阈值搜索过程、指标和 badcase")
    parser.add_argument("--ask", metavar="QUESTION", default=None, help="对单条问题做检索，输出 Top-3 或拒答")
    args = parser.parse_args()

    if not args.eval and not args.ask:
        parser.print_help()
        return

    index = build_index(load_kb())

    if args.eval:
        print_eval_report(index, load_eval())
    if args.ask is not None:
        if args.eval:
            print("")
            print("=" * 62)
        print_ask(index, args.ask)


if __name__ == "__main__":
    main()
