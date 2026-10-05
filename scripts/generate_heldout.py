"""Generate a held-out synthetic benchmark with generator-recorded injected defects.

The task packages (cases/) never contain the truth file. Truth is computed here by
code that is independent of aml_qc.core, from the generator's own design choices.
This is a synthetic injected-defect benchmark, not a blind human reference.
"""
import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random

TZ = timezone(timedelta(hours=8))
FIELDS = ["transaction_id", "account_id", "direction", "amount", "currency", "timestamp", "counterparty_token"]
LABELS = ["F1", "F2", "count", "material_relation", "alert_response", "amount_sum", "counterparty", "time_range"]
BUSINESSES = [
    ("合成社区餐饮店", "冷链食材", "餐饮营业收入"), ("合成服装批发档口", "成衣面料", "批发零售收入"),
    ("合成建材门市", "瓷砖板材", "建材销售收入"), ("合成手机配件店", "数码配件", "门店零售收入"),
    ("合成水果连锁店", "时令水果", "门店销售收入"), ("合成文具批发店", "办公文具", "批发销售收入"),
    ("合成宠物用品店", "宠物食品", "门店零售收入"), ("合成五金工具店", "五金工具", "门市销售收入"),
]
SUPPLIERS = ["丁食品", "庚纺织", "戊建材", "己电子", "辛农产", "壬文化", "丙商贸", "癸五金"]  # aligned with BUSINESSES
OTHER_PAYEES = [("子物业", "门店租金"), ("丑设备", "设备维修费"), ("寅物流", "运输费"), ("卯装修", "装修尾款")]
PROFILES = ["clean", "count_error", "amount_error", "counterparty_error", "time_error",
            "material_mismatch", "focus_omission", "partial_coverage"]


def iso(value):
    return value.isoformat()


def money(cents):
    return f"{cents // 100}.{cents % 100:02d}"


def cn_date(value):
    return f"{value.month}月{value.day}日"


def independent_features(start, end, transactions, coverage_full_until):
    """Spec §5 truth written separately from aml_qc.core; windows fully covered only before coverage_full_until."""
    def window_rows(a, b):
        return [t for t in transactions if a <= t["_ts"] < b]
    # F1: calendar-day windows, ratio out/in >= 0.8, both > 0, at least 3 met complete days.
    met = undetermined = 0
    day = start
    while day < end:
        nxt = day + timedelta(days=1)
        complete = nxt <= coverage_full_until
        rows = window_rows(day, nxt)
        tin = sum(t["_cents"] for t in rows if t["direction"] == "in")
        tout = sum(t["_cents"] for t in rows if t["direction"] == "out")
        if not complete:
            undetermined += 1
        elif tin > 0 and tout > 0 and tout * 10 >= tin * 8:
            met += 1
        day = nxt
    f1 = "met" if met >= 3 else "not_met" if met + undetermined < 3 else "undeterminable"
    # F2: 7-day windows anchored at start; any complete window met -> met.
    any_met, all_complete = False, True
    w = start
    while w < end:
        b = w + timedelta(days=7)
        if b > end:
            all_complete = False
            break
        if b > coverage_full_until:
            all_complete = False
        else:
            rows = window_rows(w, b)
            tin = sum(t["_cents"] for t in rows if t["direction"] == "in")
            tout = sum(t["_cents"] for t in rows if t["direction"] == "out")
            ins = {t["counterparty_token"] for t in rows if t["direction"] == "in"}
            outs = {t["counterparty_token"] for t in rows if t["direction"] == "out"}
            if tin > 0 and tout > 0 and len(ins) >= 10 and 1 <= len(outs) <= 2 and tout * 10 >= tin * 8:
                any_met = True
        w = b
    f2 = "met" if any_met else "not_met" if all_complete else "undeterminable"
    return {"F1": f1, "F2": f2}


def build_case(index, profile, variant, rng, family):
    case_id = f"heldout-{index:02d}"
    business, goods, income = BUSINESSES[index % len(BUSINESSES)]
    supplier = SUPPLIERS[index % len(SUPPLIERS)]
    other, other_use = OTHER_PAYEES[(index + variant) % len(OTHER_PAYEES)]
    month = 7 if variant % 2 else 8
    start = datetime(2026, month, 6 if month == 7 else 3, tzinfo=TZ)
    end = start + timedelta(days=14)
    account = f"hm-{index:02d}"
    s_token, o_token = f"hs-{index:02d}", f"ho-{index:02d}"
    concentrated = variant != 3  # one variant per profile has dispersed incoming (F2 not met).
    payers = 12 if concentrated else 6
    rows, counterparties = [], [
        {"counterparty_token": s_token, "display_name_masked": supplier, "credit_code": f"SYN-{s_token.upper()}", "type": "business"},
        {"counterparty_token": o_token, "display_name_masked": other, "credit_code": f"SYN-{o_token.upper()}", "type": "business"}]

    def add(direction, cents, ts, token, name, memo):
        rows.append({"transaction_id": f"{case_id}-{direction}-{len(rows):02d}", "account_id": account, "direction": direction,
                     "amount": money(cents), "currency": "CNY", "timestamp": iso(ts), "channel": "synthetic_qr",
                     "counterparty_token": token, "counterparty_name_masked": name, "memo": memo,
                     "_cents": cents, "_ts": ts})

    week1_in = 0
    for p in range(payers):
        token = f"hp-{index:02d}-{p:02d}"
        counterparties.append({"counterparty_token": token, "display_name_masked": f"合成付款账户{p:02d}", "type": "payment_account"})
        cents = rng.randrange(150, 420) * 100
        ts = start + timedelta(days=p % 5, hours=9 + p % 8, minutes=rng.randrange(0, 59))
        add("in", cents, ts, token, f"合成付款账户{p:02d}", "合成零售收款")
        week1_in += cents
    for p in range(4):
        token = f"hp-{index:02d}-{p:02d}"
        add("in", rng.randrange(80, 160) * 100, start + timedelta(days=8 + p, hours=11), token, f"合成付款账户{p:02d}", "合成零售收款")

    n_s = 1 if variant == 0 else 2 if variant in (1, 3) else 3
    target = week1_in * rng.randrange(85, 95) // 100
    parts = [target // n_s] * n_s
    parts[-1] += target - sum(parts)
    parts = [c // 100 * 100 for c in parts]
    s_days = [1, 3, 4][:n_s] if n_s > 1 else [3]
    for c, d in zip(parts, s_days):
        add("out", c, start + timedelta(days=d, hours=15, minutes=rng.randrange(0, 50)), s_token, supplier, f"合成{goods}货款")
    s_total = sum(parts)
    late_day = None
    if profile == "time_error":
        late_day = 11
        late_cents = rng.randrange(8, 20) * 10000
        add("out", late_cents, start + timedelta(days=late_day, hours=16), s_token, supplier, f"合成{goods}货款")
        n_s_true, s_total_true = n_s + 1, s_total + late_cents
    else:
        n_s_true, s_total_true = n_s, s_total
    has_other = profile in ("counterparty_error", "focus_omission") or variant == 2
    other_cents = rng.randrange(20, 45) * 10000
    other_ts = start + timedelta(days=9, hours=10)
    if has_other:
        add("out", other_cents, other_ts, o_token, other, f"合成{other_use}")

    # Coverage: partial_coverage profile has the second week incomplete.
    full_until = end
    coverage = [{"coverage_id": "transactions-all", "source": "transactions", "account_id": account, "fields": FIELDS,
                 "start": iso(start), "end": iso(end), "status": "full", "revision": "1"}]
    if profile == "partial_coverage":
        full_until = start + timedelta(days=7)
        coverage = [{"coverage_id": "week-1", "source": "transactions", "account_id": account, "fields": FIELDS,
                     "start": iso(start), "end": iso(full_until), "status": "full", "revision": "1"},
                    {"coverage_id": "week-2", "source": "transactions", "account_id": account, "fields": FIELDS,
                     "start": iso(full_until), "end": iso(end), "status": "partial", "revision": "1"}]

    # Narrative statements and their designed truth (independent of aml_qc.core).
    truth_claims = {}
    stated_n, stated_total = n_s_true, s_total_true
    if profile == "count_error":
        stated_n = 1 if n_s_true > 1 else 2
    if profile == "amount_error":
        stated_total = s_total_true - rng.choice([30000, 50000, 80000])
    if profile == "partial_coverage" and variant in (2, 3):
        stated_n = n_s_true - 1  # all supplier payments sit in the complete week: visible counterexample
    window_phrase = f"{cn_date(start)}至{cn_date(end - timedelta(days=1))}"
    s_last_stated = start + timedelta(days=6)
    count_word = {1: "一", 2: "两", 3: "三"}
    n_text = count_word.get(stated_n, str(stated_n))
    styles = [
        f"检查期间（{window_phrase}）仅向{supplier}付款{n_text}次，合计{money(stated_total)}元，用于采购{goods}",
        f"{window_phrase}本账户共向{supplier}支付{goods}货款{stated_n}笔，金额合计{money(stated_total)}元",
        f"本期向{supplier}共支付{stated_n}笔、合计{money(stated_total)}元，均为{goods}采购款",
        f"经核对，{window_phrase}向{supplier}转出{n_text}笔，总额{money(stated_total)}元，系{goods}进货支出",
    ]
    statement = styles[variant]
    truth_claims["count"] = "contradicted" if stated_n != n_s_true else "supported"
    truth_claims["amount_sum"] = "contradicted" if stated_total != s_total_true else "supported"
    if profile == "partial_coverage":
        # Second week incomplete: an exact whole-period count/sum cannot be proven,
        # but a visible complete-week counterexample still contradicts it.
        truth_claims["amount_sum"] = "insufficient_evidence"
        truth_claims["count"] = "contradicted" if stated_n != n_s_true else "insufficient_evidence"
    extra = []
    if profile == "counterparty_error" or (profile == "clean" and variant == 1):
        extra.append(f"检查期间转出资金全部支付给{supplier}，未向其他账户付款")
        truth_claims["counterparty"] = "contradicted" if has_other else "supported"
    if profile == "time_error" or (profile == "clean" and variant == 0) or (profile == "partial_coverage" and variant == 3):
        extra.append(f"上述付款均发生在{cn_date(start)}至{cn_date(s_last_stated)}之间")
        truth_claims["time_range"] = ("contradicted" if late_day is not None
                                      else "insufficient_evidence" if profile == "partial_coverage" else "supported")
    background = f"收款为{business.replace('合成', '')}{income}，来自散客扫码付款"
    other_sentence = ""
    focuses = [{"focus_id": "focus-1", "text": f"请核实集中收款的资金来源，以及向{supplier}付款的业务背景。"}]
    focus_truth = {"focus-1": "addressed"}
    if has_other and profile != "counterparty_error":
        focuses.append({"focus_id": "focus-2", "text": f"请说明{cn_date(other_ts)}向{other}转出{money(other_cents)}元的用途。"})
        if profile == "focus_omission":
            focus_truth["focus-2"] = "not_addressed"
        else:
            focus_truth["focus-2"] = "addressed"
            other_sentence = f"；{cn_date(other_ts)}向{other}支付的{money(other_cents)}元为{other_use}，相关单据已提交"
    narrative = "；".join([statement] + extra) + "；" + background + other_sentence + "。"

    # Material bound to the supplier payments.
    s_rows = [r for r in rows if r["counterparty_token"] == s_token]
    linked = [r for r in s_rows if r["_ts"] < start + timedelta(days=7)]
    template = "single_purchase_payment" if len(linked) == 1 else "contract_installments"
    m_amount = sum(r["_cents"] for r in linked) if template == "single_purchase_payment" else sum(r["_cents"] for r in linked) + rng.randrange(5, 30) * 10000
    m_start, m_end = start - timedelta(days=20), start + timedelta(days=60)
    material_truth = "corresponds"
    if profile == "material_mismatch":
        if template == "single_purchase_payment":
            m_amount -= 20000
        elif variant % 2:
            m_amount = sum(r["_cents"] for r in linked) - 30000
        else:
            m_end = start + timedelta(days=2)
        material_truth = "mismatch"
    material = {"material_id": "supply-doc", "revision": "1",
                "material_type": "purchase_invoice" if template == "single_purchase_payment" else "purchase_contract",
                "source": "synthetic", "subject": {"account_id": account, "role": "buyer"},
                "counterparty": {"counterparty_token": s_token, "role": "seller"},
                "period": {"start": iso(m_start), "end": iso(m_end)}, "amount": money(m_amount), "currency": "CNY",
                "text": f"合成{goods}{'采购发票' if template == 'single_purchase_payment' else '采购合同'}，仅用于字段对应核验，不证明真伪或全部收款来源。"}
    link = {"link_id": "supply-link", "material_id": "supply-doc", "revision": "1", "claim_or_issue_id": "focus-1",
            "transaction_ids": sorted(r["transaction_id"] for r in linked),
            "field_paths": ["subject.account_id", "counterparty.counterparty_token", "period", "amount"],
            "relation_template": template, "schema_version": "S1.0", "provenance": "synthetic_operator_link"}
    narrative += f"{goods}{'采购发票' if template == 'single_purchase_payment' else '采购合同'}见材料。"

    features = independent_features(start, end, rows, full_until)
    visible = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    case = {"case_id": case_id, "case_family": family, "task_mode": "alert_review", "subject_account_id": account,
            "profile": {"business_type": business, "data_origin": "synthetic"}, "data_version": "1",
            "coverage_start": iso(start), "coverage_end": iso(end), "currency": "CNY", "timezone": "Asia/Shanghai",
            "schema_version": "S1.0", "coverage": coverage, "counterparties": counterparties,
            "transactions": sorted(visible, key=lambda r: r["timestamp"]), "materials": [material], "material_links": [link],
            "entity_mappings": [
                {"mapping_id": f"map-{s_token}", "source_ref": supplier, "target_token": s_token, "confirmed": True, "revision": "1",
                 "basis": f"合成案例明确主体标识SYN-{s_token.upper()}；不依据名称相似"},
                {"mapping_id": f"map-{o_token}", "source_ref": other, "target_token": o_token, "confirmed": True, "revision": "1",
                 "basis": f"合成案例明确主体标识SYN-{o_token.upper()}；不依据名称相似"}],
            "alert": {"alert_id": f"alert-{case_id}", "revision": "1", "focuses": focuses, "original_focus": focuses[0]["text"],
                      "trigger_features": ["F1", "F2"], "subject_account_id": account, "start": iso(start), "end": iso(end),
                      "rule_source": "合成演示预警，非监管阈值", "rule_version": "S1.0"},
            "documents": [{"document_id": "narrative", "revision": "1", "source": "synthetic_operator_statement", "text": narrative},
                          {"document_id": "kyc", "revision": "1", "source": "synthetic_unverified_profile",
                           "text": f"本例为虚构{business.replace('合成', '')}，经营{goods}相关业务。客户自述未经外部核实。"}],
            "review_scope": {"target_labels": LABELS}}
    expected_issue_types = sorted({"claim_error" for v in truth_claims.values() if v == "contradicted"}
                                  | ({"material_mismatch"} if material_truth == "mismatch" else set())
                                  | ({"focus_not_addressed"} if "not_addressed" in focus_truth.values() else set()))
    truth = {"case_id": case_id, "profile": profile, "variant": variant, "family": family,
             "claims": truth_claims, "material_relation": material_truth, "focuses": focus_truth, "features": features,
             "expected_issue_types": expected_issue_types,
             "expected_return_for_revision": bool(expected_issue_types),
             "partial_coverage_after": iso(full_until) if full_until < end else None}
    return case, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    cases_dir, truth_dir = args.output / "cases", args.output / "truth"
    cases_dir.mkdir(parents=True, exist_ok=False)
    truth_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"contract": "heldout-injected-defect-benchmark-1", "seed": args.seed,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "truth_kind": "generator_injected_defects_not_human_reference", "cases": []}
    index = 1
    for p, profile in enumerate(PROFILES):
        for variant in range(4):
            family = f"heldout-family-{profile}"
            case, truth = build_case(index, profile, variant, rng, family)
            body = json.dumps(case, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
            (cases_dir / f"{case['case_id']}.json").write_text(body)
            (truth_dir / f"{case['case_id']}.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
            manifest["cases"].append({"case_id": case["case_id"], "family": family,
                                      "case_sha256": hashlib.sha256(body.encode()).hexdigest()})
            index += 1
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"cases": len(manifest["cases"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
