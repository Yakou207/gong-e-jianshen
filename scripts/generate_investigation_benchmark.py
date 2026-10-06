"""Held-out benchmark for the AI 研判 Agent: 24 synthetic alerts whose expected recommendation is set by design.

Three designed classes (8 each): legitimate merchant (exclude), pass-through "跑分" account (report_suspicious),
and plausible business with an unexplained large outflow or incomplete flow (insufficient_evidence).
Truth stays in truth/, never in the task package. Synthetic and generator-labelled, not a human reference.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random

TZ = timezone(timedelta(hours=8))
FIELDS = ["transaction_id", "account_id", "direction", "amount", "currency", "timestamp", "counterparty_token"]
MERCHANTS = [("社区餐饮店", "冷链食材", "丁食品"), ("服装批发档口", "成衣面料", "庚纺织"), ("建材门市", "瓷砖板材", "戊建材"),
             ("手机配件店", "数码配件", "己电子"), ("水果连锁店", "时令水果", "辛农产"), ("文具批发店", "办公文具", "壬文化"),
             ("宠物用品店", "宠物食品", "丙商贸"), ("五金工具店", "五金工具", "癸五金")]
PERSONS = ["学生，无经营收入申报", "自由职业，自述做网络兼职", "无业，开户时未提供工作单位", "在校研究生，月生活费约两千元",
           "外卖骑手，月收入约六千元", "个人账户，开户用途填写为日常消费", "退休人员，主要收入为养老金", "务农，季节性收入"]
FOCUSES = {
    "exclude": ["请核实集中收款的资金来源，以及向{s}付款的业务背景。"],
    "report_suspicious": ["请核实大量个人小额转入的来源，以及资金快速转出至少数个人账户的用途。"],
    "insufficient_evidence": ["请核实集中收款的资金来源，以及向{s}付款的业务背景。", "请说明{d}向{o}转出{a}元的用途。"],
}


def money(cents):
    return f"{cents // 100}.{cents % 100:02d}"


def build(index, kind, variant, rng):
    case_id = f"inv-{index:02d}"
    start = datetime(2026, 8 if variant % 2 else 7, 3 + variant, tzinfo=TZ)
    end = start + timedelta(days=14)
    account = f"ia-{index:02d}"
    rows, parties, materials, links, documents = [], [], [], [], []

    def party(token, name, kind_="payment_account"):
        if all(p["counterparty_token"] != token for p in parties):
            parties.append({"counterparty_token": token, "display_name_masked": name, "type": kind_})

    def tx(direction, cents, ts, token, name, memo):
        rows.append({"transaction_id": f"{case_id}-{direction}-{len(rows):02d}", "account_id": account, "direction": direction,
                     "amount": money(cents), "currency": "CNY", "timestamp": ts.isoformat(), "channel": "synthetic",
                     "counterparty_token": token, "counterparty_name_masked": name, "memo": memo})
        return rows[-1]["transaction_id"]

    coverage_status, focus_vars = "full", {}
    if kind in ("exclude", "insufficient_evidence"):
        business, goods, supplier = MERCHANTS[(index + variant) % len(MERCHANTS)]
        s_token = f"is-{index:02d}"
        party(s_token, supplier, "business")
        total_in = 0
        for p in range(12 + variant % 3):
            token = f"ip-{index:02d}-{p:02d}"
            party(token, f"合成顾客{p:02d}")
            cents = rng.randrange(60, 380) * 100 + rng.randrange(0, 99)
            total_in += cents
            tx("in", cents, start + timedelta(days=p % 7, hours=10 + p % 9, minutes=rng.randrange(60)), token, f"合成顾客{p:02d}", "合成扫码收款")
        pay = total_in * rng.randrange(55, 75) // 100 // 100 * 100
        out_id = tx("out", pay, start + timedelta(days=5, hours=15), s_token, supplier, f"合成{goods}货款")
        materials.append({"material_id": "invoice", "revision": "1", "material_type": "purchase_invoice", "source": "synthetic",
                          "subject": {"account_id": account, "role": "buyer"}, "counterparty": {"counterparty_token": s_token, "role": "seller"},
                          "period": {"start": (start - timedelta(days=10)).isoformat(), "end": (end + timedelta(days=20)).isoformat()},
                          "amount": money(pay), "currency": "CNY", "text": f"合成{goods}采购发票，金额 {money(pay)} 元。"})
        links.append({"link_id": "invoice-link", "material_id": "invoice", "revision": "1", "claim_or_issue_id": "focus-1",
                      "transaction_ids": [out_id], "field_paths": ["subject.account_id", "counterparty.counterparty_token", "period", "amount"],
                      "relation_template": "single_purchase_payment", "schema_version": "S1.0", "provenance": "synthetic_operator_link"})
        kyc = f"本例为虚构{business}，经营{goods}零售，开业三年，月营业额约 {round(total_in * 2 / 100000)} 千元，以散客扫码收款为主，向固定供应商进货。"
        focus_vars["s"] = supplier
        if kind == "insufficient_evidence":
            o_token, other = f"iu-{index:02d}", ["合成个人账户甲", "合成个人账户乙", "某咨询工作室", "合成个人账户丙"][variant % 4]
            party(o_token, other)
            big = rng.randrange(40, 90) * 1000 * 100 // 10
            day = start + timedelta(days=9, hours=11)
            if variant < 5:
                tx("out", big, day, o_token, other, "转账")
                focus_vars.update(d=f"{day.month}月{day.day}日", o=other, a=money(big))
            else:
                coverage_status = "partial"  # second week is not available
                focus_vars.update(d=f"{day.month}月{day.day}日", o=other, a=money(big))
                tx("out", big, day, o_token, other, "转账")
    else:
        kyc = f"本例为虚构个人客户，{PERSONS[variant % len(PERSONS)]}。客户自述未经核实。"
        receivers = [(f"ir-{index:02d}-{k}", f"合成个人账户{'甲乙丙'[k]}") for k in range(1 + variant % 2)]
        for token, name in receivers:
            party(token, name)
        for day in range(0, 12, 1 + variant % 2):
            base = start + timedelta(days=day)
            day_in = 0
            for k in range(rng.randrange(4, 8)):
                token = f"ip-{index:02d}-{day:02d}-{k}"
                party(token, f"合成个人{day:02d}{k}")
                cents = rng.choice([500, 1000, 1500, 2000, 3000, 5000]) * 100
                day_in += cents
                hour = rng.choice([1, 2, 3, 22, 23, 14, 15])
                tx("in", cents, base + timedelta(hours=hour, minutes=rng.randrange(60)), token, f"合成个人{day:02d}{k}", "转账")
            token, name = receivers[day % len(receivers)]
            tx("out", day_in * rng.randrange(95, 100) // 100 // 100 * 100, base + timedelta(hours=23, minutes=40), token, name, "转账")
    coverage = [{"coverage_id": "flow", "source": "transactions", "account_id": account, "fields": FIELDS,
                 "start": start.isoformat(), "end": end.isoformat(), "status": "full", "revision": "1"}]
    if coverage_status == "partial":
        mid = start + timedelta(days=7)
        coverage = [{**coverage[0], "coverage_id": "week-1", "end": mid.isoformat()},
                    {**coverage[0], "coverage_id": "week-2", "start": mid.isoformat(), "status": "partial"}]
    focuses = [{"focus_id": f"focus-{i}", "text": t.format(**focus_vars)} for i, t in enumerate(FOCUSES[kind], 1)]
    case = {"case_id": case_id, "case_family": "inv-" + kind, "task_mode": "alert_review", "subject_account_id": account,
            "profile": {"business_type": "个人账户" if kind == "report_suspicious" else MERCHANTS[(index + variant) % 8][0],
                        "data_origin": "synthetic"},
            "data_version": "1", "coverage_start": start.isoformat(), "coverage_end": end.isoformat(), "currency": "CNY",
            "timezone": "Asia/Shanghai", "schema_version": "S1.0", "coverage": coverage, "counterparties": parties,
            "transactions": sorted(rows, key=lambda r: r["timestamp"]), "materials": materials, "material_links": links,
            "entity_mappings": [{"mapping_id": "map-" + p["counterparty_token"], "source_ref": p["display_name_masked"],
                                 "target_token": p["counterparty_token"], "confirmed": True, "revision": "1",
                                 "basis": "合成案例明确标识"} for p in parties if p["type"] == "business"],
            "alert": {"alert_id": "alert-" + case_id, "revision": "1", "focuses": focuses, "original_focus": focuses[0]["text"],
                      "trigger_features": ["F1", "F2"], "subject_account_id": account, "start": start.isoformat(),
                      "end": end.isoformat(), "rule_source": "合成演示预警", "rule_version": "S1.0"},
            "documents": [{"document_id": "kyc", "revision": "1", "source": "synthetic_unverified_profile", "text": kyc}],
            "review_scope": {"target_labels": ["F1", "F2", "alert_response"]}}
    reason = {"exclude": "收款为零售散客、支出有对应发票，客户资料与资金特征一致，流水完整",
              "report_suspicious": "客户无经营背景，大量个人整数转入后于当日深夜集中转给少数个人，无任何业务材料",
              "insufficient_evidence": "经营收付可解释，但存在无材料支持的大额转出" + ("且第二周流水不完整" if coverage_status == "partial" else "")}[kind]
    return case, {"case_id": case_id, "expected_recommendation": kind, "variant": variant, "design_reason": reason,
                  "coverage": coverage_status}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    (args.output / "cases").mkdir(parents=True, exist_ok=False)
    (args.output / "truth").mkdir()
    manifest = {"contract": "investigation-benchmark-1", "seed": args.seed, "truth_kind": "generator_designed_not_human_reference",
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": []}
    index = 1
    for variant in range(8):
        for kind in ("exclude", "report_suspicious", "insufficient_evidence"):
            case, truth = build(index, kind, variant, rng)
            body = json.dumps(case, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
            (args.output / "cases" / f"{case['case_id']}.json").write_text(body)
            (args.output / "truth" / f"{case['case_id']}.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1) + "\n")
            manifest["cases"].append({"case_id": case["case_id"], "family": case["case_family"],
                                      "case_sha256": hashlib.sha256(body.encode()).hexdigest()})
            index += 1
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(len(manifest["cases"]), "cases")


if __name__ == "__main__":
    main()
