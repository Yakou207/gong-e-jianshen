import pytest

from aml_qc.intake import build_case

FORM = {"case_id": "up-1", "account_id": "a1", "coverage_end": "2026-09-08", "focuses": ["收付背景"], "synthetic_confirmed": True}


def test_chinese_bank_export_headers_and_formats_are_accepted():
    csv = ("交易流水号,借贷标志,交易金额,交易时间,对方账号,对方户名,摘要\n"
           "t1,贷,\"¥1,200.5\",2026/9/1 9:05,p1,付款人一,货款\n"
           "t2,借,190.50元,2026年9月2日 15时00分,s1,某贸易,采购\n")
    case = build_case({**FORM, "transactions_csv": csv})
    rows = case["transactions"]
    assert [(r["direction"], r["amount"], r["timestamp"]) for r in rows] == [
        ("in", "1200.50", "2026-09-01T09:05:00+08:00"), ("out", "190.50", "2026-09-02T15:00:00+08:00")]
    assert rows[1]["counterparty_name_masked"] == "某贸易" and rows[0]["memo"] == "货款"


def test_every_bad_row_is_reported_together_and_duplicates_are_refused():
    csv = ("transaction_id,direction,amount,timestamp,counterparty_token\n"
           "t1,in,100,2026-09-01 09:00,p1\n"
           "t1,in,100,2026-09-01 10:00,p1\n"
           "t3,sideways,100,2026-09-01 10:00,p1\n"
           "t4,out,-5,2026-02-31,p1\n")
    with pytest.raises(ValueError) as error:
        build_case({**FORM, "transactions_csv": csv})
    message = str(error.value)
    assert "3 行无法导入" in message and "与第 2 行重复" in message and "方向无效" in message and "金额无效" in message


def test_missing_headers_name_the_chinese_alternative():
    with pytest.raises(ValueError, match="交易时间"):
        build_case({**FORM, "transactions_csv": "交易流水号,借贷标志,交易金额,对方账号\nt1,贷,1,p1\n"})
