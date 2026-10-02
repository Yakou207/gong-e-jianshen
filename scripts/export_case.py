import argparse
from pathlib import Path
from aml_qc.exports import export_bundle
from aml_qc.llm import settings
from aml_qc.store import Store

parser = argparse.ArgumentParser(description="导出合成案件的JSON/JSONL/CSV审计包")
parser.add_argument("case_id")
parser.add_argument("output", type=Path)
args = parser.parse_args()
result = export_bundle(Store(settings()["AML_QC_DB"]).export(args.case_id), args.output)
print(result.resolve())
