"""Validate an evaluation-freeze-1 manifest without running models or scorers.

All file references are {"path": "relative/to/manifest", "sha256": "..."}.
Required manifest fields: experiment_id, status (draft/frozen), cases,
references, artifacts, methods, repeat_count, retry_policy, reference_counts,
pricing, currency, and total_currency_budget. Frozen manifests also require
frozen_at and approved_by. artifacts.family_grouping records reviewed economic
groups, their splits, reasons and reviewers independently of runtime names.
Cases declare case_id/family_id/split and required_check_ids. References point
to independent reference records, not business-engine output. See the tests
for a small complete, hand-authored example. This tool checks declarations
and exact bytes; it does not authenticate people or establish answer quality.

artifacts: spec, schema, generator, dependency_lock, scorer, family_grouping,
           prompts (list), tools (list); each value is a file reference.
family_grouping: reviewed_by, cases [{case_id, economic_group, split, reason}].
reference file: case_id, case_sha256, visible_case_sha256, reviewers, check_units.
The first hash binds original case bytes; the second binds raw_case_input(case)
serialized with ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
and a terminal newline. Both bindings are required and independently checked.
Each check unit
declares reference_check_id, label, object_scope, applicability
(applicable/not_applicable/unresolved), adjudication_status
(adjudicated/unresolved), reference_value, reason, evidence_sets (OR of AND
lists). object_scope uses focus_id or material_link_id for static objects.
People declare person_id/signed_at; reviewers additionally declare every
EXPOSURES field below as a boolean. reference_protocol defaults to strict-blind-1,
which rejects any declared exposure. independent-initial-references-1 permits
authored_narrative only with a nonempty reviewer.authorship_bias_disclosure.
Reviewer exposure flags and signed_at describe the original declaration before
that reviewer's independent initial reference seal. Preserve the initial drafts
and declarations; record subsequent discussion separately rather than rewriting
pre-seal exposure flags. prior_seen_related_family describes exposure outside
this declared authoring/reference process, such as calibration or previously
answered related cases; the actual exposure must still be declared.
methods: B0/Fixed/Agent each declare model, generation, runner file reference,
and budget {max_calls, max_output_tokens, total_token_budget, currency_limit}.
reference_counts contains adjudicated/unresolved integers. retry_policy has
max_retries/retryable_errors. pricing references a JSON with currency,
effective_at, source_url, unit_tokens, rates {input_cache_hit,input_cache_miss,
output}. Currency amounts/rates use decimal strings. No data is synthesized.
"""
import argparse
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re

from aml_qc.schema import LABEL_VALUES
from aml_qc.baseline import raw_case_input


METHODS = ("B0", "Fixed", "Agent")
SPLITS = {"development", "validation", "test"}
EXPOSURES = ("saw_generator_truth", "saw_generator_private", "saw_other_reference", "saw_model_outputs",
             "authored_narrative", "prior_seen_related_family")
REFERENCE_PROTOCOLS = ("strict-blind-1", "independent-initial-references-1")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_json(raw):
    def invalid_constant(value):
        raise ValueError("non-finite JSON number")
    return json.loads(raw, object_pairs_hook=_object, parse_constant=invalid_constant)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _timestamp(value):
    try:
        return _text(value) and datetime.fromisoformat(value).utcoffset() is not None
    except ValueError:
        return False


def _positive_decimal(value, *, zero=False):
    try:
        number = Decimal(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else Decimal("NaN")
        return number.is_finite() and (number >= 0 if zero else number > 0)
    except InvalidOperation:
        return False


def _credential_path(path):
    return any(part.lower() == ".env" or part.lower().startswith(".env.") for part in path.parts)


class FreezeAudit:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path).absolute()
        self.issues = []
        self.files = []
        self.exposures = []
        self.reference_protocol = "strict-blind-1"

    def check(self, condition, code, location, message):
        if not condition:
            self.issues.append({"code": code, "location": location, "message": message})
        return bool(condition)

    def file(self, ref, location, *, json_object=False):
        if not self.check(isinstance(ref, dict), "file_reference_required", location,
                          "Provide an explicit path and SHA-256; no file is inferred."):
            return None
        name, expected = ref.get("path"), ref.get("sha256")
        if not self.check(_text(name), "path_required", location, "File path is missing."):
            return None
        if not self.check(not Path(name).is_absolute(), "absolute_path_forbidden", location,
                          "Use a path relative to the manifest, including ../ when needed."):
            return None
        path = self.path.parent / name
        try:
            resolved = path.resolve()
            if not self.check(not _credential_path(path) and not _credential_path(resolved),
                              "credential_file_forbidden", location, ".env files are never read."):
                return None
            raw = resolved.read_bytes()
        except (OSError, ValueError, RuntimeError):
            self.check(False, "file_unreadable", location, "Declared file cannot be read.")
            return None
        actual = hashlib.sha256(raw).hexdigest()
        matched = self.check(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None
                             and expected == actual, "file_hash_mismatch", location,
                             "Declared SHA-256 is missing, malformed, or differs from exact file bytes.")
        self.files.append({"location": location, "path": name, "sha256_observed": actual, "hash_matches": matched})
        if json_object:
            try:
                value = _read_json(raw)
                if not isinstance(value, dict):
                    raise ValueError("JSON object required")
                return value
            except (ValueError, UnicodeError):
                self.check(False, "invalid_json_object", location, "Expected an unambiguous JSON object.")
        return None

    def people(self, value, location, minimum=1):
        if not self.check(isinstance(value, list) and len(value) >= minimum,
                          "person_declarations_missing", location, "Actual personnel declarations are required."):
            return
        identities = []
        for index, person in enumerate(value):
            loc = f"{location}[{index}]"
            if not self.check(isinstance(person, dict), "invalid_person_declaration", loc, "Expected a personnel record."):
                continue
            identity = person.get("person_id")
            self.check(_text(identity), "person_id_missing", loc, "Actual person_id is required; not an authenticated identity.")
            if _text(identity):
                identities.append(identity.strip().casefold())
            self.check(_timestamp(person.get("signed_at")), "person_timestamp_missing", loc,
                       "An actual timezone-aware signing timestamp is required.")
            if minimum >= 2:
                exposure = person.get("exposure")
                complete = isinstance(exposure, dict) and all(type(exposure.get(k)) is bool for k in EXPOSURES)
                self.check(complete, "exposure_declaration_missing", loc,
                           "Explicit boolean declarations for every exposure category are required.")
                authored = isinstance(exposure, dict) and exposure.get("authored_narrative") is True
                author_allowed = self.reference_protocol == "independent-initial-references-1"
                disclosure = person.get("authorship_bias_disclosure")
                if authored and author_allowed:
                    self.check(_text(disclosure), "authorship_bias_disclosure_missing", loc,
                               "Author-reviewers must explicitly disclose authorship bias; they are not fully blinded.")
                self.exposures.append({"location": loc, "person_id": identity,
                    "declaration_complete": complete,
                    "authorship_bias_disclosure": disclosure if authored else None,
                    "declared_exposures": [k for k in EXPOSURES if isinstance(exposure, dict) and exposure.get(k) is True]})
                blocked = [k for k in EXPOSURES if k != "authored_narrative" or not author_allowed]
                self.check(not isinstance(exposure, dict) or not any(exposure.get(k) is True for k in blocked),
                           "reference_exposure", loc,
                           "Private/truth, peer-reference, model-output and outside-protocol related-family exposure block initial references; strict-blind-1 also blocks author exposure.")
        self.check(len(identities) == len(set(identities)), "duplicate_person", location,
                   "Personnel IDs must be distinct after whitespace/case normalization.")

    def run(self, manifest):
        self.check(manifest.get("contract_version") == "evaluation-freeze-1", "contract_version", "contract_version",
                   "Expected evaluation-freeze-1; older templates need explicit conversion.")
        self.check(_text(manifest.get("experiment_id")), "experiment_id_missing", "experiment_id", "Experiment ID is required.")
        self.reference_protocol = manifest.get("reference_protocol", "strict-blind-1")
        self.check(_text(self.reference_protocol) and self.reference_protocol in REFERENCE_PROTOCOLS,
                   "reference_protocol_invalid", "reference_protocol", "Use strict-blind-1 or independent-initial-references-1.")
        status = manifest.get("status")
        self.check(_text(status) and status in {"draft", "frozen"}, "invalid_status", "status", "Expected draft or frozen.")
        if status == "frozen":
            self.check(_timestamp(manifest.get("frozen_at")), "freeze_timestamp_missing", "frozen_at",
                       "A timezone-aware freeze timestamp is required.")
            self.people(manifest.get("approved_by"), "approved_by")

        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
        for key in ("spec", "generator", "dependency_lock", "scorer"):
            self.file(artifacts.get(key), "artifacts." + key)
        schema = self.file(artifacts.get("schema"), "artifacts.schema", json_object=True) or {}
        self.check(isinstance(schema.get("labels"), dict) and bool(schema["labels"]), "schema_labels_missing",
                   "artifacts.schema", "Frozen schema must declare its target labels.")
        grouping = self.file(artifacts.get("family_grouping"), "artifacts.family_grouping", json_object=True) or {}
        self.people(grouping.get("reviewed_by"), "artifacts.family_grouping.reviewed_by")
        group_rows = grouping.get("cases", [])
        group_map = {}
        if self.check(isinstance(group_rows, list) and bool(group_rows), "family_grouping_empty", "artifacts.family_grouping",
                      "A reviewed economic grouping must cover every case."):
            for number, row in enumerate(group_rows):
                loc = f"artifacts.family_grouping.cases[{number}]"
                if not self.check(isinstance(row, dict) and _text(row.get("case_id"))
                                  and _text(row.get("economic_group")) and _text(row.get("reason")),
                                  "invalid_family_grouping", loc, "Declare case_id, economic_group, split, and grouping reason."):
                    continue
                self.check(row["case_id"] not in group_map, "duplicate_family_grouping", loc, "Duplicate grouping case.")
                group_map[row["case_id"]] = row
        for key in ("prompts", "tools"):
            refs = artifacts.get(key)
            if self.check(isinstance(refs, list) and bool(refs), "artifact_list_empty", "artifacts." + key,
                          "Declare every used file and its hash."):
                for index, ref in enumerate(refs):
                    self.file(ref, f"artifacts.{key}[{index}]")

        cases = manifest.get("cases")
        if not self.check(isinstance(cases, list) and bool(cases), "case_coverage_empty", "cases", "No case coverage declared."):
            cases = []
        indexed, families, packages, known_groups = {}, {}, {}, set()
        visible_hashes = {}
        for index, case in enumerate(cases):
            loc = f"cases[{index}]"
            if not self.check(isinstance(case, dict), "invalid_case", loc, "Expected a case record."):
                continue
            case_id, family, split = case.get("case_id"), case.get("family_id"), case.get("split")
            if not self.check(_text(case_id), "case_id_missing", loc, "case_id is required."):
                continue
            self.check(case_id not in indexed, "duplicate_case", loc, "case_id must be unique.")
            indexed[case_id] = case
            self.check(_text(family), "family_missing", loc, "Each case requires an independently reviewed family grouping.")
            self.check(_text(split) and split in SPLITS, "invalid_split", loc, "Use development, validation, or test.")
            if _text(family):
                self.check(family not in families or families[family] == split, "family_crosses_split", loc,
                           "All variants of one case family must remain in the same split.")
                families[family] = split
            grouping_row = group_map.get(case_id, {})
            self.check(grouping_row.get("economic_group") == family and grouping_row.get("split") == split,
                       "family_grouping_mismatch", loc, "Case grouping/split must match the separately reviewed grouping file.")
            if case_id in {f"seed-{n:02d}" for n in range(1, 7)}:
                self.check(split == "development", "known_seed_not_development", loc,
                           "The six previously exposed product seeds are development-only.")
                if _text(family):
                    known_groups.add(family)
            units = case.get("required_check_ids")
            self.check(isinstance(units, list) and bool(units) and all(_text(x) for x in units)
                       and len(units) == len(set(x for x in units if isinstance(x, str))),
                       "required_checks_invalid", loc, "Declare a nonempty unique check inventory before any execution.")
            package = self.file(case, loc, json_object=True)
            if package is not None:
                self.check(package.get("case_id") == case_id, "case_id_mismatch", loc, "Case file and manifest IDs differ.")
                packages[case_id] = package
                try:
                    visible_raw = (json.dumps(raw_case_input(package), ensure_ascii=False, sort_keys=True,
                                              indent=2, allow_nan=False) + "\n").encode()
                    visible_hashes[case_id] = hashlib.sha256(visible_raw).hexdigest()
                except (ValueError, TypeError, KeyError, OSError, UnicodeError):
                    self.check(False, "case_visible_projection_invalid", loc,
                               "Case must produce a valid visible reviewer projection before freezing.")
        self.check(set(group_map) == set(indexed), "family_grouping_coverage", "artifacts.family_grouping",
                   "Reviewed economic grouping must exactly cover the declared cases.")
        self.check(len(known_groups) <= 1, "known_seed_family_split", "cases",
                   "Known seed-01 through seed-06 belong to one conservative development group, not six independent families.")

        references = manifest.get("references")
        if not self.check(isinstance(references, list) and bool(references), "reference_coverage_empty", "references",
                          "Independent references must be explicitly supplied; engine outputs are not references."):
            references = []
        seen, counts = set(), Counter(adjudicated=0, unresolved=0)
        for index, ref in enumerate(references):
            loc = f"references[{index}]"
            if not self.check(isinstance(ref, dict), "invalid_reference", loc, "Expected a reference file declaration."):
                continue
            cid = ref.get("case_id")
            if not self.check(_text(cid) and cid in indexed, "unknown_reference_case", loc,
                              "Reference must belong to a declared case."):
                continue
            self.check(cid not in seen, "duplicate_reference", loc, "Exactly one final reference file per case is required.")
            seen.add(cid)
            record = self.file(ref, loc, json_object=True)
            if record is None:
                continue
            self.check(record.get("case_id") == cid and record.get("case_sha256") == indexed[cid].get("sha256"),
                       "reference_case_binding", loc, "Reference must bind the exact declared case bytes and ID.")
            visible_hash = record.get("visible_case_sha256")
            self.check(_text(visible_hash) and re.fullmatch(r"[0-9a-f]{64}", visible_hash) is not None
                       and visible_hash == visible_hashes.get(cid), "reference_visible_case_binding", loc,
                       "Reference must also bind the exact current visible reviewer projection, including its schema.")
            self.people(record.get("reviewers"), loc + ".reviewers", minimum=2)
            units = record.get("check_units")
            if not self.check(isinstance(units, list) and bool(units), "reference_units_empty", loc,
                              "Reference check_units cannot be empty."):
                units = []
            unit_ids = []
            for number, unit in enumerate(units):
                at = f"{loc}.check_units[{number}]"
                if not self.check(isinstance(unit, dict), "invalid_reference_unit", at, "Expected a reference unit."):
                    continue
                uid, adjudication = unit.get("reference_check_id"), unit.get("adjudication_status")
                self.check(_text(uid), "reference_check_id_missing", at, "Independent check ID is required.")
                if _text(uid):
                    unit_ids.append(uid)
                label = unit.get("label")
                self.check(_text(label) and label in LABEL_VALUES, "reference_label_not_scoreable", at,
                           "Reference label must have an implemented scoring domain.")
                self.check(_text(label) and isinstance(unit.get("object_scope"), dict) and bool(unit["object_scope"]),
                           "reference_scope_missing", at, "Explicit check type and object/range are required.")
                applicability = unit.get("applicability")
                self.check(_text(applicability) and applicability in {"applicable", "not_applicable", "unresolved"}, "reference_applicability", at,
                           "Applicability must be declared separately from execution or business status.")
                self.check(_text(unit.get("reason")), "reference_reason_missing", at, "Human reference reasoning is required.")
                if self.check(_text(adjudication) and adjudication in {"adjudicated", "unresolved"}, "reference_not_reviewed", at,
                              "Explicitly record adjudicated or unresolved; templates are not references."):
                    counts[adjudication] += 1
                self.check("reference_value" in unit, "reference_value_missing", at, "Explicit value or unresolved null is required.")
                if adjudication == "unresolved":
                    self.check(unit.get("reference_value") is None, "unresolved_has_value", at,
                               "Unresolved references cannot carry a fabricated final value.")
                elif adjudication == "adjudicated":
                    self.check(applicability != "unresolved" and (applicability == "not_applicable" or unit.get("reference_value") is not None),
                               "adjudicated_value_missing", at, "Adjudicated applicable units require a final value.")
                    if applicability == "applicable":
                        self.check(_text(label) and label in LABEL_VALUES and unit.get("reference_value") in LABEL_VALUES[label],
                                   "reference_value_not_scoreable", at,
                                   "Adjudicated applicable values must belong to the label's implemented scoring domain.")
                evidence = unit.get("evidence_sets")
                self.check(isinstance(evidence, list) and all(isinstance(group, list) and bool(group)
                           and all(isinstance(item, dict) and bool(item) for item in group) for group in evidence)
                           and (bool(evidence) or adjudication == "unresolved"), "reference_evidence_sets", at,
                           "Use OR between evidence sets and AND within each nonempty set; unresolved may have no set.")
            required = indexed[cid].get("required_check_ids", [])
            self.check(len(unit_ids) == len(set(unit_ids)), "duplicate_reference_unit", loc, "Reference units must be unique.")
            self.check(isinstance(required, list) and set(unit_ids) == {x for x in required if isinstance(x, str)},
                       "reference_check_coverage", loc, "Reference units must exactly cover the predeclared checks.")
            package = packages.get(cid, {})
            scope = package.get("review_scope", {})
            targets = scope.get("target_labels") if isinstance(scope, dict) else None
            if targets is None:
                targets = list(schema.get("labels", {})) if isinstance(schema.get("labels"), dict) else []
            labels = {u.get("label") for u in units if isinstance(u, dict) and _text(u.get("label"))}
            self.check(isinstance(targets, list) and bool(targets) and all(_text(t) for t in targets)
                       and set(t for t in targets if isinstance(t, str)) <= labels,
                       "reference_target_coverage", loc,
                       "Cover every case target label; absence of facts requires an explicit not_applicable reference with evidence.")
            alert = package.get("alert", {})
            focuses = alert.get("focuses", []) if isinstance(alert, dict) else []
            upgrades = scope.get("upgraded_leads", []) if isinstance(scope, dict) else []
            if isinstance(targets, list) and "alert_response" in targets:
                for focus in (focuses if isinstance(focuses, list) else []) + (upgrades if isinstance(upgrades, list) else []):
                    focus_id = focus.get("focus_id") if isinstance(focus, dict) else None
                    self.check(_text(focus_id) and any(isinstance(u, dict) and u.get("label") == "alert_response"
                               and isinstance(u.get("object_scope"), dict) and u["object_scope"].get("focus_id") == focus_id for u in units),
                               "reference_focus_coverage", loc, "Each static alert or upgraded focus needs its own reference object.")
            if isinstance(targets, list) and "material_relation" in targets:
                links = package.get("material_links", [])
                for link in links if isinstance(links, list) else []:
                    link_id = link.get("link_id") if isinstance(link, dict) else None
                    self.check(_text(link_id) and any(isinstance(u, dict) and u.get("label") == "material_relation"
                               and isinstance(u.get("object_scope"), dict) and u["object_scope"].get("material_link_id") == link_id for u in units),
                               "reference_material_link_coverage", loc, "Each static MaterialLink needs its own reference object.")
        self.check(seen == set(indexed), "reference_case_coverage", "references", "Every declared case needs its own reference.")
        declared = manifest.get("reference_counts")
        self.check(isinstance(declared, dict) and all(type(declared.get(k)) is int and declared[k] == counts[k]
                   for k in ("adjudicated", "unresolved")), "reference_count_mismatch", "reference_counts",
                   "Declared adjudicated/unresolved counts must equal the read unit records; missing is not zero.")

        methods = manifest.get("methods")
        if not isinstance(methods, dict):
            methods = {}
        self.check(set(methods) == set(METHODS), "required_methods", "methods", "B0, Fixed, and Agent are all required by section 13.")
        common, budgets = [], []
        for name in METHODS:
            method = methods.get(name)
            if not self.check(isinstance(method, dict), "method_missing", "methods." + name, "Method configuration is missing."):
                continue
            loc = "methods." + name
            self.check(_text(method.get("model")) and isinstance(method.get("generation"), dict) and bool(method["generation"]),
                       "model_config_missing", loc, "Freeze the model and actual generation parameters.")
            common.append((method.get("model"), method.get("generation")))
            self.file(method.get("runner"), loc + ".runner")
            budget = method.get("budget")
            if not isinstance(budget, dict):
                budget = {}
            self.check(all(type(budget.get(k)) is int and budget[k] > 0 for k in
                           ("max_calls", "max_output_tokens", "total_token_budget")) and _positive_decimal(budget.get("currency_limit")),
                       "method_budget_missing", loc + ".budget", "Freeze positive call, output, total-token and currency limits.")
            if name in {"Fixed", "Agent"}:
                budgets.append(budget)
        self.check(not common or all(value == common[0] for value in common), "method_config_differs", "methods",
                   "Comparison methods must use the same model and generation parameters.")
        self.check(not budgets or all(value == budgets[0] for value in budgets), "method_budget_differs", "methods",
                   "Fixed and Agent must have identical budget limits; B0 and actual costs are evaluated separately.")
        retry = manifest.get("retry_policy")
        self.check(isinstance(retry, dict) and type(retry.get("max_retries")) is int and retry["max_retries"] >= 0
                   and isinstance(retry.get("retryable_errors"), list) and all(_text(x) for x in retry["retryable_errors"]),
                   "retry_policy_missing", "retry_policy", "Freeze the shared retry count and allowed error classes.")
        pricing = self.file(manifest.get("pricing"), "pricing", json_object=True)
        if pricing is not None:
            self.check(_text(pricing.get("currency")) and _timestamp(pricing.get("effective_at"))
                       and _text(pricing.get("source_url")) and type(pricing.get("unit_tokens")) is int
                       and pricing["unit_tokens"] > 0 and isinstance(pricing.get("rates"), dict)
                       and all(_positive_decimal(pricing["rates"].get(k), zero=True) for k in
                               ("input_cache_hit", "input_cache_miss", "output")),
                       "pricing_incomplete", "pricing", "Freeze currency, price date/source, billing unit and all token rates.")
        currency, total_budget = manifest.get("currency"), manifest.get("total_currency_budget")
        self.check(_text(currency) and pricing is not None and currency == pricing.get("currency"),
                   "currency_mismatch", "currency", "Experiment and pricing currency must be explicit and equal.")
        self.check(_positive_decimal(total_budget), "total_budget_missing", "total_currency_budget",
                   "An explicit positive total experiment budget is required.")
        repeats = manifest.get("repeat_count")
        repeat_valid = self.check(type(repeats) is int and repeats > 0, "repeat_count_missing", "repeat_count", "Positive repeat_count is required.")
        plan = []
        if repeat_valid and self.check(len(indexed) * len(METHODS) * repeats <= 100000, "plan_too_large", "repeat_count",
                                       "Limit one manifest to 100000 planned executions."):
            for cid, case in indexed.items():
                for repeat in range(1, repeats + 1):
                    for method in METHODS:
                        key = [manifest.get("experiment_id"), cid, method, repeat]
                        plan.append({"planned_run_id": hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest(),
                            "case_id": cid, "family_id": case.get("family_id"), "split": case.get("split"),
                            "method": method, "repeat": repeat, "status": "not_run"})
        if "planned_runs" in manifest:
            supplied = manifest["planned_runs"]
            keys = []
            valid = isinstance(supplied, list)
            for row in supplied if isinstance(supplied, list) else []:
                good = (isinstance(row, dict) and _text(row.get("case_id")) and _text(row.get("method"))
                        and type(row.get("repeat")) is int and row["repeat"] > 0)
                valid = valid and good
                if good:
                    keys.append((row["case_id"], row["method"], row["repeat"]))
            self.check(valid, "invalid_planned_runs", "planned_runs", "Plan rows need case_id, method, and positive repeat.")
            self.check(len(keys) == len(set(keys)), "duplicate_planned_run", "planned_runs", "Duplicate planned execution.")
            self.check(set(keys) == {(r["case_id"], r["method"], r["repeat"]) for r in plan},
                       "planned_run_coverage", "planned_runs", "Plan must cover every declared case, all three methods, and every repeat.")
        ceiling = None
        if plan and all(isinstance(methods.get(m), dict) and isinstance(methods[m].get("budget"), dict)
                        and _positive_decimal(methods[m]["budget"].get("currency_limit")) for m in METHODS):
            ceiling = sum(Decimal(methods[r["method"]]["budget"]["currency_limit"]) for r in plan)
            self.check(_positive_decimal(total_budget) and ceiling <= Decimal(total_budget), "planned_budget_exceeds_total",
                       "total_currency_budget", "Sum of all planned per-run currency ceilings exceeds the declared total budget.")
        return {"contract_version": "evaluation-freeze-validation-1", "experiment_id": manifest.get("experiment_id"),
            "manifest_status": status, "ready_to_freeze": not self.issues, "frozen_valid": status == "frozen" and not self.issues,
            "issues": self.issues, "files": self.files, "reference_counts_observed": dict(counts),
            "planned_runs": plan, "planned_run_count": len(plan),
            "planned_currency_ceiling": str(ceiling) if ceiling is not None else None,
            "currency": currency, "execution_budget_enforced": False,
            "personnel_declarations": self.exposures,
            "reference_protocol": self.reference_protocol,
            "exposure_declaration_scope": "before_independent_initial_reference_seal",
            "author_reviewer_count": len({row["person_id"].strip().casefold() for row in self.exposures
                if _text(row["person_id"]) and "authored_narrative" in row["declared_exposures"]}),
            "blind_reference_declared_clear": bool(self.exposures) and all(
                row["declaration_complete"] and not row["declared_exposures"] for row in self.exposures),
            "limitations": ["File hashes and personnel declarations do not authenticate identities, independence, or reference quality.",
                "This validator executes no runner or scorer; invoke the saved-run scorer separately after freezing.",
                "Unresolved reference units remain explicit; this report is not a quality score or paid-cost result.",
                "Only submitted economic grouping declarations are checked; no automatic proof of family independence or complete Claim semantics.",
                "Budget checks compare declared ceilings only; no execution-time currency stop or actual-cost reconciliation is implemented.",
                "Pre-seal exposure declarations do not authenticate initial seals, preserved independent drafts, or later discussion chronology."]}


def validate_manifest(path):
    audit = FreezeAudit(path)
    try:
        if _credential_path(audit.path) or _credential_path(audit.path.resolve()):
            raise ValueError("credential manifest forbidden")
        raw = audit.path.read_bytes()
        manifest = _read_json(raw)
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
    except (OSError, ValueError, UnicodeError, RuntimeError):
        return {"frozen_valid": False, "ready_to_freeze": False, "issues": [
            {"code": "manifest_unreadable", "location": "manifest", "message": "Expected a readable non-credential JSON object."}],
            "planned_runs": [], "planned_run_count": 0}
    report = audit.run(manifest)
    report["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-frozen", action="store_true", help="Exit nonzero unless the complete frozen contract passes.")
    args = parser.parse_args()
    report = validate_manifest(args.manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(args.require_frozen and not report["frozen_valid"])


if __name__ == "__main__":
    raise SystemExit(main())
