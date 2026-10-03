"""Bounded model calls. Credentials never enter requests saved for replay."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import httpx

from .depgraph import digest

ROOT = Path(__file__).resolve().parents[1]


def settings():
    values = {}
    file = ROOT / ".env"
    if file.exists():
        for line in file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key in {"DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "AML_QC_DB"}:
                    values[key] = value.strip().strip("\"'")
    return {key: os.getenv(key, values.get(key, default)) for key, default in {
        "DEEPSEEK_API_KEY": "", "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
        "DEEPSEEK_MODEL": "deepseek-flash", "AML_QC_DB": str(ROOT / "runtime/workbench.sqlite3")}.items()}


class ModelError(RuntimeError):
    pass


class DeepSeek:
    def __init__(self, max_calls=6):
        self.config = settings()
        self.model = self.config["DEEPSEEK_MODEL"]
        self.max_calls = max_calls
        self.calls = []

    def complete(self, messages, tools=None):
        if not self.config["DEEPSEEK_API_KEY"]:
            raise ModelError("未配置 DeepSeek API Key；真实模型验收尚未开始")
        if len(self.calls) >= self.max_calls:
            raise ModelError("模型调用预算已耗尽，保留未完成项")
        payload = {"model": self.model, "messages": messages, "max_tokens": 4096,
                   "temperature": 0, "thinking": {"type": "disabled"}}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        else:
            payload["response_format"] = {"type": "json_object"}
        record = {"request_hash": digest(payload), "request": deepcopy(payload), "status": "started"}
        self.calls.append(record)
        start = perf_counter()
        try:
            response = httpx.post(self.config["DEEPSEEK_BASE_URL"].rstrip("/") + "/chat/completions",
                                  headers={"Authorization": "Bearer " + self.config["DEEPSEEK_API_KEY"]},
                                  json=payload, timeout=60)
            if response.status_code != 200:
                raise ModelError(f"DeepSeek HTTP {response.status_code}，本次模型步骤未完成")
            body = response.json()
            record.update(usage=body.get("usage"), model_returned=body.get("model"),
                          system_fingerprint=body.get("system_fingerprint"))
            choice = body["choices"][0]
            message = choice["message"]
            # Do not retain or display hidden chain-of-thought.
            message = {k: v for k, v in message.items() if k in {"role", "content", "tool_calls"} and v is not None}
            record.update(response=message, finish_reason=choice.get("finish_reason"))
            if choice.get("finish_reason") == "length":
                raise ModelError("模型输出被截断，不能用于业务通过")
            record["status"] = "completed"
            return message
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            record["status"] = "failed"
            raise ModelError("DeepSeek 请求或响应校验失败") from exc
        except ModelError:
            record["status"] = "failed"
            raise
        finally:
            record["duration_ms"] = round((perf_counter() - start) * 1000, 3)


class FrozenModel:
    """Mechanism tests only: exact complete-request lookup, never a live Agent."""
    def __init__(self, responses, model="frozen-test"):
        self.responses, self.model, self.calls = responses, model, []

    def complete(self, messages, tools=None):
        key = digest({"model": self.model, "messages": messages, "tools": tools})
        if key not in self.responses:
            raise ModelError("冻结请求未命中；禁止借用不同输入的模型产物")
        response = deepcopy(self.responses[key])
        self.calls.append({"request_hash": key, "status": "frozen", "usage": {}, "response": response})
        return response


def json_answer(message):
    try:
        result = json.loads(message.get("content") or "")
    except (ValueError, TypeError):
        raise ModelError("模型未返回有效 JSON") from None
    if not isinstance(result, dict):
        raise ModelError("模型 JSON 必须为对象")
    return result
