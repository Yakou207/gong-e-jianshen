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
REQUEST_TIMEOUT_SECONDS = 180
GENERATION = {"temperature": 0, "temperature_applies_to": "non_thinking_only",
              "max_output_tokens": 16384,
              "final_transport": "sse-with-usage-1",
              "stage_max_output_tokens": {"extraction": 4096, "tool_review": 4096, "final": 16384},
              "stage_thinking": {"extraction": "disabled", "tool_review": "disabled", "final": "enabled"},
              "reasoning_effort": "low", "reasoning_effort_applies_to": "thinking_only"}


def generation_request(model, messages, tools=None, stage=None):
    stage = stage if stage is not None else "tool_review" if tools else "final"
    if not isinstance(stage, str) or stage not in GENERATION["stage_thinking"]:
        raise ValueError("未知模型生成阶段")
    if (stage == "tool_review") != bool(tools):
        raise ValueError("模型生成阶段与工具参数不一致")
    thinking = GENERATION["stage_thinking"][stage]
    payload = {"model": model, "messages": [{k: deepcopy(v) for k, v in message.items() if k != "reasoning_content"}
                                             for message in messages],
               "max_tokens": GENERATION["stage_max_output_tokens"][stage], "thinking": {"type": thinking}}
    if thinking == "enabled":
        payload["reasoning_effort"] = GENERATION["reasoning_effort"]
    else:
        payload["temperature"] = GENERATION["temperature"]
    if tools:
        payload.update(tools=deepcopy(tools), tool_choice="auto")
    else:
        payload.update(response_format={"type": "json_object"})
        if stage == "final":
            payload.update(stream=True, stream_options={"include_usage": True})
    return payload


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


def read_completion_stream(response, record):
    """Consume the terminal answer and usage, never retain reasoning deltas."""
    if "text/event-stream" not in response.headers.get("content-type", ""):
        response.read()
        record["response_transport"] = "json"
        return response.json()
    record["response_transport"] = "sse"
    parts, data = [], []
    finish = None
    for line in response.iter_lines():
        if line.startswith(":"):
            record["stream_keepalives"] = record.get("stream_keepalives", 0) + 1
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
            continue
        if line or not data:
            continue
        event = "\n".join(data)
        data = []
        if event == "[DONE]":
            if finish not in {"stop", "length"}:
                raise ModelError("模型流式响应未正常完成")
            return {"model": record.get("model_returned"), "usage": record.get("usage"),
                    "system_fingerprint": record.get("system_fingerprint"),
                    "choices": [{"finish_reason": finish,
                                 "message": {"role": "assistant", "content": "".join(parts)}}]}
        chunk = json.loads(event)
        if not isinstance(chunk, dict):
            raise ModelError("模型流式响应格式无效")
        record["stream_chunks"] = record.get("stream_chunks", 0) + 1
        for key, source in (("usage", "usage"), ("model_returned", "model"),
                            ("system_fingerprint", "system_fingerprint")):
            if chunk.get(source) is not None:
                record[key] = chunk[source]
        choices = chunk.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ModelError("模型流式选择格式无效")
        if choices:
            choice = choices[0]
            delta = choice.get("delta", {})
            if not isinstance(delta, dict) or delta.get("tool_calls"):
                raise ModelError("最终流式答复不能调用工具")
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str) or finish is not None:
                    raise ModelError("模型流式文本格式无效")
                parts.append(content)
            if choice.get("finish_reason") is not None:
                if finish is not None:
                    raise ModelError("模型流式结束标记重复")
                finish = choice["finish_reason"]
                record["finish_reason"] = finish
    raise ModelError("模型流式响应中断，不能用于业务通过")


class DeepSeek:
    def __init__(self, max_calls=6):
        self.config = settings()
        self.model = self.config["DEEPSEEK_MODEL"]
        self.max_calls = max_calls
        self.calls = []

    def complete(self, messages, tools=None, stage=None):
        if not self.config["DEEPSEEK_API_KEY"]:
            raise ModelError("未配置 DeepSeek API Key；真实模型验收尚未开始")
        if len(self.calls) >= self.max_calls:
            raise ModelError("模型调用预算已耗尽，保留未完成项")
        payload = generation_request(self.model, messages, tools, stage=stage)
        record = {"request_hash": digest(payload), "request": deepcopy(payload),
                  "generation": deepcopy(GENERATION), "status": "started",
                  "stage": stage if stage is not None else "tool_review" if tools else "final"}
        self.calls.append(record)
        start = perf_counter()
        try:
            url = self.config["DEEPSEEK_BASE_URL"].rstrip("/") + "/chat/completions"
            kwargs = {"headers": {"Authorization": "Bearer " + self.config["DEEPSEEK_API_KEY"]},
                      "json": payload, "timeout": REQUEST_TIMEOUT_SECONDS}
            if payload.get("stream"):
                with httpx.stream("POST", url, **kwargs) as response:
                    if response.status_code != 200:
                        raise ModelError(f"DeepSeek HTTP {response.status_code}，本次模型步骤未完成")
                    body = read_completion_stream(response, record)
            else:
                response = httpx.post(url, **kwargs)
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
            if choice.get("finish_reason") not in ({"stop", "tool_calls"} if tools else {"stop"}):
                raise ModelError("模型响应未正常完成，不能用于业务通过")
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

    def complete(self, messages, tools=None, stage=None):
        payload = generation_request(self.model, messages, tools, stage=stage)
        request = {"model": self.model, "messages": payload["messages"], "tools": tools}
        if stage is not None:
            request["stage"] = stage
        key = digest(request)
        if key not in self.responses:
            raise ModelError("冻结请求未命中；禁止借用不同输入的模型产物")
        response = {k: deepcopy(v) for k, v in self.responses[key].items() if k != "reasoning_content"}
        self.calls.append({"request_hash": key, "request": deepcopy(request), "generation": deepcopy(GENERATION),
                           "status": "frozen", "usage": {}, "response": response,
                           "stage": stage if stage is not None else "tool_review" if tools else "final"})
        return response


def json_answer(message):
    try:
        result = json.loads(message.get("content") or "")
    except (ValueError, TypeError):
        raise ModelError("模型未返回有效 JSON") from None
    if not isinstance(result, dict):
        raise ModelError("模型 JSON 必须为对象")
    return result
