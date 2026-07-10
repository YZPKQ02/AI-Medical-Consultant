from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
import json
import threading
from typing import Any
from urllib import error, parse, request

from app.core.config import settings

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:  # pragma: no cover - exercised only when optional runtime deps are absent.
    ClientSession = None
    streamablehttp_client = None


AMAP_PLACE_TEXT_ENDPOINT = "https://restapi.amap.com/v3/place/text"


DEPARTMENT_TERMS = {
    "急诊科": ("急诊", "综合医院", "急救"),
    "心内科": ("心内科", "心血管", "心脏"),
    "呼吸内科": ("呼吸内科", "呼吸"),
    "神经内科": ("神经内科", "神经"),
    "感染科": ("感染科", "发热门诊"),
    "消化内科": ("消化内科", "消化"),
    "普外科": ("普外科", "外科"),
    "皮肤科": ("皮肤科", "皮肤"),
    "泌尿外科": ("泌尿外科", "泌尿"),
    "肾内科": ("肾内科", "肾病"),
    "内分泌科": ("内分泌科", "内分泌"),
    "药学门诊": ("药学门诊", "药学"),
    "全科": ("全科", "综合医院"),
    "普通内科": ("普通内科", "内科"),
}


@dataclass(frozen=True)
class HospitalRecommendationSettings:
    enabled: bool = False
    amap_mcp_url: str = ""
    amap_web_service_key: str = ""
    limit: int = 5
    timeout_seconds: int = 5


class AmapHospitalSearchClient:
    def __init__(self, api_key: str, timeout_seconds: int = 5):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.last_request_params: dict[str, Any] | None = None

    def search(self, *, city: str, department: str, urgency_level: int, offset: int = 20) -> list[dict]:
        department_term = choose_department_term(department, urgency_level)
        params = {
            "key": self.api_key,
            "keywords": f"{department_term} 医院",
            "city": city,
            "citylimit": "true",
            "offset": str(offset),
            "page": "1",
            "extensions": "all",
            "output": "JSON",
        }
        self.last_request_params = dict(params)
        url = f"{AMAP_PLACE_TEXT_ENDPOINT}?{parse.urlencode(params)}"

        req = request.Request(url, method="GET")
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"AMap HTTP {exc.code}: {detail[:200]}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"AMap network error: {exc.reason}") from exc

        if str(payload.get("status")) != "1":
            raise RuntimeError(payload.get("info") or "AMap search failed")

        pois = payload.get("pois") or []
        if not isinstance(pois, list):
            return []
        return [poi for poi in pois if isinstance(poi, dict)]


class AmapMcpHospitalSearchClient:
    """Adapter for AMap's official Streamable HTTP MCP server."""

    def __init__(self, mcp_url: str, timeout_seconds: int = 5):
        self.mcp_url = mcp_url
        self.timeout_seconds = timeout_seconds
        self.last_tool_name: str | None = None
        self.last_tool_arguments: dict[str, Any] | None = None

    def search(self, *, city: str, department: str, urgency_level: int, offset: int = 20) -> list[dict]:
        if ClientSession is None or streamablehttp_client is None:
            raise RuntimeError("MCP Python SDK is not installed")
        return run_async_from_sync(
            lambda: self._search_async(
                city=city,
                department=department,
                urgency_level=urgency_level,
                offset=offset,
            )
        )

    async def _search_async(self, *, city: str, department: str, urgency_level: int, offset: int) -> list[dict]:
        department_term = choose_department_term(department, urgency_level)
        timeout = max(1, int(self.timeout_seconds or 5))

        async with streamablehttp_client(
            self.mcp_url,
            timeout=timeout,
            sse_read_timeout=timeout,
        ) as (read_stream, write_stream, _):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=timeout),
            ) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                tool = choose_mcp_search_tool(getattr(tools_result, "tools", []) or [])
                if tool is None:
                    raise RuntimeError("No compatible AMap MCP search tool found")

                arguments = build_mcp_search_arguments(
                    tool=tool,
                    city=city,
                    department_term=department_term,
                    offset=offset,
                )
                self.last_tool_name = get_tool_name(tool)
                self.last_tool_arguments = dict(arguments)
                result = await session.call_tool(
                    self.last_tool_name,
                    arguments=arguments,
                    read_timeout_seconds=timedelta(seconds=timeout),
                )

        return extract_mcp_pois(result)


class HospitalRecommendationService:
    def __init__(
        self,
        config: HospitalRecommendationSettings | None = None,
        amap_client: Any | None = None,
    ):
        self.config = config or HospitalRecommendationSettings(
            enabled=settings.hospital_recommender_enabled,
            amap_mcp_url=settings.amap_mcp_url,
            amap_web_service_key=settings.amap_web_service_key,
            limit=settings.hospital_recommender_limit,
            timeout_seconds=settings.hospital_recommender_timeout_seconds,
        )
        self.amap_client = amap_client or self._build_default_client()

    def _build_default_client(self) -> Any | None:
        if self.config.amap_mcp_url:
            return AmapMcpHospitalSearchClient(
                mcp_url=self.config.amap_mcp_url,
                timeout_seconds=self.config.timeout_seconds,
            )
        if self.config.amap_web_service_key:
            return AmapHospitalSearchClient(
                api_key=self.config.amap_web_service_key,
                timeout_seconds=self.config.timeout_seconds,
            )
        return None

    def recommend(
        self,
        *,
        city: str,
        department: str,
        urgency_level: int,
        symptoms: list[str] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        normalized_city = str(city or "").strip()
        if not normalized_city:
            return {
                "status": "missing_city",
                "city": "",
                "department": department,
                "recommendations": [],
                "fallback_reason": "city is missing",
            }

        if not self.config.enabled:
            return unavailable_result(
                city=normalized_city,
                department=department,
                fallback_reason="hospital recommender is disabled",
            )

        if not self.config.amap_mcp_url and not self.config.amap_web_service_key:
            return unavailable_result(
                city=normalized_city,
                department=department,
                fallback_reason="AMAP_MCP_URL or AMAP_WEB_SERVICE_KEY is missing",
            )
        if self.amap_client is None:
            return unavailable_result(
                city=normalized_city,
                department=department,
                fallback_reason="hospital recommender client is unavailable",
            )

        result_limit = max(1, min(int(limit or self.config.limit or 5), 10))

        try:
            pois = self.amap_client.search(
                city=normalized_city,
                department=department,
                urgency_level=urgency_level,
            )
        except Exception as exc:
            return unavailable_result(
                city=normalized_city,
                department=department,
                fallback_reason=str(exc),
            )

        recommendations = rank_hospitals(
            pois=pois,
            department=department,
            urgency_level=urgency_level,
            limit=result_limit,
        )
        return {
            "status": "available" if recommendations else "empty",
            "city": normalized_city,
            "department": department,
            "recommendations": recommendations,
            "fallback_reason": None,
        }


def recommend_hospitals(
    *,
    city: str,
    department: str,
    urgency_level: int,
    symptoms: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    service = HospitalRecommendationService()
    return service.recommend(
        city=city,
        department=department,
        urgency_level=urgency_level,
        symptoms=symptoms,
        limit=limit,
    )


def unavailable_result(*, city: str, department: str, fallback_reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "city": city,
        "department": department,
        "recommendations": [],
        "fallback_reason": fallback_reason,
    }


def run_async_from_sync(coro_factory):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro_factory())
        except Exception as exc:  # pragma: no cover - depends on async caller context.
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def choose_mcp_search_tool(tools: list[Any]) -> Any | None:
    scored_tools = []
    for index, tool in enumerate(tools):
        name = get_tool_name(tool)
        description = get_tool_description(tool)
        text = f"{name} {description}".lower()
        score = 0

        if name in {"maps_text_search", "amap_maps_text_search", "place_text", "poi_search"}:
            score += 100
        if any(token in text for token in ("text_search", "place", "poi", "keyword", "search")):
            score += 50
        if any(token in text for token in ("地点", "位置", "搜索", "关键字", "兴趣点")):
            score += 50
        if any(token in text for token in ("route", "weather", "geocode", "逆地理", "路径", "天气")):
            score -= 40

        if score > 0:
            scored_tools.append((score, -index, tool))

    if not scored_tools:
        return None
    scored_tools.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return scored_tools[0][2]


def build_mcp_search_arguments(*, tool: Any, city: str, department_term: str, offset: int) -> dict[str, Any]:
    schema = get_tool_input_schema(tool)
    properties = schema.get("properties") if isinstance(schema, dict) else {}
    properties = properties if isinstance(properties, dict) else {}
    keyword = f"{department_term} 医院"

    if not properties:
        return {
            "keywords": keyword,
            "city": city,
            "citylimit": True,
            "offset": offset,
            "page": 1,
            "extensions": "all",
        }

    arguments: dict[str, Any] = {}
    for key, spec in properties.items():
        key_lower = str(key).lower()
        value: Any | None = None

        if key_lower in {"keywords", "keyword", "query", "q", "search"}:
            value = keyword
        elif key_lower in {"city", "cityname"}:
            value = city
        elif key_lower in {"citylimit", "city_limit"}:
            value = True if schema_property_type(spec) == "boolean" else "true"
        elif key_lower in {"offset", "limit", "size"}:
            value = offset
        elif key_lower in {"page", "page_num", "pagenum"}:
            value = 1
        elif key_lower == "extensions":
            value = "all"
        elif key_lower in {"types", "type"}:
            value = "医疗保健服务"

        if value is not None:
            arguments[key] = value

    if not any(key.lower() in {"keywords", "keyword", "query", "q", "search"} for key in arguments):
        arguments["keywords"] = keyword
    if not any(key.lower() in {"city", "cityname"} for key in arguments):
        arguments["city"] = city

    return arguments


def extract_mcp_pois(result: Any) -> list[dict]:
    payload = extract_mcp_payload(result)
    pois = find_pois_in_payload(payload)
    return [poi for poi in pois if isinstance(poi, dict)]


def extract_mcp_payload(result: Any) -> Any:
    direct = result.model_dump() if hasattr(result, "model_dump") else result
    content = direct.get("content") if isinstance(direct, dict) else getattr(result, "content", None)

    if isinstance(content, list) and content:
        for item in content:
            item_payload = item.model_dump() if hasattr(item, "model_dump") else item
            text = None
            if isinstance(item_payload, dict):
                text = item_payload.get("text")
                if text is None and "data" in item_payload:
                    return item_payload["data"]
            else:
                text = getattr(item, "text", None)
            if isinstance(text, str):
                return parse_jsonish_text(text)

    return direct


def parse_jsonish_text(text: str) -> Any:
    cleaned = text.strip()
    if not cleaned:
        return {}
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = min((pos for pos in [cleaned.find("{"), cleaned.find("[")] if pos >= 0), default=-1)
        if start >= 0:
            try:
                return json.loads(cleaned[start:])
            except json.JSONDecodeError:
                return {"text": cleaned}
    return {"text": cleaned}


def find_pois_in_payload(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    for key in ("pois", "poi", "results", "result", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        nested = find_pois_in_payload(value)
        if nested:
            return nested
    return []


def get_tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or (tool.get("name") if isinstance(tool, dict) else "")).strip()


def get_tool_description(tool: Any) -> str:
    if isinstance(tool, dict):
        return str(tool.get("description") or "")
    return str(getattr(tool, "description", "") or "")


def get_tool_input_schema(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        schema = tool.get("inputSchema") or tool.get("input_schema") or {}
    else:
        schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
    if hasattr(schema, "model_dump"):
        schema = schema.model_dump()
    return schema if isinstance(schema, dict) else {}


def schema_property_type(spec: Any) -> str:
    if isinstance(spec, dict):
        value = spec.get("type")
        if isinstance(value, list):
            return str(value[0] if value else "")
        return str(value or "")
    return ""


def choose_department_term(department: str, urgency_level: int) -> str:
    normalized = str(department or "")
    if urgency_level >= 4:
        return "急诊"

    for key in DEPARTMENT_TERMS:
        if key in normalized:
            return DEPARTMENT_TERMS[key][0]

    if "/" in normalized:
        return normalized.split("/", 1)[0].strip()
    return normalized.strip() or "综合医院"


def rank_hospitals(
    *,
    pois: list[dict],
    department: str,
    urgency_level: int,
    limit: int,
) -> list[dict[str, Any]]:
    terms = department_match_terms(department, urgency_level)
    ranked = []

    for index, poi in enumerate(pois):
        name = str(poi.get("name") or "").strip()
        if not name:
            continue

        address = stringify_field(poi.get("address"))
        district = stringify_field(poi.get("adname") or poi.get("district"))
        adcode = stringify_field(poi.get("adcode"))
        location = stringify_field(poi.get("location"))
        poi_type = stringify_field(poi.get("type"))
        searchable = " ".join([name, address, district, poi_type])
        matched_terms = [term for term in terms if term and term in searchable]

        score = 50 + max(0, 20 - index)
        score += len(matched_terms) * 12
        reason_parts = []

        if matched_terms:
            reason_parts.append(f"匹配{department}相关关键词：{'、'.join(matched_terms[:3])}")

        if urgency_level >= 4:
            urgent_terms = ("急诊", "急救", "综合医院", "三甲")
            urgent_matches = [term for term in urgent_terms if term in searchable]
            if urgent_matches:
                score += 30
                reason_parts.append("急症风险下优先急诊/线下急诊相关机构")
            else:
                reason_parts.append("急症风险下请优先急诊或拨打急救电话，并核实该机构急诊接诊能力")

        if not reason_parts:
            reason_parts.append("按城市和建议科室检索到的医院候选，请核实挂号与接诊信息")

        ranked.append(
            {
                "name": name,
                "address": address,
                "district": district,
                "adcode": adcode,
                "location": location,
                "poi_id": stringify_field(poi.get("id")),
                "matched_department": department,
                "reason": "；".join(reason_parts),
                "score": score,
                "source": "amap",
                "_rank": index,
            }
        )

    ranked.sort(key=lambda item: (-int(item["score"]), int(item["_rank"])))
    return [{key: value for key, value in item.items() if key != "_rank"} for item in ranked[:limit]]


def department_match_terms(department: str, urgency_level: int) -> tuple[str, ...]:
    terms: list[str] = []
    if urgency_level >= 4:
        terms.extend(["急诊", "急救", "综合医院", "三甲"])

    normalized = str(department or "")
    for key, values in DEPARTMENT_TERMS.items():
        if key in normalized:
            terms.extend(values)

    for part in normalized.replace("／", "/").split("/"):
        cleaned = part.strip()
        if cleaned:
            terms.append(cleaned)

    return tuple(dict.fromkeys(terms))


def stringify_field(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    if value is None:
        return ""
    return str(value)
