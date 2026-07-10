import json
import unittest
from unittest.mock import patch

from app.services.hospital_recommender import (
    AmapHospitalSearchClient,
    AmapMcpHospitalSearchClient,
    HospitalRecommendationService,
    HospitalRecommendationSettings,
    build_mcp_search_arguments,
    choose_mcp_search_tool,
    extract_mcp_pois,
    rank_hospitals,
)


class MockAmapClient:
    def __init__(self, pois=None, error=None):
        self.pois = pois or []
        self.error = error
        self.calls = []

    def search(self, *, city, department, urgency_level, offset=20):
        self.calls.append(
            {
                "city": city,
                "department": department,
                "urgency_level": urgency_level,
                "offset": offset,
            }
        )
        if self.error:
            raise self.error
        return self.pois


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class HospitalRecommenderTests(unittest.TestCase):
    def test_service_returns_ranked_recommendations(self):
        client = MockAmapClient(
            pois=[
                {
                    "id": "1",
                    "name": "杭州普通门诊",
                    "address": "A路",
                    "adname": "西湖区",
                    "adcode": "330106",
                    "location": "120,30",
                    "type": "医疗保健服务",
                },
                {
                    "id": "2",
                    "name": "杭州某三甲医院急诊中心",
                    "address": "B路",
                    "adname": "上城区",
                    "adcode": "330102",
                    "location": "120,31",
                    "type": "综合医院",
                },
            ]
        )
        service = HospitalRecommendationService(
            config=HospitalRecommendationSettings(
                enabled=True,
                amap_mcp_url="",
                amap_web_service_key="mock-key",
                limit=2,
                timeout_seconds=5,
            ),
            amap_client=client,
        )

        result = service.recommend(
            city="杭州",
            department="急诊科",
            urgency_level=4,
            symptoms=["胸痛"],
        )

        self.assertEqual(result["status"], "available")
        self.assertEqual(result["recommendations"][0]["name"], "杭州某三甲医院急诊中心")
        self.assertIn("优先急诊", result["recommendations"][0]["reason"])
        self.assertEqual(client.calls[0]["city"], "杭州")
        self.assertEqual(client.calls[0]["department"], "急诊科")
        self.assertNotIn("symptoms", client.calls[0])

    def test_service_failure_does_not_raise(self):
        service = HospitalRecommendationService(
            config=HospitalRecommendationSettings(
                enabled=True,
                amap_mcp_url="",
                amap_web_service_key="mock-key",
                limit=2,
                timeout_seconds=5,
            ),
            amap_client=MockAmapClient(error=RuntimeError("network down")),
        )

        result = service.recommend(
            city="上海",
            department="呼吸内科",
            urgency_level=2,
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("network down", result["fallback_reason"])

    def test_amap_client_sends_only_city_and_department_search_terms(self):
        client = AmapHospitalSearchClient(api_key="secret", timeout_seconds=1)

        with patch("app.services.hospital_recommender.request.urlopen") as urlopen:
            urlopen.return_value = FakeResponse({"status": "1", "pois": []})
            client.search(city="北京", department="心内科", urgency_level=2)

        params = client.last_request_params
        self.assertEqual(params["city"], "北京")
        self.assertIn("心内科", params["keywords"])
        self.assertNotIn("age", params)
        self.assertNotIn("allergies", params)
        self.assertNotIn("history", params)

    def test_service_prefers_official_amap_mcp_url(self):
        service = HospitalRecommendationService(
            config=HospitalRecommendationSettings(
                enabled=True,
                amap_mcp_url="https://mcp.amap.com/mcp?key=mock-key",
                amap_web_service_key="legacy-key",
                limit=2,
                timeout_seconds=5,
            )
        )

        self.assertIsInstance(service.amap_client, AmapMcpHospitalSearchClient)

    def test_mcp_arguments_send_only_city_and_department_search_terms(self):
        tool = {
            "name": "maps_text_search",
            "description": "地点关键词搜索",
            "inputSchema": {
                "properties": {
                    "keywords": {"type": "string"},
                    "city": {"type": "string"},
                    "citylimit": {"type": "boolean"},
                    "offset": {"type": "integer"},
                    "page": {"type": "integer"},
                    "extensions": {"type": "string"},
                }
            },
        }

        arguments = build_mcp_search_arguments(
            tool=tool,
            city="广州",
            department_term="心内科",
            offset=20,
        )

        self.assertEqual(arguments["city"], "广州")
        self.assertEqual(arguments["keywords"], "心内科 医院")
        self.assertIs(arguments["citylimit"], True)
        self.assertNotIn("age", arguments)
        self.assertNotIn("allergies", arguments)
        self.assertNotIn("symptoms", arguments)
        self.assertNotIn("history", arguments)

    def test_mcp_tool_discovery_and_payload_parsing(self):
        weather_tool = {
            "name": "maps_weather",
            "description": "天气查询",
            "inputSchema": {"properties": {"city": {"type": "string"}}},
        }
        search_tool = {
            "name": "maps_text_search",
            "description": "根据关键字搜索 POI 地点",
            "inputSchema": {"properties": {"keywords": {"type": "string"}}},
        }

        self.assertEqual(choose_mcp_search_tool([weather_tool, search_tool]), search_tool)

        result = {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "pois": [
                                {
                                    "id": "p1",
                                    "name": "广州心血管医院",
                                    "address": "中山路",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                }
            ]
        }

        pois = extract_mcp_pois(result)
        self.assertEqual(pois[0]["id"], "p1")

    def test_rank_hospitals_keeps_limit_and_required_fields(self):
        result = rank_hospitals(
            pois=[
                {
                    "id": "a",
                    "name": "北京心血管医院",
                    "address": "甲路",
                    "adname": "东城区",
                    "adcode": "110101",
                    "location": "116,39",
                    "type": "专科医院",
                }
            ],
            department="心内科",
            urgency_level=2,
            limit=1,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source"], "amap")
        self.assertEqual(result[0]["matched_department"], "心内科")
        self.assertIn("poi_id", result[0])


if __name__ == "__main__":
    unittest.main()
