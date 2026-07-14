from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any

from app.agents.agent_state import AgentState
from app.agents.decision_workflow import LLMSettings, MedicalDecisionWorkflow
from app.agents.toolbox import MedicalToolbox, ToolExecution
from app.core.config import settings
from app.services.hospital_recommender import HospitalRecommendationService
from app.services.knowledge_base import normalize_text
from app.services.rag_service import RAGService


EMERGENCY_SIGNALS = (
    "胸痛",
    "呼吸困难",
    "意识不清",
    "昏迷",
    "晕厥",
    "大出血",
    "剧烈腹痛",
    "抽搐",
    "口唇发紫",
    "严重过敏",
    "喉咙紧缩",
    "偏瘫",
    "言语不清",
    "咯血",
    "黑便",
)

SYMPTOM_RULES = (
    {"keyword": "胸痛", "department": "急诊科 / 心内科", "system": "心血管系统", "urgency": 4},
    {"keyword": "胸闷", "department": "心内科 / 呼吸内科", "system": "心肺系统", "urgency": 3},
    {"keyword": "呼吸困难", "department": "急诊科 / 呼吸内科", "system": "呼吸系统", "urgency": 4},
    {"keyword": "头痛", "department": "神经内科", "system": "神经系统", "urgency": 2},
    {"keyword": "头晕", "department": "神经内科 / 心内科", "system": "神经或循环系统", "urgency": 2},
    {"keyword": "发热", "department": "全科 / 感染科", "system": "感染相关", "urgency": 2},
    {"keyword": "咳嗽", "department": "呼吸内科", "system": "呼吸系统", "urgency": 2},
    {"keyword": "腹痛", "department": "消化内科 / 普外科", "system": "消化系统", "urgency": 2},
    {"keyword": "腹泻", "department": "消化内科", "system": "消化系统", "urgency": 2},
    {"keyword": "呕吐", "department": "消化内科", "system": "消化系统", "urgency": 2},
    {"keyword": "皮疹", "department": "皮肤科", "system": "皮肤 / 免疫系统", "urgency": 1},
    {"keyword": "过敏", "department": "皮肤科 / 急诊科", "system": "免疫系统", "urgency": 2},
    {"keyword": "尿频", "department": "泌尿外科 / 肾内科", "system": "泌尿系统", "urgency": 2},
    {"keyword": "尿痛", "department": "泌尿外科 / 肾内科", "system": "泌尿系统", "urgency": 2},
    {"keyword": "血糖", "department": "内分泌科", "system": "内分泌系统", "urgency": 2},
    {"keyword": "高血压", "department": "心内科", "system": "心血管系统", "urgency": 2},
)

INTENT_RULES = (
    ("medication", ("吃什么药", "用药", "药", "剂量", "副作用", "禁忌", "能不能吃")),
    ("report_interpretation", ("报告", "化验", "检查单", "指标", "ct", "核磁", "b超", "血常规")),
    ("follow_up", ("复查", "复诊", "术后", "恢复", "随访")),
    ("symptom_analysis", ("疼", "痛", "发热", "咳嗽", "头晕", "腹泻", "胸闷", "不舒服")),
)

DURATION_PATTERN = re.compile(r"(\d+)\s*(分钟|小时|天|日|周|个月|年)")
TEMPERATURE_PATTERN = re.compile(r"([3-4]\d(?:\.\d)?)\s*(度|℃|摄氏度)?")
BLOOD_PRESSURE_PATTERN = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})")


@dataclass
class MedicalAgent:
    rag_top_k: int = 3
    enable_llm: bool | None = None
    hospital_recommendation_service: HospitalRecommendationService | None = None
    toolbox: MedicalToolbox | None = None

    def __post_init__(self) -> None:
        self.rag_service = RAGService(top_k=self.rag_top_k, index_path=settings.rag_index_path)
        if self.hospital_recommendation_service is None:
            self.hospital_recommendation_service = HospitalRecommendationService()
        if self.toolbox is None:
            self.toolbox = MedicalToolbox(
                rag_service=self.rag_service,
                hospital_recommendation_service=self.hospital_recommendation_service,
            )
        self.decision_workflow = MedicalDecisionWorkflow(
            LLMSettings(
                enabled=settings.agent_llm_enabled if self.enable_llm is None else self.enable_llm,
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                max_tokens=settings.llm_max_tokens,
                timeout_seconds=settings.llm_timeout_seconds,
            )
        )

    def chat(
        self,
        message: str,
        conversation_history: list[dict] | None = None,
        user_context: dict | None = None,
    ) -> dict:
        history = conversation_history or []
        context = user_context or {}
        state = AgentState(user_context=redact_user_context(context))
        patient_profile = self._build_patient_profile(context)
        state.patient_profile = self._build_patient_profile(state.user_context)
        state.mark_step("patient_context", fields=[key for key, value in context.items() if value])

        intent = self.classify_intent(message)
        state.intent = intent
        state.mark_step("intent_classification", intent=intent)

        slot_state = self.extract_slots(message, history, context)
        state.slots = slot_state
        state.mark_step("slot_extraction", missing=[key for key, value in slot_state.items() if not value])

        analysis = self.analyze_symptoms(message, slot_state)
        state.symptom_analysis = analysis
        state.mark_step(
            "symptom_analysis",
            symptom_count=len(analysis["symptoms"]),
            urgency_level=analysis["urgency_level"],
        )

        categories = self._categories_for_intent(intent)
        knowledge_tool = self.toolbox.search_medical_knowledge(
            message=message,
            top_k=self.rag_top_k,
            categories=categories,
            conversation_history=history,
            user_context=context,
        )
        self._record_tool_execution(state, knowledge_tool)
        rag_context = knowledge_tool.payload
        sources = rag_context["retrieved_docs"]
        state.rag = {
            "query_expansion": rag_context["query_expansion"],
            "pipeline": rag_context["pipeline"],
            "source_count": len(sources),
            "retrieved_sources": [
                {
                    "id": source.get("id"),
                    "title": source.get("title"),
                    "score": source.get("score"),
                    "retrieval_reason": source.get("retrieval_reason"),
                }
                for source in sources
            ],
        }
        state.mark_step("rag_retrieval", source_count=len(sources), categories=categories)

        risk = self._assess_risk(analysis, slot_state, sources)
        state.risk = risk
        department = self._choose_department(analysis, sources, risk)
        state.department = department
        state.mark_step("risk_and_department", risk_level=risk["level"], department=department)

        decision = self._build_decision(intent, analysis, risk, sources, department)
        state.decision = decision
        follow_up_questions = self._build_follow_up_questions(intent, slot_state, analysis, risk)

        drug_safety = self.toolbox.check_drug_safety(
            message=message,
            intent=intent,
            slots=slot_state,
            user_context=context,
        )
        self._record_tool_execution(state, drug_safety)
        decision = self._merge_drug_safety_decision(decision, drug_safety.payload)

        hospital_tool = self.toolbox.recommend_hospitals(
            city=str(context.get("city") or ""),
            department=department,
            risk=risk,
            symptoms=analysis["symptoms"],
        )
        self._record_tool_execution(state, hospital_tool)
        hospital_recommendations = hospital_tool.payload
        state.mark_step(
            "mcp_toolbox",
            status=hospital_recommendations.get("status", "unknown"),
            invoked_tools=[tool["name"] for tool in state.tool_results],
        )

        follow_up_questions = self._merge_city_follow_up(
            follow_up_questions,
            context,
            hospital_recommendations,
        )
        workflow_result = self.decision_workflow.run(
            message=message,
            conversation_history=history,
            user_context=context,
            patient_profile=patient_profile,
            intent=intent,
            slots=slot_state,
            analysis=analysis,
            rag_context=rag_context,
            sources=sources,
            risk=risk,
            department=department,
            decision=decision,
            follow_up_questions=follow_up_questions,
        )
        decision = self._merge_llm_decision(decision, workflow_result.decision, risk)
        follow_up_questions = self._merge_follow_up_questions(
            follow_up_questions,
            workflow_result.decision,
            risk,
        )
        state.decision = decision
        state.follow_up_questions = follow_up_questions
        state.llm = workflow_result.metadata
        state.mark_step(
            "llm_orchestration",
            enabled=workflow_result.metadata.get("enabled"),
            used=workflow_result.metadata.get("used"),
        )

        result_analysis = {
            **analysis,
            "intent": intent,
            "stage": self._choose_stage(slot_state, risk),
            "patient_profile": patient_profile,
            "slots": slot_state,
            "risk": risk,
            "department": department,
            "hospital_recommendations": hospital_recommendations,
            "drug_safety": drug_safety.payload,
            "possible_causes": decision["possible_causes"],
            "suggested_examinations": decision["suggested_examinations"],
            "self_care_tips": decision["self_care_tips"],
            "needs_urgent_care": risk["level"] >= 4,
            "follow_up_questions": follow_up_questions,
            "context_summary": self._summarize_context(history, context),
            "llm": workflow_result.metadata,
            "rag": state.rag,
            "tool_results": state.tool_results,
            "agent_state": state.to_dict(),
            "run_id": state.run_id,
        }
        content = self._build_reply(result_analysis, sources)
        if workflow_result.content and not result_analysis["needs_urgent_care"]:
            content = self._finalize_llm_reply(workflow_result.content)
            content = self._append_hospital_recommendation_section(
                content,
                result_analysis["hospital_recommendations"],
            )

        return {
            "id": f"assistant_{time.time_ns()}",
            "role": "assistant",
            "content": content,
            "created_at": current_timestamp(),
            "analysis": result_analysis,
            "source_knowledge": [
                {
                    "title": source["title"],
                    "relevance": source["score"],
                    "matched_keywords": source["matched_keywords"],
                    "retrieval_reason": source["retrieval_reason"],
                    "content": source["content"],
                }
                for source in sources
            ],
        }

    def classify_intent(self, message: str) -> str:
        normalized = normalize_text(message)
        for intent, keywords in INTENT_RULES:
            if any(keyword in normalized for keyword in keywords):
                return intent
        return "general_health"

    def extract_slots(self, message: str, history: list[dict], context: dict) -> dict[str, Any]:
        normalized = normalize_text(message)
        duration = DURATION_PATTERN.search(normalized)
        temperature = TEMPERATURE_PATTERN.search(normalized)
        blood_pressure = BLOOD_PRESSURE_PATTERN.search(normalized)

        slots = {
            "chief_complaint": message.strip(),
            "duration": duration.group(0) if duration else None,
            "temperature": temperature.group(1) if temperature else None,
            "blood_pressure": blood_pressure.group(0) if blood_pressure else None,
            "age": context.get("age") or None,
            "gender": context.get("gender") or None,
            "allergies": context.get("allergies") or None,
            "chronic_diseases": context.get("chronic_diseases") or None,
            "city": context.get("city") or None,
            "medications": extract_after_keywords(normalized, ("正在吃", "服用", "吃了")),
            "history_mentions": self._recent_user_messages(history),
        }

        return slots

    def analyze_symptoms(self, message: str, slots: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_text(message)
        matched_rules = [
            rule
            for rule in SYMPTOM_RULES
            if rule["keyword"] in normalized and not is_negated(normalized, str(rule["keyword"]))
        ]
        emergency_matches = [
            signal
            for signal in EMERGENCY_SIGNALS
            if signal in normalized and not is_negated(normalized, signal)
        ]
        urgency = max(
            [4 if emergency_matches else 1] + [int(rule["urgency"]) for rule in matched_rules]
        )

        if slots.get("temperature"):
            try:
                if float(slots["temperature"]) >= 39:
                    urgency = max(urgency, 3)
            except ValueError:
                pass

        return {
            "symptoms": sorted({str(rule["keyword"]) for rule in matched_rules}),
            "duration": slots.get("duration"),
            "systems": sorted({str(rule["system"]) for rule in matched_rules}),
            "emergency_signals": emergency_matches,
            "urgency_level": urgency,
        }

    def _assess_risk(self, analysis: dict, slots: dict, sources: list[dict]) -> dict:
        level = analysis["urgency_level"]
        reasons = []

        if analysis["emergency_signals"]:
            reasons.append("识别到急症危险信号：" + "、".join(analysis["emergency_signals"]))

        if slots.get("temperature"):
            try:
                if float(slots["temperature"]) >= 39:
                    level = max(level, 3)
                    reasons.append("体温达到或超过39摄氏度")
            except ValueError:
                pass

        if analysis["emergency_signals"] and any(source["severity_hint"] >= 4 for source in sources):
            level = max(level, 4)
            reasons.append("检索到急症相关医学知识")

        if not reasons and level <= 1:
            reasons.append("暂未识别急症信号，但信息仍不完整")

        label = {1: "低风险", 2: "中低风险", 3: "需尽快就医评估", 4: "急症风险"}.get(level, "需评估")
        return {"level": level, "label": label, "reasons": reasons}

    def _choose_department(self, analysis: dict, sources: list[dict], risk: dict) -> str:
        if risk["level"] >= 4:
            return "急诊科"

        for rule in SYMPTOM_RULES:
            if rule["keyword"] in analysis["symptoms"]:
                return str(rule["department"])

        if sources:
            return str(sources[0]["department"])

        return "全科 / 普通内科"

    def _build_decision(self, intent: str, analysis: dict, risk: dict, sources: list[dict], department: str) -> dict:
        symptoms = set(analysis["symptoms"])

        if risk["level"] >= 4:
            return {
                "possible_causes": ["心肺或神经系统急症等需要优先排除的情况"],
                "suggested_examinations": ["急诊生命体征评估", "心电图/血氧/血压监测", "由医生决定进一步检查"],
                "self_care_tips": ["立即停止活动", "请身边人陪同", "尽快急诊或拨打急救电话"],
            }

        if intent == "medication":
            return {
                "possible_causes": ["用药问题需要结合诊断、年龄、过敏史和合并用药判断"],
                "suggested_examinations": ["核对药品通用名、剂量、频次", "确认过敏史、肝肾功能和妊娠状态"],
                "self_care_tips": ["不要自行叠加同类药", "处方药请遵医嘱", "出现过敏或严重不适立即停药并就医"],
            }

        if "发热" in symptoms or "咳嗽" in symptoms:
            return {
                "possible_causes": ["上呼吸道感染", "流感或其他病毒感染", "细菌感染可能"],
                "suggested_examinations": ["体温记录", "血常规 / C反应蛋白", "必要时胸片或病原检测"],
                "self_care_tips": ["补充水分", "休息并观察体温", "避免自行叠加多种退烧药"],
            }

        if "头痛" in symptoms or "头晕" in symptoms:
            return {
                "possible_causes": ["紧张性头痛或偏头痛", "感染或睡眠不足相关不适", "血压波动或神经系统问题需排除"],
                "suggested_examinations": ["血压测量", "神经系统体征评估", "必要时由医生决定影像检查"],
                "self_care_tips": ["规律休息", "避免饮酒和熬夜", "记录发作诱因和持续时间"],
            }

        if "腹痛" in symptoms or "腹泻" in symptoms or "呕吐" in symptoms:
            return {
                "possible_causes": ["胃肠炎或消化不良", "胆囊/阑尾等腹部疾病", "泌尿或妇科相关问题"],
                "suggested_examinations": ["腹部查体", "血常规", "必要时腹部超声或尿检"],
                "self_care_tips": ["清淡饮食", "补液", "避免饮酒和油腻食物"],
            }

        if sources:
            return {
                "possible_causes": [str(source["title"]) for source in sources[:3]],
                "suggested_examinations": ["补充症状持续时间和伴随症状", f"建议到{department}进一步评估"],
                "self_care_tips": ["记录症状变化", "避免自行使用不确定药物", "症状加重及时就医"],
            }

        return {
            "possible_causes": ["信息不足，暂无法可靠判断"],
            "suggested_examinations": ["补充年龄、持续时间、部位、诱因、伴随症状和既往病史"],
            "self_care_tips": ["先记录症状变化", "如出现危险信号及时就医"],
        }

    def _build_follow_up_questions(self, intent: str, slots: dict, analysis: dict, risk: dict) -> list[str]:
        if risk["level"] >= 4:
            return ["现在是否有人陪同？", "是否已经准备前往急诊或拨打急救电话？"]

        questions = []
        if not slots.get("duration"):
            questions.append("症状持续多久了？")
        if not analysis["symptoms"]:
            questions.append("主要不舒服的部位和表现是什么？")
        if not slots.get("age"):
            questions.append("患者年龄是多少？")
        if intent == "medication" and not slots.get("allergies"):
            questions.append("是否有药物过敏史、肝肾功能异常或正在服用其他药？")

        questions.append("是否伴有胸痛、呼吸困难、意识改变、持续高热、便血或剧烈疼痛？")
        return questions[:4]

    def _build_reply(self, analysis: dict, sources: list[dict]) -> str:
        if analysis["needs_urgent_care"]:
            opening = "我识别到急症风险信号，请优先线下就医或拨打急救电话。"
        else:
            opening = "我先按问诊 Agent 流程给你做初步分诊和建议，不能替代医生面诊。"

        systems = "、".join(analysis["systems"]) or "暂未明确，需要继续补充信息"
        source_lines = [
            f"- {source['title']}（{source['retrieval_reason']}）" for source in sources[:3]
        ] or ["- 暂无高匹配知识来源"]
        drug_safety_lines = self._format_drug_safety(analysis.get("drug_safety") or {})
        hospital_lines = self._format_hospital_recommendations(analysis["hospital_recommendations"])

        return "\n".join(
            [
                opening,
                "",
                f"【意图识别】{intent_label(analysis['intent'])}",
                f"【问诊阶段】{stage_label(analysis['stage'])}",
                f"【初步判断】{systems}",
                f"【风险分层】{analysis['risk']['label']}（{analysis['risk']['level']}/4）",
                f"【建议科室】{analysis['department']}",
                "",
                "【可能原因】",
                *[f"- {item}" for item in analysis["possible_causes"]],
                "",
                "【建议检查/下一步】",
                *[f"- {item}" for item in analysis["suggested_examinations"]],
                "",
                "【护理与安全建议】",
                *[f"- {item}" for item in analysis["self_care_tips"]],
                "",
                "【用药安全核对】",
                *drug_safety_lines,
                "",
                "【参考知识】",
                *source_lines,
                "",
                "【医院推荐】",
                *hospital_lines,
                "",
                "【需要补充】",
                *[f"- {item}" for item in analysis["follow_up_questions"]],
                "",
                "免责声明：以上内容仅供健康信息参考，不能替代专业医生诊断。若症状加重或出现危险信号，请及时就医。",
            ]
        )

    def _choose_stage(self, slots: dict, risk: dict) -> str:
        if risk["level"] >= 4:
            return "urgent_guidance"

        required = ("duration", "age")
        missing = [key for key in required if not slots.get(key)]
        return "history_collection" if missing else "assessment"

    def _categories_for_intent(self, intent: str) -> list[str] | None:
        if intent == "medication":
            return ["medication", "allergy", "chronic"]
        if intent == "report_interpretation":
            return ["chronic", "infection", "respiratory", "digestive", "urinary"]
        return None

    def _build_patient_profile(self, context: dict) -> dict:
        return {
            "age": context.get("age") or "未填写",
            "gender": context.get("gender") or "未填写",
            "allergies": context.get("allergies") or "未填写",
            "chronic_diseases": context.get("chronic_diseases") or "未填写",
            "city": context.get("city") or "未填写",
        }

    def _recent_user_messages(self, history: list[dict]) -> list[str]:
        return [
            str(message.get("content", ""))
            for message in history[-6:]
            if message.get("role") == "user"
        ]

    def _summarize_context(self, history: list[dict], context: dict) -> dict:
        profile = ", ".join(f"{key}:{value}" for key, value in context.items() if value)
        return {
            "profile": profile or "暂无用户基础信息",
            "previous_turns": len(history),
        }

    def _merge_llm_decision(self, decision: dict, llm_decision: dict | None, risk: dict) -> dict:
        if not llm_decision:
            return decision

        merged = dict(decision)
        for key in ("possible_causes", "suggested_examinations", "self_care_tips"):
            values = llm_decision.get(key) or []
            if values and not (risk["level"] >= 4 and key == "self_care_tips"):
                merged[key] = values

        if llm_decision.get("safety_notes"):
            merged["self_care_tips"] = list(
                dict.fromkeys([*merged.get("self_care_tips", []), *llm_decision["safety_notes"]])
            )[:6]

        return merged

    def _merge_follow_up_questions(
        self,
        current_questions: list[str],
        llm_decision: dict | None,
        risk: dict,
    ) -> list[str]:
        if risk["level"] >= 4 or not llm_decision:
            return current_questions

        llm_questions = llm_decision.get("follow_up_questions") or []
        if not llm_questions:
            return current_questions

        return list(dict.fromkeys([*llm_questions, *current_questions]))[:4]

    def _record_tool_execution(self, state: AgentState, tool: ToolExecution) -> None:
        state.record_tool_result(
            name=tool.name,
            status=tool.status,
            input_summary=tool.input_summary,
            output_summary=tool.output_summary,
        )

    def _merge_drug_safety_decision(self, decision: dict, drug_safety: dict) -> dict:
        if drug_safety.get("status") != "available":
            return decision

        merged = dict(decision)
        warnings = drug_safety.get("warnings") or []
        recommendations = drug_safety.get("recommendations") or []
        if warnings or recommendations:
            merged["self_care_tips"] = list(
                dict.fromkeys(
                    [
                        *merged.get("self_care_tips", []),
                        *warnings,
                        *recommendations,
                    ]
                )
            )[:8]
        return merged

    def _finalize_llm_reply(self, content: str) -> str:
        disclaimer = "免责声明：以上内容仅供健康信息参考，不能替代专业医生诊断。若症状加重或出现危险信号，请及时就医。"
        if "免责声明" in content or "不能替代" in content:
            return content
        return f"{content.rstrip()}\n\n{disclaimer}"

    def _append_hospital_recommendation_section(
        self,
        content: str,
        hospital_recommendations: dict,
    ) -> str:
        if "【医院推荐】" in content:
            return content
        hospital_lines = self._format_hospital_recommendations(hospital_recommendations)
        return "\n".join([content.rstrip(), "", "【医院推荐】", *hospital_lines])

    def _format_drug_safety(self, drug_safety: dict) -> list[str]:
        if drug_safety.get("status") == "skipped":
            return ["- 本轮不是用药咨询，未调用药物安全工具。"]
        if drug_safety.get("status") != "available":
            return ["- 暂无可用的药物安全核对结果。"]

        lines = []
        warnings = drug_safety.get("warnings") or []
        missing_context = drug_safety.get("missing_context") or []
        recommendations = drug_safety.get("recommendations") or []

        if warnings:
            lines.extend(f"- {warning}" for warning in warnings)
        if missing_context:
            lines.append(f"- 用药判断仍缺少：{'、'.join(missing_context)}。")
        if recommendations:
            lines.extend(f"- {item}" for item in recommendations[:2])
        return lines or ["- 未识别到明确药物禁忌，但仍需按说明书或医嘱用药。"]

    def _merge_city_follow_up(
        self,
        current_questions: list[str],
        context: dict,
        hospital_recommendations: dict,
    ) -> list[str]:
        if context.get("city") or hospital_recommendations.get("status") != "missing_city":
            return current_questions
        return list(dict.fromkeys([*current_questions, "你当前所在城市是哪里？"]))[:5]

    def _format_hospital_recommendations(self, hospital_recommendations: dict) -> list[str]:
        recommendations = hospital_recommendations.get("recommendations") or []
        if recommendations:
            lines = []
            for item in recommendations[:3]:
                address = f"；地址：{item['address']}" if item.get("address") else ""
                lines.append(f"- {item['name']}（{item['matched_department']}）{address}；{item['reason']}")
            lines.append("- 请核实医院挂号、科室开放和急诊接诊情况；急症请优先线下急诊或拨打急救电话。")
            return lines

        status = hospital_recommendations.get("status")
        if status == "missing_city":
            return ["- 提供当前所在城市后，我可以结合建议科室给出该城市医院候选。"]
        if status == "unavailable":
            return ["- 医院推荐服务暂不可用，请根据建议科室自行核实当地医院挂号与接诊信息。"]
        return ["- 暂未检索到合适医院候选，请核实当地医院挂号与接诊信息。"]


def current_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def redact_user_context(context: dict[str, Any]) -> dict[str, Any]:
    allowed = ("age", "gender", "city")
    redacted = {key: context.get(key) for key in allowed if context.get(key)}
    for key in ("allergies", "chronic_diseases"):
        if context.get(key):
            redacted[key] = "provided"
    return redacted


def extract_after_keywords(message: str, keywords: tuple[str, ...]) -> str | None:
    for keyword in keywords:
        index = message.find(keyword)
        if index >= 0:
            return message[index : index + 40]
    return None


def intent_label(intent: str) -> str:
    return {
        "symptom_analysis": "症状分析",
        "medication": "用药咨询",
        "report_interpretation": "检查报告解读",
        "follow_up": "复查/随访咨询",
        "general_health": "一般健康咨询",
    }.get(intent, intent)


def stage_label(stage: str) -> str:
    return {
        "history_collection": "病史采集",
        "assessment": "初步评估",
        "urgent_guidance": "急症引导",
    }.get(stage, stage)


def is_negated(message: str, keyword: str) -> bool:
    negation_patterns = (
        f"没有{keyword}",
        f"无{keyword}",
        f"不伴{keyword}",
        f"未见{keyword}",
        f"否认{keyword}",
        f"不是{keyword}",
        f"并非{keyword}",
    )
    return any(pattern in message for pattern in negation_patterns)
