from __future__ import annotations

from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    title: str
    category: str
    department: str
    severity_hint: int
    keywords: tuple[str, ...]
    aliases: tuple[str, ...]
    content: str
    red_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["keywords"] = list(self.keywords)
        data["aliases"] = list(self.aliases)
        data["red_flags"] = list(self.red_flags)
        return data


MEDICAL_KNOWLEDGE_BASE: tuple[KnowledgeDocument, ...] = (
    KnowledgeDocument(
        id="kb-fever",
        title="发热与感染初步处理",
        category="infection",
        department="全科 / 感染科",
        severity_hint=2,
        keywords=("发热", "发烧", "体温", "感染", "寒战", "咳嗽", "流感", "病毒"),
        aliases=("高烧", "低烧", "浑身发冷"),
        red_flags=("体温超过39摄氏度", "持续超过3天", "意识改变", "严重脱水"),
        content=(
            "发热常见于病毒或细菌感染，也可见于炎症、自身免疫等情况。建议记录体温、"
            "持续时间、伴随症状和用药情况。若体温超过39摄氏度、持续超过3天，或伴意识改变、"
            "呼吸困难、皮疹、严重脱水，应及时就医。"
        ),
    ),
    KnowledgeDocument(
        id="kb-headache",
        title="头痛危险信号",
        category="neurology",
        department="神经内科",
        severity_hint=2,
        keywords=("头痛", "头疼", "偏头痛", "眩晕", "呕吐", "视物模糊", "高血压"),
        aliases=("脑袋疼", "头胀", "头晕"),
        red_flags=("突然剧烈头痛", "肢体无力", "言语不清", "发热颈项强直", "外伤后头痛"),
        content=(
            "多数头痛与紧张、睡眠不足、偏头痛、鼻窦炎或感染相关。突然发生的剧烈头痛、"
            "头痛伴肢体无力或言语不清、发热颈项强直、外伤后头痛、进行性加重头痛需要尽快就医。"
        ),
    ),
    KnowledgeDocument(
        id="kb-chest-pain",
        title="胸痛与心肺急症识别",
        category="emergency",
        department="急诊科 / 心内科",
        severity_hint=4,
        keywords=("胸痛", "胸闷", "心悸", "呼吸困难", "大汗", "濒死感", "左肩痛"),
        aliases=("胸口疼", "心口疼", "喘不上气", "气短"),
        red_flags=("胸痛伴呼吸困难", "大汗", "左肩或下颌放射痛", "晕厥", "濒死感"),
        content=(
            "胸痛需要首先排除心肌梗死、肺栓塞、气胸等急症。胸痛伴呼吸困难、大汗、恶心、"
            "左肩或下颌放射痛、晕厥时，应立即拨打急救电话或前往急诊。"
        ),
    ),
    KnowledgeDocument(
        id="kb-abdominal-pain",
        title="腹痛常见原因与就诊建议",
        category="digestive",
        department="消化内科 / 普外科",
        severity_hint=2,
        keywords=("腹痛", "肚子疼", "胃痛", "腹泻", "呕吐", "便血", "阑尾炎"),
        aliases=("胃疼", "肚子痛", "拉肚子"),
        red_flags=("剧烈腹痛", "右下腹持续疼痛", "黑便", "便血", "腹部板硬"),
        content=(
            "腹痛可能与胃肠炎、消化不良、胆囊疾病、泌尿系统问题或阑尾炎相关。剧烈腹痛、"
            "右下腹持续疼痛、腹痛伴发热呕吐、黑便或便血、腹部板硬，应及时就医。"
        ),
    ),
    KnowledgeDocument(
        id="kb-cough",
        title="咳嗽与呼吸道症状",
        category="respiratory",
        department="呼吸内科",
        severity_hint=2,
        keywords=("咳嗽", "咳痰", "喉咙痛", "鼻塞", "气短", "喘", "肺炎"),
        aliases=("嗓子疼", "喉咙不舒服", "有痰"),
        red_flags=("持续高热", "胸痛", "呼吸困难", "咯血", "血氧下降"),
        content=(
            "咳嗽常见于上呼吸道感染、过敏、支气管炎或肺炎。若咳嗽伴持续高热、胸痛、"
            "呼吸困难、咯血、血氧下降或基础肺病加重，应尽快就医。"
        ),
    ),
    KnowledgeDocument(
        id="kb-allergy",
        title="过敏反应与用药安全",
        category="allergy",
        department="皮肤科 / 急诊科",
        severity_hint=2,
        keywords=("过敏", "皮疹", "瘙痒", "荨麻疹", "药物", "喉头水肿", "呼吸困难"),
        aliases=("起疹子", "身上痒", "药疹"),
        red_flags=("呼吸困难", "喉咙紧缩", "面唇舌肿胀", "头晕低血压"),
        content=(
            "过敏可能表现为皮疹、瘙痒、眼鼻症状或胃肠不适。若出现呼吸困难、喉咙紧缩、"
            "面唇舌肿胀、头晕低血压，可能为严重过敏反应，应立即急诊处理。"
        ),
    ),
    KnowledgeDocument(
        id="kb-diabetes",
        title="糖尿病相关不适",
        category="chronic",
        department="内分泌科",
        severity_hint=2,
        keywords=("糖尿病", "血糖", "口渴", "多尿", "低血糖", "头晕", "乏力"),
        aliases=("高血糖", "低糖", "多饮"),
        red_flags=("意识异常", "反复低血糖", "明显脱水", "酮症酸中毒"),
        content=(
            "糖尿病患者出现乏力、口渴、多尿、恶心或意识异常时，需要关注血糖波动。"
            "低血糖可先补充含糖食物并复测血糖，意识不清或反复低血糖应急诊处理。"
        ),
    ),
    KnowledgeDocument(
        id="kb-hypertension",
        title="高血压与头晕胸闷",
        category="chronic",
        department="心内科",
        severity_hint=2,
        keywords=("高血压", "血压", "头晕", "胸闷", "心慌", "降压药"),
        aliases=("血压高", "血压升高", "心悸"),
        red_flags=("严重头痛", "视物模糊", "肢体无力", "胸痛", "意识异常"),
        content=(
            "高血压患者应规律监测血压并按医嘱用药。血压明显升高并伴胸痛、严重头痛、"
            "视物模糊、肢体无力或意识异常，需警惕高血压急症并及时就医。"
        ),
    ),
    KnowledgeDocument(
        id="kb-urinary",
        title="尿频尿痛与泌尿系统感染",
        category="urinary",
        department="泌尿外科 / 肾内科",
        severity_hint=2,
        keywords=("尿频", "尿痛", "尿急", "血尿", "腰痛", "发热", "泌尿感染"),
        aliases=("小便疼", "尿不舒服", "尿里有血"),
        red_flags=("高热寒战", "腰痛明显", "血尿", "孕期尿痛", "男性尿潴留"),
        content=(
            "尿频、尿急、尿痛常见于泌尿系统感染，也可能与结石、前列腺或妇科问题有关。"
            "若伴高热寒战、腰痛明显、血尿、孕期症状或男性排尿困难，应尽快就医。"
        ),
    ),
    KnowledgeDocument(
        id="kb-medication",
        title="用药咨询的安全边界",
        category="medication",
        department="药学门诊 / 对应专科",
        severity_hint=1,
        keywords=("用药", "药", "剂量", "副作用", "禁忌", "相互作用", "孕妇", "儿童"),
        aliases=("吃什么药", "能不能吃", "药量", "说明书"),
        red_flags=("过敏史", "孕期", "儿童", "肝肾功能异常", "多药同服"),
        content=(
            "用药建议需要结合年龄、体重、妊娠状态、过敏史、肝肾功能、既往病史和正在使用的药物。"
            "线上建议不能替代医生或药师处方，尤其是儿童、孕妇、老人和多病共存患者。"
        ),
    ),
)

SYNONYMS = {
    "发烧": "发热",
    "高烧": "发热",
    "低烧": "发热",
    "头疼": "头痛",
    "脑袋疼": "头痛",
    "肚子疼": "腹痛",
    "肚子痛": "腹痛",
    "胃疼": "胃痛",
    "喘不上气": "呼吸困难",
    "气短": "呼吸困难",
    "心慌": "心悸",
    "血压高": "高血压",
    "小便疼": "尿痛",
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def normalize_text(value: str) -> str:
    normalized = str(value or "").strip().lower()
    for source, target in SYNONYMS.items():
        normalized = normalized.replace(source, target)
    return normalized


def tokenize(value: str) -> list[str]:
    normalized = normalize_text(value)
    tokens = TOKEN_PATTERN.findall(normalized)

    for doc in MEDICAL_KNOWLEDGE_BASE:
        for keyword in doc.keywords + doc.aliases:
            if keyword in normalized and keyword not in tokens:
                tokens.append(keyword)

    return tokens


def search_knowledge(query: str, top_k: int = 3, categories: list[str] | None = None) -> list[dict]:
    from app.services.rag_service import RAGService

    return RAGService(top_k=top_k).retrieve(query=query, top_k=top_k, categories=categories)


def build_retrieval_reason(matches: list[str], doc: KnowledgeDocument) -> str:
    if not matches:
        return f"与{doc.department}相关的基础医学知识。"

    top_matches = "、".join(matches[:4])
    return f"命中关键词：{top_matches}；关联科室：{doc.department}。"
