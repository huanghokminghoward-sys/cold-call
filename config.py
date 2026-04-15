WHISPER_MODELS = {
    "openai/whisper-large-v3": "Whisper Large v3（最準確）",
    "openai/whisper-medium": "Whisper Medium（平衡）",
    "openai/whisper-small": "Whisper Small（快速）",
}

CANTONESE_SPECIFIC_MODELS = set()

GEMINI_MODELS = {
    "gemini-1.5-flash": "Gemini 1.5 Flash（快速）",
    "gemini-1.5-pro": "Gemini 1.5 Pro（最強）",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
}

ANALYSIS_SECTIONS = [
    ("opening", "🚪 開場白"),
    ("needs_discovery", "🔍 需求發掘"),
    ("product_presentation", "📣 產品介紹"),
    ("objection_handling", "🛡️ 異議處理"),
    ("closing", "🎯 成交技巧"),
    ("language_quality", "🗣️ 語言質量"),
]

SCORE_GREEN_THRESHOLD = 7.0
SCORE_YELLOW_THRESHOLD = 4.0

HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_MAX_FILE_SIZE_MB = 25
HF_MAX_RETRIES = 5
HF_REQUEST_TIMEOUT = 180
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY = 2
ANALYSIS_CACHE_SIZE = 50

SAMPLE_TRANSCRIPT = """銷售員：你好，請問係唔係陳先生呀？
客戶：係呀，邊位呀？
銷售員：陳先生你好，我係ABC理財公司嘅李小明，打嚟係想同您介紹一個最近好多客戶都好感興趣嘅投資計劃。請問你而家方唔方便聽我講兩分鐘？
客戶：唔係幾有興趣，唔該哂。
銷售員：好，多謝你今日嘅時間，祝你事事順心！"""
