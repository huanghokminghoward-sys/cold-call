# ============================================================
# 粵語 Cold Call 分析系統 - 設定常數
# ============================================================

WHISPER_MODELS = {
    "openai/whisper-large-v3": "Whisper Large v3（最準確）",
    "openai/whisper-medium": "Whisper Medium（平衡）",
    "openai/whisper-small": "Whisper Small（較快）",
}

GEMINI_MODELS = {
    "gemini-1.5-flash": "Gemini 1.5 Flash（快速）",
    "gemini-1.5-pro": "Gemini 1.5 Pro（最強）",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
}

ANALYSIS_SECTIONS = [
    ("opening", "🎯 開場白"),
    ("needs_discovery", "🔍 需求發掘"),
    ("product_presentation", "📢 產品介紹"),
    ("objection_handling", "🛡️ 異議處理"),
    ("closing", "🤝 成交技巧"),
    ("language_quality", "🗣️ 語言質量"),
]

SCORE_GREEN_THRESHOLD = 7.0
SCORE_YELLOW_THRESHOLD = 4.0

SAMPLE_TRANSCRIPT = """銷售員：你好，請問係唔係陳先生呀？
客戶：係呀，邊位呀？
銷售員：陳先生你好，我係ABC理財公司嘅李小明，打嚟係想同您介紹一個最近好多客戶都好感興趣嘅投資計劃。請問你而家方唔方便聽我講兩分鐘？
客戶：唔係咁啱呀，我而家好忙。
銷售員：明白明白，陳先生，咁我唔會耽誤你太多時間，只係想了解一下你有冇做緊任何投資規劃呀？
客戶：有做緊少少啫，不過唔係太多。
銷售員：係咁樣，其實而家市場波動好大，好多人都諗住點樣令自己嘅資產更穩健。我地有個計劃係專門為忙碌嘅專業人士設計，每月只需要幾千蚊就可以開始。你有冇興趣了解多少少？
客戶：你地係咩公司黎㗎？我之前買過其他公司嘅，結果輸咗唔少。
銷售員：陳先生，你嘅顧慮係好合理嘅。我地ABC理財係有20年歷史嘅持牌公司，受證監會監管。同埋我地有提供保本計劃，就算市場下跌都唔會蝕本。
客戶：保本？真係假㗎？現在哪有保本嘅投資？
銷售員：係，我明白你有疑慮。其實係有一定條件嘅，唔係完全保本，而係有下限保障。你方唔方便我下個禮拜約個時間詳細解釋畀你聽？
客戶：我先考慮下啦。
銷售員：好，陳先生，咁我留低我嘅聯絡方式畀你，你有任何問題隨時搵我。多謝你今日嘅時間，祝你事事順心！"""
CANTONESE_SPECIFIC_MODELS = set()
HF_API_BASE = "https://api-inference.huggingface.co/models"
HF_MAX_FILE_SIZE_MB = 25
HF_MAX_RETRIES = 5
HF_REQUEST_TIMEOUT = 180
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY = 2
ANALYSIS_CACHE_SIZE = 50
