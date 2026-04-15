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

SAMPLE_TRANSCRIPTS = [
    """銷售員：你好，請問係唔係陳先生呀？
客戶：係呀，邊位呀？
銷售員：陳先生你好，我係ABC理財公司嘅李小明，打嚟係想同您介紹一個最近好多客戶都好感興趣嘅投資計劃。請問你而家方唔方便聽我講兩分鐘？
客戶：唔係幾有興趣，唔該哂。
銷售員：好，多謝你今日嘅時間，祝你事事順心！""",

    """銷售員：你好，請問係王小姐嗎？
客戶：係呀，咩事？
銷售員：王小姐你好！我係新世代保險嘅張志豪，我哋有款全新醫療保障計劃，保費好抵，我想約你出嚟了解一下，你今個星期幾方便？
客戶：我唔需要保險。
銷售員：明白！其實好多客戶最初都係噉諗，但了解完之後覺得好有用。你有冇時間今個星期四或者五？
客戶：唔方便，再見。
銷售員：噉你留個最方便嘅時間俾我？
客戶：唔用喇，拜拜。""",

    """銷售員：你好請問係李先生嗎？
客戶：係，邊位？
銷售員：李先生你好，我係科技通訊嘅陳美玲，我哋公司而家推緊一個企業寬頻升級計劃，比你而家用緊嘅快三倍，仲平20%。請問你公司而家用緊邊間？
客戶：用緊電訊盈科。
銷售員：明白！其實好多客戶由電訊盈科轉過嚟，話我哋服務更穩定。我可唔可以下個禮拜安排個15分鐘嘅示範俾你睇睇？
客戶：可以呀，你send個proposal俾我先。
銷售員：好嘅！請問你嘅電郵係幾多？
客戶：係li@company.com。
銷售員：多謝李先生！我今日內send俾你，然後再跟進。""",

    """銷售員：你好，麻煩請林老闆聽電話。
客戶：我就係，咩事？
銷售員：林老闆你好！我係數碼行銷嘅黃偉明，我哋專門幫中小企做網上推廣，我哋有客戶用咗我哋服務之後生意額增加咗三成，我想了解下你哋公司而家點做推廣？
客戶：我哋而家主要靠口碑。
銷售員：口碑係好重要㗎！但係如果配合網上推廣，效果可以更好。我可唔可以約你出嚟傾15分鐘？
客戶：你send資料俾我先睇睇。
銷售員：好嘅！你whatsapp幾多號？我send詳細資料同成功案例俾你。
客戶：98765432。
銷售員：多謝林老闆！我依家send俾你，有問題隨時搵我。"""
]

import random
def get_random_sample():
    return random.choice(SAMPLE_TRANSCRIPTS)

SAMPLE_TRANSCRIPT = SAMPLE_TRANSCRIPTS[0]
