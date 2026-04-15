import os
import json
import re
import time
import hashlib
import logging

import google.generativeai as genai

from config import (
    GEMINI_MAX_RETRIES,
    GEMINI_RETRY_DELAY,
    ANALYSIS_CACHE_SIZE,
)

logger = logging.getLogger(__name__)

_SCHEMA = """{
  "overall_score": <1-10整數>,
  "success_probability": <0-100整數>,
  "opening": {"score": <1-10>, "analysis": "<廣東話>", "suggestions": ["<建議>"]},
  "needs_discovery": {"score": <1-10>, "analysis": "<廣東話>", "suggestions": ["<建議>"]},
  "product_presentation": {"score": <1-10>, "analysis": "<廣東話>", "suggestions": ["<建議>"]},
  "objection_handling": {"score": <1-10>, "objections_raised": ["<反對意見>"], "handling_quality": "<廣東話>", "better_responses": ["<改善回應>"]},
  "closing": {"score": <1-10>, "analysis": "<廣東話>", "next_steps": ["<下一步>"]},
  "language_quality": {"score": <1-10>, "cantonese_naturalness": "<廣東話>", "tone_assessment": "<廣東話>"},
  "customer_sentiment": {"overall": "<positive/neutral/negative>", "trajectory": "<走向>", "key_signals": ["<信號>"]},
  "competitor_mentions": {"detected": <true/false>, "competitors": ["<競爭對手>"], "context": "<廣東話>"},
  "call_stage": "<完整通話/片段-開場/片段-中段/片段-收尾>",
  "top_strengths": ["<優點1>", "<優點2>", "<優點3>"],
  "top_improvements": ["<改善1>", "<改善2>", "<改善3>"],
  "recommended_followup": "<廣東話>",
  "summary": "<100字以內，廣東話>"
}"""


class ColdCallAnalyzer:
    def __init__(self, gemini_api_key="", model_name="gemini-1.5-flash"):
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self._model = None
        self._init_error = None
        self._cache = {}
        if self.api_key:
            self._init_model()

    def analyze(self, transcript):
        if not self.api_key:
            return {"error": "請在側邊欄輸入 Gemini API Key"}
        if not transcript.strip():
            return {"error": "對話文字不能為空"}
        if self._init_error:
            return {"error": f"模型初始化失敗：{self._init_error}"}
        if self._model is None:
            self._init_model()
            if self._model is None:
                return {"error": self._init_error or "模型初始化失敗"}
        cache_key = hashlib.md5(f"{self.model_name}:{transcript}".encode()).hexdigest()
        if cache_key in self._cache:
            cached = self._cache[cache_key].copy()
            cached["_from_cache"] = True
            return cached
        result = self._analyze_with_retry(transcript)
        if "error" not in result:
            if len(self._cache) >= ANALYSIS_CACHE_SIZE:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = result
        return result

    def analyze_batch(self, transcripts):
        return [self.analyze(t) for t in transcripts]

    def clear_cache(self):
        self._cache.clear()

    def update_api_key(self, new_key):
        self.api_key = new_key
        self._init_error = None
        self._model = None
        if new_key:
            self._init_model()

    def update_model(self, model_name):
        self.model_name = model_name
        self._model = None
        self._init_error = None
        if self.api_key:
            self._init_model()

    def _init_model(self):
        try:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
            self._init_error = None
        except Exception as e:
            self._model = None
            self._init_error = str(e)
            logger.error(f"Gemini 初始化失敗：{e}")

    def _analyze_with_retry(self, transcript):
        prompt = self._build_prompt(transcript)
        for attempt in range(GEMINI_MAX_RETRIES):
            result = self._generate_with_json_mode(prompt)
            if result is not None:
                return result
            result = self._generate_plain(prompt)
            if "error" in result:
                if any(kw in result["error"] for kw in ["503", "500", "超時", "連接"]) and attempt < GEMINI_MAX_RETRIES - 1:
                    time.sleep(GEMINI_RETRY_DELAY * (2 ** attempt))
                    continue
            return result
        return {"error": "多次重試後仍然失敗"}

    def _generate_with_json_mode(self, prompt):
        try:
            config = genai.GenerationConfig(temperature=0.2, response_mime_type="application/json")
            response = self._model.generate_content(prompt, generation_config=config)
            return self._parse_response(response.text)
        except Exception:
            return None

    def _generate_plain(self, prompt):
        try:
            config = genai.GenerationConfig(temperature=0.2)
            response = self._model.generate_content(prompt, generation_config=config)
            return self._parse_response(response.text)
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "400" in error_msg:
                return {"error": "Gemini API Key 無效"}
            if "QUOTA_EXCEEDED" in error_msg or "429" in error_msg:
                return {"error": "已超出 Gemini 免費配額，請稍後再試"}
            return {"error": f"分析失敗：{error_msg[:200]}"}

    def _build_prompt(self, transcript):
        return f"""你係一個擁有10年經驗嘅專業銷售培訓師，專門分析粵語電話銷售（Cold Call）。
請仔細分析以下對話，用地道廣東話撰寫深入嘅評估報告。
如果對話係片段而非完整通話，請在 call_stage 欄位標明。

=== Cold Call 對話文字 ===
{transcript}
===========================

請嚴格按照以下 JSON 格式輸出（唔好加任何額外文字）：
{_SCHEMA}

重要要求：
1. 所有分析文字必須用地道廣東話
2. 每條建議必須具體、可立即執行
3. 評分要客觀
4. better_responses 必須提供可直接使用嘅粵語對白範例
5. 輸出必須係有效 JSON"""

    def _parse_response(self, text):
        if not text:
            return {"error": "Gemini 返回空回應"}
        try:
            return json.loads(text.strip())
        except Exception:
            pass
        md = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
        if md:
            try:
                return json.loads(md.group(1).strip())
            except Exception:
                pass
        s, e = text.find("{"), text.rfind("}") + 1
        if s >= 0 and e > s:
            try:
                return json.loads(text[s:e])
            except Exception:
                pass
        return {"error": "無法解析結果，請重試", "overall_score": 0, "success_probability": 0, "summary": "格式錯誤", "top_strengths": [], "top_improvements": []}
