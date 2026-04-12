# ============================================================
# 粵語 Cold Call 分析系統 - AI 分析模組
# ============================================================

import hashlib
import json
import re

import google.generativeai as genai


# ============================================================
# 分析 Prompt 模板
# ============================================================

_PROMPT_TEMPLATE = """你係一個專業嘅銷售培訓師，專門分析粵語 Cold Call 錄音。
請仔細分析以下對話，並以 JSON 格式返回詳細分析報告。

對話內容：
{transcript}

請返回以下 JSON 格式（必須係有效嘅 JSON，不要加任何其他文字、markdown 代碼塊或解釋）：
{{
  "overall_score": <0–10 浮點數>,
  "success_probability": <0–100 整數>,
  "call_stage": "<cold_call | follow_up | closing | other>",
  "opening": {{
    "score": <0–10 浮點數>,
    "analysis": "<開場白分析>",
    "suggestions": ["<建議1>", "<建議2>"]
  }},
  "needs_discovery": {{
    "score": <0–10 浮點數>,
    "analysis": "<需求發掘分析>",
    "suggestions": ["<建議>"]
  }},
  "product_presentation": {{
    "score": <0–10 浮點數>,
    "analysis": "<產品介紹分析>",
    "suggestions": ["<建議>"]
  }},
  "objection_handling": {{
    "score": <0–10 浮點數>,
    "analysis": "<異議處理分析>",
    "objections_raised": ["<客戶反對意見>"],
    "handling_quality": "<處理質量評語>",
    "suggestions": ["<建議>"]
  }},
  "closing": {{
    "score": <0–10 浮點數>,
    "analysis": "<成交技巧分析>",
    "next_steps": ["<後續步驟>"],
    "suggestions": ["<建議>"]
  }},
  "language_quality": {{
    "score": <0–10 浮點數>,
    "cantonese_naturalness": "<粵語自然度分析>",
    "tone_assessment": "<語調評估>",
    "suggestions": ["<建議>"]
  }},
  "customer_sentiment": {{
    "overall": "<positive | neutral | negative>",
    "trajectory": "<improving | stable | declining>",
    "key_signals": ["<情緒信號>"]
  }},
  "competitor_mentions": {{
    "detected": <true | false>,
    "competitors": ["<競爭對手名稱>"],
    "context": "<提及競爭對手嘅背景>"
  }},
  "top_strengths": ["<主要優點1>", "<主要優點2>", "<主要優點3>"],
  "top_improvements": ["<改善建議1>", "<改善建議2>", "<改善建議3>"],
  "recommended_followup": "<跟進建議>",
  "summary": "<整體總結>"
}}"""


# ============================================================
# ColdCallAnalyzer
# ============================================================


class ColdCallAnalyzer:
    """使用 Gemini 模型分析 Cold Call 對話文字"""

    def __init__(
        self,
        gemini_api_key: str = "",
        model_name: str = "gemini-1.5-flash",
    ):
        self.gemini_api_key = gemini_api_key
        self.model_name = model_name
        self._cache: dict = {}
        self._model = None

    # ------------------------------------------------------------------
    # 設定更新（Streamlit 側邊欄會動態呼叫）
    # ------------------------------------------------------------------

    def update_api_key(self, api_key: str):
        if api_key != self.gemini_api_key:
            self.gemini_api_key = api_key
            self._model = None  # 強制重新初始化

    def update_model(self, model_name: str):
        if model_name != self.model_name:
            self.model_name = model_name
            self._model = None

    def clear_cache(self):
        self._cache.clear()

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------

    def _get_model(self):
        if self._model is None:
            genai.configure(api_key=self.gemini_api_key)
            self._model = genai.GenerativeModel(self.model_name)
        return self._model

    @staticmethod
    def _cache_key(transcript: str) -> str:
        return hashlib.md5(transcript.encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_json(raw: str) -> str:
        """從 AI 回應中提取 JSON 物件字串"""
        # 去掉可能的 markdown 代碼塊
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            return match.group()
        raise ValueError("回應中找不到有效的 JSON 物件")

    # ------------------------------------------------------------------
    # 主要分析方法
    # ------------------------------------------------------------------

    def analyze(self, transcript: str) -> dict:
        """
        分析 Cold Call 對話。

        Returns:
            dict — 成功時包含分析欄位；失敗時包含 "error" 鍵。
            若結果來自 cache，額外附帶 "_from_cache": True。
        """
        key = self._cache_key(transcript)
        if key in self._cache:
            result = dict(self._cache[key])
            result["_from_cache"] = True
            return result

        raw = ""
        try:
            model = self._get_model()
            prompt = _PROMPT_TEMPLATE.format(transcript=transcript)
            response = model.generate_content(prompt)
            raw = response.text.strip()

            json_str = self._extract_json(raw)
            result = json.loads(json_str)
            self._cache[key] = result
            return result

        except json.JSONDecodeError as exc:
            return {
                "error": f"JSON 解析錯誤：{exc}",
                "raw_response": raw,
            }
        except ValueError as exc:
            return {
                "error": str(exc),
                "raw_response": raw,
            }
        except Exception as exc:
            return {"error": f"分析失敗：{exc}"}
