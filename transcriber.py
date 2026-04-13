# ============================================================
# 粵語 Cold Call 分析系統 - 語音轉文字模組
# ============================================================

import os
import requests
from pathlib import Path


class WhisperTranscriber:
    """使用 HuggingFace Inference API 或本地 Whisper 模型進行語音轉文字"""

    SUPPORTED_FORMATS = ["mp3", "mp4", "wav", "flac", "ogg", "m4a", "webm"]
    HF_API_URL = "https://api-inference.huggingface.co/models/{model_id}"

    def __init__(
        self,
        hf_token: str = "",
        model_id: str = "openai/whisper-large-v3",
        use_local: bool = False,
    ):
        self.hf_token = hf_token
        self.model_id = model_id
        self.use_local = use_local

    def update_token(self, hf_token: str):
        self.hf_token = hf_token

    def update_model(self, model_id: str):
        self.model_id = model_id

    @classmethod
    def get_supported_formats(cls) -> list:
        return cls.SUPPORTED_FORMATS

    def transcribe(self, file_path: str) -> dict:
        """
        轉錄音頻文件。

        Returns:
            dict with keys:
              success (bool), text (str), model (str), warning (str|None)
              — or —
              success (bool), error (str)
        """
        if self.use_local:
            return self._transcribe_local(file_path)
        return self._transcribe_hf_api(file_path)

    # ------------------------------------------------------------------
    # HuggingFace Inference API
    # ------------------------------------------------------------------

    def _transcribe_hf_api(self, file_path: str) -> dict:
        try:
            url = self.HF_API_URL.format(model_id=self.model_id)
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/octet-stream",
            }

            file_size = os.path.getsize(file_path)
            warning = None
            if file_size > 25 * 1024 * 1024:
                warning = "檔案超過 25 MB，HuggingFace API 可能拒絕請求，建議改用本地模式"

            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            response = requests.post(url, headers=headers, data=audio_bytes, timeout=180)

            if response.status_code == 200:
                data = response.json()
                text = data.get("text", "")
                return {
                    "success": True,
                    "text": text,
                    "model": self.model_id,
                    "warning": warning,
                }
            elif response.status_code == 503:
                return {
                    "success": False,
                    "error": "模型正在載入，請稍後重試（HF 冷啟動約需 20–60 秒）",
                }
            elif response.status_code == 401:
                return {"success": False, "error": "HuggingFace Token 無效或缺少存取權限"}
            elif response.status_code == 413:
                return {"success": False, "error": "檔案過大，請壓縮後再試或改用本地模式"}
            else:
                snippet = response.text[:300]
                return {
                    "success": False,
                    "error": f"API 返回錯誤 {response.status_code}：{snippet}",
                }

        except requests.exceptions.Timeout:
            return {"success": False, "error": "請求逾時，請稍後重試"}
        except Exception as exc:
            return {"success": False, "error": f"轉錄失敗：{exc}"}

    # ------------------------------------------------------------------
    # 本地 Whisper（需安裝 torch + transformers）
    # ------------------------------------------------------------------

    def _transcribe_local(self, file_path: str) -> dict:
        try:
            from transformers import pipeline
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=device,
            )
            result = pipe(file_path, return_timestamps=False)
            return {
                "success": True,
                "text": result["text"],
                "model": self.model_id,
                "warning": None,
            }
        except ImportError:
            return {
                "success": False,
                "error": "本地模式需要安裝 torch 及 transformers，請執行：pip install torch transformers",
            }
        except Exception as exc:
            return {"success": False, "error": f"本地轉錄失敗：{exc}"}
