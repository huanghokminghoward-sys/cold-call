import os
import time
import requests
from pathlib import Path


class WhisperTranscriber:
    SUPPORTED_FORMATS = ["mp3", "mp4", "wav", "flac", "ogg", "m4a", "webm"]
    HF_API_BASE = "https://api-inference.huggingface.co/models"

    def __init__(self, hf_token="", model_id="openai/whisper-large-v3", use_local=False):
        self.hf_token = hf_token
        self.model_id = model_id
        self.use_local = use_local

    def update_token(self, hf_token):
        self.hf_token = hf_token

    def update_model(self, model_id):
        self.model_id = model_id

    @classmethod
    def get_supported_formats(cls):
        return cls.SUPPORTED_FORMATS

    def transcribe(self, file_path):
        if self.use_local:
            return self._transcribe_local(file_path)
        return self._transcribe_hf_api(file_path)

    def _transcribe_hf_api(self, file_path):
        try:
            url = f"{self.HF_API_BASE}/{self.model_id}"
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/octet-stream",
            }
            file_size = os.path.getsize(file_path)
            warning = None
            if file_size > 25 * 1024 * 1024:
                warning = "檔案超過 25 MB，建議改用本地模式"
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            for attempt in range(5):
                response = requests.post(url, headers=headers, data=audio_bytes, timeout=180)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict):
                        text = data.get("text", "")
                        if not text and "chunks" in data:
                            text = " ".join(c.get("text", "") for c in data["chunks"])
                    elif isinstance(data, list):
                        text = " ".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in data)
                    else:
                        text = str(data)
                    return {"success": True, "text": text.strip(), "model": self.model_id, "warning": warning}
                if response.status_code == 503:
                    try:
                        wait = min(float(response.json().get("estimated_time", 20)), 40)
                    except Exception:
                        wait = 20
                    if attempt < 4:
                        time.sleep(wait)
                        continue
                    return {"success": False, "error": "模型載入中，請稍後重試（約1-2分鐘）"}
                if response.status_code == 429:
                    if attempt < 4:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    return {"success": False, "error": "請求過於頻繁，請稍後重試"}
                if response.status_code == 401:
                    return {"success": False, "error": "HuggingFace Token 無效"}
                return {"success": False, "error": f"API 錯誤 {response.status_code}：{response.text[:200]}"}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "請求逾時，請嘗試較短片段（建議5分鐘以內）"}
        except Exception as exc:
            return {"success": False, "error": f"轉錄失敗：{exc}"}

    def _transcribe_local(self, file_path):
        try:
            from transformers import pipeline
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipeline("automatic-speech-recognition", model=self.model_id, device=device)
            result = pipe(file_path, return_timestamps=False)
            return {"success": True, "text": result["text"], "model": self.model_id, "warning": None}
        except ImportError:
            return {"success": False, "error": "本地模式需安裝：pip install torch transformers"}
        except Exception as exc:
            return {"success": False, "error": f"本地轉錄失敗：{exc}"}
