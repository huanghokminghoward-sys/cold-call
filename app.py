# ============================================================
# 粵語 Cold Call 分析系統 - Streamlit 主介面
# ============================================================

import os
import json
import tempfile
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from config import (
    WHISPER_MODELS,
    GEMINI_MODELS,
    ANALYSIS_SECTIONS,
    SCORE_GREEN_THRESHOLD,
    SCORE_YELLOW_THRESHOLD,
    SAMPLE_TRANSCRIPT,
)
from transcriber import WhisperTranscriber
from analyzer import ColdCallAnalyzer

# ============================================================
# 頁面設定
# ============================================================

st.set_page_config(
    page_title="粵語 Cold Call 分析系統",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Session State 初始化
# ============================================================


def _init_state():
    defaults = {
        "hf_token": os.getenv("HF_TOKEN", ""),
        "gemini_key": os.getenv("GEMINI_API_KEY", ""),
        "whisper_model": list(WHISPER_MODELS.keys())[0],
        "gemini_model": list(GEMINI_MODELS.keys())[0],
        "use_local": False,
        "transcript": "",
        "analysis": None,
        "transcriber": None,
        "analyzer": None,
        "history": [],  # [{name, transcript, result}]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ============================================================
# 取得或建立 Transcriber / Analyzer（避免重複初始化）
# ============================================================


def get_transcriber() -> WhisperTranscriber:
    t = st.session_state.transcriber
    if t is None:
        t = WhisperTranscriber(
            hf_token=st.session_state.hf_token,
            model_id=st.session_state.whisper_model,
            use_local=st.session_state.use_local,
        )
        st.session_state.transcriber = t
    return t


def get_analyzer() -> ColdCallAnalyzer:
    a = st.session_state.analyzer
    if a is None:
        a = ColdCallAnalyzer(
            gemini_api_key=st.session_state.gemini_key,
            model_name=st.session_state.gemini_model,
        )
        st.session_state.analyzer = a
    return a


# ============================================================
# 輔助函數（必須在 tab 代碼之前定義）
# ============================================================


def _score_color(score: float) -> str:
    if score >= SCORE_GREEN_THRESHOLD:
        return "🟢"
    elif score >= SCORE_YELLOW_THRESHOLD:
        return "🟡"
    return "🔴"


def _render_analysis(r: dict):
    """渲染完整分析報告"""

    # 頂部指標
    overall = r.get("overall_score", 0)
    prob = r.get("success_probability", 0)
    call_stage = r.get("call_stage", "")

    m1, m2, m3 = st.columns(3)
    m1.metric("整體評分", f"{overall} / 10")
    m2.metric("成功率預測", f"{prob}%")
    if call_stage:
        m3.metric("通話類型", call_stage)

    # 雷達圖
    section_scores = []
    section_labels = []
    for key, label in ANALYSIS_SECTIONS:
        section = r.get(key, {})
        if isinstance(section, dict) and "score" in section:
            section_scores.append(section["score"])
            section_labels.append(label.split(" ", 1)[-1])  # 去掉 emoji

    if section_scores:
        fig = go.Figure(
            data=go.Scatterpolar(
                r=section_scores + [section_scores[0]],
                theta=section_labels + [section_labels[0]],
                fill="toself",
                fillcolor="rgba(99, 110, 250, 0.2)",
                line=dict(color="rgb(99, 110, 250)"),
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    # 各維度評分
    st.subheader("📋 各維度分析")
    for key, label in ANALYSIS_SECTIONS:
        section = r.get(key, {})
        if not isinstance(section, dict):
            continue
        score = section.get("score", 0)
        analysis_text = section.get("analysis", "") or section.get(
            "cantonese_naturalness", ""
        )
        suggestions = (
            section.get("suggestions", [])
            or section.get("key_moments", [])
            or section.get("missing_points", [])
            or section.get("better_responses", [])
            or section.get("next_steps", [])
        )
        with st.expander(f"{_score_color(score)} {label}  ·  {score}/10"):
            if analysis_text:
                st.write(analysis_text)
            if section.get("tone_assessment"):
                st.write(section["tone_assessment"])
            if section.get("handling_quality"):
                st.write(f"**處理質量：** {section['handling_quality']}")
            if section.get("objections_raised"):
                st.write("**客戶反對意見：**")
                for obj in section["objections_raised"]:
                    st.write(f"  • {obj}")
            if suggestions:
                st.write("**建議：**")
                for s in suggestions:
                    st.write(f"  • {s}")

    # 客戶情緒
    sentiment = r.get("customer_sentiment", {})
    if sentiment:
        st.subheader("😐 客戶情緒")
        overall_sent = sentiment.get("overall", "")
        emoji = {"positive": "😊", "neutral": "😐", "negative": "😟"}.get(
            overall_sent, ""
        )
        st.write(f"{emoji} **整體：** {overall_sent}")
        st.write(f"**走向：** {sentiment.get('trajectory', '')}")
        if sentiment.get("key_signals"):
            for sig in sentiment["key_signals"]:
                st.write(f"  • {sig}")

    # 競爭對手情報
    competitor = r.get("competitor_mentions", {})
    if competitor and competitor.get("detected"):
        st.subheader("🕵️ 競爭對手情報")
        for c in competitor.get("competitors", []):
            st.write(f"  • {c}")
        if competitor.get("context"):
            st.caption(competitor["context"])

    # 優點與改善
    col_str, col_imp = st.columns(2)
    with col_str:
        st.subheader("💪 主要優點")
        for s in r.get("top_strengths", []):
            st.success(s)
    with col_imp:
        st.subheader("🎯 需要改善")
        for s in r.get("top_improvements", []):
            st.warning(s)

    # 跟進建議 + 總結
    if r.get("recommended_followup"):
        st.subheader("📌 跟進建議")
        st.info(r["recommended_followup"])

    if r.get("summary"):
        st.subheader("📝 總結")
        st.write(r["summary"])

    # 匯出 JSON
    with st.expander("🔧 匯出原始數據"):
        st.json(r)
        st.download_button(
            "下載 JSON",
            data=json.dumps(r, ensure_ascii=False, indent=2),
            file_name="cold_call_analysis.json",
            mime="application/json",
        )


# ============================================================
# 側邊欄
# ============================================================

with st.sidebar:
    st.title("⚙️ 設定")

    st.subheader("🔑 API 金鑰")

    hf_token = st.text_input(
        "HuggingFace Token",
        value=st.session_state.hf_token,
        type="password",
        help="免費申請：https://huggingface.co/settings/tokens",
    )
    if hf_token != st.session_state.hf_token:
        st.session_state.hf_token = hf_token
        st.session_state.transcriber = None  # 重置 transcriber

    gemini_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.gemini_key,
        type="password",
        help="免費申請：https://aistudio.google.com/apikey",
    )
    if gemini_key != st.session_state.gemini_key:
        st.session_state.gemini_key = gemini_key
        if st.session_state.analyzer:
            st.session_state.analyzer.update_api_key(gemini_key)
        else:
            st.session_state.analyzer = None

    st.divider()

    st.subheader("🤖 模型選擇")

    whisper_choice = st.selectbox(
        "語音識別模型",
        options=list(WHISPER_MODELS.keys()),
        format_func=lambda k: WHISPER_MODELS[k],
        index=list(WHISPER_MODELS.keys()).index(st.session_state.whisper_model),
    )
    if whisper_choice != st.session_state.whisper_model:
        st.session_state.whisper_model = whisper_choice
        st.session_state.transcriber = None

    gemini_choice = st.selectbox(
        "分析模型",
        options=list(GEMINI_MODELS.keys()),
        format_func=lambda k: GEMINI_MODELS[k],
        index=list(GEMINI_MODELS.keys()).index(st.session_state.gemini_model),
    )
    if gemini_choice != st.session_state.gemini_model:
        st.session_state.gemini_model = gemini_choice
        if st.session_state.analyzer:
            st.session_state.analyzer.update_model(gemini_choice)

    use_local = st.toggle(
        "使用本地 Whisper 模式",
        value=st.session_state.use_local,
        help="需要額外安裝 torch、transformers。速度較慢但私隱度更高。",
    )
    if use_local != st.session_state.use_local:
        st.session_state.use_local = use_local
        st.session_state.transcriber = None

    st.divider()

    # Cache 狀態
    if st.session_state.analyzer:
        cache_count = len(st.session_state.analyzer._cache)
        st.caption(f"📦 已緩存分析結果：{cache_count} 筆")
        if cache_count > 0 and st.button("清除 Cache"):
            st.session_state.analyzer.clear_cache()
            st.success("Cache 已清除")

    # 歷史記錄數量
    if st.session_state.history:
        st.caption(f"📋 歷史記錄：{len(st.session_state.history)} 筆")


# ============================================================
# 主介面
# ============================================================

st.title("📞 粵語 Cold Call 分析系統")
st.caption("上傳錄音或輸入對話文字，用 AI 深度分析銷售技巧")

tab_analyze, tab_history, tab_batch = st.tabs(["🔍 單次分析", "📋 歷史記錄", "📦 批量分析"])

# ============================================================
# Tab 1：單次分析
# ============================================================

with tab_analyze:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📥 輸入來源")

        input_mode = st.radio(
            "選擇輸入方式",
           ["🎙️ 上傳錄音", "🎤 當場錄音", "📝 直接輸入文字"],
            horizontal=True,
        )

        transcript_text = ""

        if input_mode == "🎙️ 上傳錄音":
            supported = WhisperTranscriber.get_supported_formats()
            uploaded = st.file_uploader(
                "上傳錄音文件",
                type=supported,
                help=f"支援格式：{', '.join(f.upper() for f in supported)}，HF API 上限 25MB",
            )

            if uploaded:
                st.audio(uploaded)
                file_size_mb = uploaded.size / (1024 * 1024)
                st.caption(f"檔案大小：{file_size_mb:.1f} MB")

                if st.button("🎙️ 開始轉錄", type="primary", use_container_width=True):
                    if not st.session_state.hf_token:
                        st.error("請先在側邊欄輸入 HuggingFace Token")
                    else:
                        with st.spinner("轉錄中，請稍候..."):
                            with tempfile.NamedTemporaryFile(
                                suffix=Path(uploaded.name).suffix, delete=False
                            ) as tmp:
                                tmp.write(uploaded.read())
                                tmp_path = tmp.name

                            t = get_transcriber()
                            t.update_token(st.session_state.hf_token)
                            t.update_model(st.session_state.whisper_model)
                            result = t.transcribe(tmp_path)

                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass

                        if result["success"]:
                            st.session_state.transcript = result["text"]
                            st.success(f"轉錄完成（模型：{result['model']}）")
                            if result.get("warning"):
                                st.warning(result["warning"])
                        else:
                            st.error(f"轉錄失敗：{result['error']}")

            if st.session_state.transcript:
                transcript_text = st.text_area(
                    "轉錄結果（可手動修改）",
                    value=st.session_state.transcript,
                    height=250,
                )
elif input_mode == "🎤 當場錄音":
            st.caption("點擊麥克風開始錄音，錄完再點一次停止")
            audio_data = st.audio_input("錄製通話")
            if audio_data:
                st.audio(audio_data)
                if st.button("🎤 開始轉錄錄音", type="primary", use_container_width=True):
                    if not st.session_state.hf_token:
                        st.error("請先在側邊欄輸入 HuggingFace Token")
                    else:
                        with st.spinner("轉錄中，請稍候..."):
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                tmp.write(audio_data.getvalue())
                                tmp_path = tmp.name
                            t = get_transcriber()
                            t.update_token(st.session_state.hf_token)
                            t.update_model(st.session_state.whisper_model)
                            result = t.transcribe(tmp_path)
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass
                        if result["success"]:
                            st.session_state.transcript = result["text"]
                            st.success("轉錄完成")
                        else:
                            st.error(f"轉錄失敗：{result['error']}")
            if st.session_state.transcript:
                transcript_text = st.text_area(
                    "轉錄結果（可手動修改）",
                    value=st.session_state.transcript,
                    height=250,
                )
        else:
            transcript_text = st.text_area(
                "輸入對話文字",
                value=st.session_state.transcript or "",
                height=300,
                placeholder="銷售員：你好，請問係唔係...\n客戶：係呀...",
            )
            if st.button("載入示例對話", use_container_width=True):
                st.session_state.transcript = SAMPLE_TRANSCRIPT
                st.rerun()

        call_name = st.text_input(
            "通話備註（選填）",
            placeholder="例如：2024-01-15 陳先生，方便之後查找記錄",
        )

        analyze_btn = st.button(
            "🔍 開始 AI 分析",
            type="primary",
            use_container_width=True,
            disabled=not transcript_text.strip(),
        )

        if analyze_btn:
            if not st.session_state.gemini_key:
                st.error("請先在側邊欄輸入 Gemini API Key")
            else:
                st.session_state.transcript = transcript_text
                with st.spinner("AI 分析中，請稍候..."):
                    a = get_analyzer()
                    a.update_api_key(st.session_state.gemini_key)
                    a.update_model(st.session_state.gemini_model)
                    result = a.analyze(transcript_text)

                st.session_state.analysis = result

                # 儲存歷史
                if "error" not in result:
                    st.session_state.history.append(
                        {
                            "name": call_name
                            or f"通話 #{len(st.session_state.history) + 1}",
                            "transcript": transcript_text,
                            "result": result,
                        }
                    )

                if result.get("_from_cache"):
                    st.info("⚡ 從 Cache 讀取（相同內容已分析過）")

    # ---- 結果顯示 ----
    with col_result:
        st.subheader("📊 分析結果")

        analysis = st.session_state.analysis
        if analysis is None:
            st.info("分析結果會顯示在這裡")
        elif "error" in analysis:
            st.error(f"❌ {analysis['error']}")
            if analysis.get("raw_response"):
                with st.expander("原始回應（debug 用）"):
                    st.text(analysis["raw_response"])
        else:
            _render_analysis(analysis)


# ============================================================
# Tab 2：歷史記錄
# ============================================================

with tab_history:
    st.subheader("📋 歷史分析記錄")

    if not st.session_state.history:
        st.info("暫無記錄，完成分析後會自動儲存")
    else:
        history = st.session_state.history

        # 進度趨勢圖
        if len(history) >= 2:
            names = [h["name"] for h in history]
            scores = [h["result"].get("overall_score", 0) for h in history]
            probs = [h["result"].get("success_probability", 0) for h in history]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=names, y=scores, name="整體評分", mode="lines+markers")
            )
            fig.add_trace(
                go.Scatter(
                    x=names,
                    y=[p / 10 for p in probs],
                    name="成功率 (/10)",
                    mode="lines+markers",
                )
            )
            fig.update_layout(
                title="銷售員進度趨勢",
                yaxis=dict(range=[0, 10]),
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

        # 逐筆顯示
        for i, record in enumerate(reversed(history)):
            result = record["result"]
            score = result.get("overall_score", 0)
            prob = result.get("success_probability", 0)
            with st.expander(
                f"{_score_color(score)} {record['name']}  ·  評分 {score}/10  ·  成功率 {prob}%"
            ):
                _render_analysis(result)

        if st.button("清除所有歷史記錄", type="secondary"):
            st.session_state.history = []
            st.rerun()


# ============================================================
# Tab 3：批量分析
# ============================================================

with tab_batch:
    st.subheader("📦 批量分析")
    st.caption("同時分析多段對話，適合主管批閱整個團隊的錄音")

    batch_input = st.text_area(
        "輸入多段對話（用 === 分隔）",
        height=300,
        placeholder="銷售員：你好...\n客戶：係呀...\n===\n銷售員：你好...\n客戶：邊位...",
    )

    if st.button("🚀 開始批量分析", type="primary", disabled=not batch_input.strip()):
        if not st.session_state.gemini_key:
            st.error("請先在側邊欄輸入 Gemini API Key")
        else:
            segments = [s.strip() for s in batch_input.split("===") if s.strip()]
            st.info(f"共偵測到 {len(segments)} 段對話")

            a = get_analyzer()
            a.update_api_key(st.session_state.gemini_key)
            a.update_model(st.session_state.gemini_model)

            results = []
            progress = st.progress(0)
            for i, seg in enumerate(segments):
                with st.spinner(f"分析第 {i + 1}/{len(segments)} 段..."):
                    r = a.analyze(seg)
                    results.append(r)
                    progress.progress((i + 1) / len(segments))

            st.success(f"批量分析完成！共 {len(results)} 段")

            # 批量結果摘要表
            rows = []
            for i, (seg, r) in enumerate(zip(segments, results)):
                if "error" not in r:
                    rows.append(
                        {
                            "#": i + 1,
                            "整體評分": r.get("overall_score", 0),
                            "成功率": f"{r.get('success_probability', 0)}%",
                            "客戶情緒": r.get("customer_sentiment", {}).get(
                                "overall", "-"
                            ),
                            "有競爭對手": (
                                "✅"
                                if r.get("competitor_mentions", {}).get("detected")
                                else "-"
                            ),
                        }
                    )
                else:
                    rows.append(
                        {
                            "#": i + 1,
                            "整體評分": "錯誤",
                            "成功率": "-",
                            "客戶情緒": "-",
                            "有競爭對手": "-",
                        }
                    )

            if rows:
                st.dataframe(rows, use_container_width=True)

            # 儲存至歷史
            for i, (seg, r) in enumerate(zip(segments, results)):
                if "error" not in r:
                    st.session_state.history.append(
                        {
                            "name": f"批量 #{len(st.session_state.history) + 1}",
                            "transcript": seg,
                            "result": r,
                        }
                    )

            # 展開逐條結果
            for i, (seg, r) in enumerate(zip(segments, results)):
                with st.expander(f"第 {i + 1} 段詳細結果"):
                    if "error" in r:
                        st.error(r["error"])
                    else:
                        _render_analysis(r)
