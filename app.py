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

st.set_page_config(
    page_title="粵語 Cold Call 分析系統",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        "history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def get_transcriber():
    if st.session_state.transcriber is None:
        st.session_state.transcriber = WhisperTranscriber(
            hf_token=st.session_state.hf_token,
            model_id=st.session_state.whisper_model,
            use_local=st.session_state.use_local,
        )
    return st.session_state.transcriber

def get_analyzer():
    if st.session_state.analyzer is None:
        st.session_state.analyzer = ColdCallAnalyzer(
            gemini_api_key=st.session_state.gemini_key,
            model_name=st.session_state.gemini_model,
        )
    return st.session_state.analyzer

def _score_color(score):
    if score >= SCORE_GREEN_THRESHOLD:
        return "🟢"
    elif score >= SCORE_YELLOW_THRESHOLD:
        return "🟡"
    return "🔴"

def _render_analysis(r):
    overall = r.get("overall_score", 0)
    prob = r.get("success_probability", 0)
    call_stage = r.get("call_stage", "")
    m1, m2, m3 = st.columns(3)
    m1.metric("整體評分", f"{overall} / 10")
    m2.metric("成功率預測", f"{prob}%")
    if call_stage:
        m3.metric("通話類型", call_stage)
    scores, labels = [], []
    for key, label in ANALYSIS_SECTIONS:
        s = r.get(key, {})
        if isinstance(s, dict) and "score" in s:
            scores.append(s["score"])
            labels.append(label.split(" ", 1)[-1])
    if scores:
        fig = go.Figure(data=go.Scatterpolar(
            r=scores + [scores[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(99,110,250,0.2)",
            line=dict(color="rgb(99,110,250)"),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("📋 各維度分析")
    for key, label in ANALYSIS_SECTIONS:
        s = r.get(key, {})
        if not isinstance(s, dict):
            continue
        score = s.get("score", 0)
        text = s.get("analysis", "") or s.get("cantonese_naturalness", "")
        tips = s.get("suggestions", []) or s.get("better_responses", []) or s.get("next_steps", [])
        with st.expander(f"{_score_color(score)} {label} · {score}/10"):
            if text:
                st.write(text)
            if s.get("tone_assessment"):
                st.write(s["tone_assessment"])
            if s.get("handling_quality"):
                st.write(f"**處理質量：** {s['handling_quality']}")
            if s.get("objections_raised"):
                st.write("**客戶反對：**")
                for o in s["objections_raised"]:
                    st.write(f"• {o}")
            if tips:
                st.write("**建議：**")
                for t in tips:
                    st.write(f"• {t}")
    sent = r.get("customer_sentiment", {})
    if sent:
        st.subheader("😐 客戶情緒")
        e = {"positive": "😊", "neutral": "😐", "negative": "😟"}.get(sent.get("overall", ""), "")
        st.write(f"{e} **整體：** {sent.get('overall', '')}")
        st.write(f"**走向：** {sent.get('trajectory', '')}")
        for sig in sent.get("key_signals", []):
            st.write(f"• {sig}")
    comp = r.get("competitor_mentions", {})
    if comp and comp.get("detected"):
        st.subheader("🕵️ 競爭對手情報")
        for c in comp.get("competitors", []):
            st.write(f"• {c}")
        if comp.get("context"):
            st.caption(comp["context"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💪 主要優點")
        for s in r.get("top_strengths", []):
            st.success(s)
    with c2:
        st.subheader("🎯 需要改善")
        for s in r.get("top_improvements", []):
            st.warning(s)
    if r.get("recommended_followup"):
        st.subheader("📌 跟進建議")
        st.info(r["recommended_followup"])
    if r.get("summary"):
        st.subheader("📝 總結")
        st.write(r["summary"])
    with st.expander("🔧 匯出原始數據"):
        st.json(r)
        st.download_button(
            "下載 JSON",
            data=json.dumps(r, ensure_ascii=False, indent=2),
            file_name="cold_call_analysis.json",
            mime="application/json",
        )

with st.sidebar:
    st.title("⚙️ 設定")
    st.subheader("🔑 API 金鑰")
    hf_token = st.text_input("HuggingFace Token", value=st.session_state.hf_token, type="password")
    if hf_token != st.session_state.hf_token:
        st.session_state.hf_token = hf_token
        st.session_state.transcriber = None
    gemini_key = st.text_input("Gemini API Key", value=st.session_state.gemini_key, type="password")
    if gemini_key != st.session_state.gemini_key:
        st.session_state.gemini_key = gemini_key
        if st.session_state.analyzer:
            st.session_state.analyzer.update_api_key(gemini_key)
        else:
            st.session_state.analyzer = None
    st.divider()
    st.subheader("🤖 模型選擇")
    wc = st.selectbox("語音識別模型", options=list(WHISPER_MODELS.keys()), format_func=lambda k: WHISPER_MODELS[k])
    if wc != st.session_state.whisper_model:
        st.session_state.whisper_model = wc
        st.session_state.transcriber = None
    gc = st.selectbox("分析模型", options=list(GEMINI_MODELS.keys()), format_func=lambda k: GEMINI_MODELS[k])
    if gc != st.session_state.gemini_model:
        st.session_state.gemini_model = gc
        if st.session_state.analyzer:
            st.session_state.analyzer.update_model(gc)
    st.divider()
    if st.session_state.analyzer:
        cache_count = len(st.session_state.analyzer._cache)
        st.caption(f"📦 已緩存：{cache_count} 筆")
        if cache_count > 0 and st.button("清除 Cache"):
            st.session_state.analyzer.clear_cache()
            st.success("已清除")
    if st.session_state.history:
        st.caption(f"📋 歷史記錄：{len(st.session_state.history)} 筆")

st.title("📞 粵語 Cold Call 分析系統")
st.caption("上傳錄音、當場錄音或輸入對話文字，用 AI 深度分析銷售技巧")

tab1, tab2, tab3 = st.tabs(["🔍 單次分析", "📋 歷史記錄", "📦 批量分析"])

with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")
    with col_input:
        st.subheader("📥 輸入來源")
        mode = st.radio(
            "選擇輸入方式",
            ["🎙️ 上傳錄音", "🎤 當場錄音", "📝 直接輸入文字"],
            horizontal=True,
        )
        if mode == "🎙️ 上傳錄音":
            uploaded = st.file_uploader("上傳錄音文件", type=WhisperTranscriber.get_supported_formats())
            if uploaded:
                st.audio(uploaded)
                st.caption(f"檔案大小：{uploaded.size / 1024 / 1024:.1f} MB")
                if st.button("🎙️ 開始轉錄", type="primary", use_container_width=True):
                    if not st.session_state.hf_token:
                        st.error("請先在側邊欄輸入 HuggingFace Token")
                    else:
                        with st.spinner("轉錄中，請稍候..."):
                            with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
                                tmp.write(uploaded.read())
                                tmp_path = tmp.name
                            t = get_transcriber()
                            t.update_token(st.session_state.hf_token)
                            t.update_model(st.session_state.whisper_model)
                            res = t.transcribe(tmp_path)
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass
                        if res["success"]:
                            st.session_state.transcript = res["text"]
                            st.success("轉錄完成")
                            if res.get("warning"):
                                st.warning(res["warning"])
                        else:
                            st.error(f"轉錄失敗：{res['error']}")
            if st.session_state.transcript:
                edited = st.text_area("轉錄結果（可修改）", value=st.session_state.transcript, height=200, key="up_txt")
                st.session_state.transcript = edited

        elif mode == "🎤 當場錄音":
            st.caption("點擊麥克風開始錄音，錄完再點一次停止")
            audio_data = st.audio_input("錄製通話")
            if audio_data is not None:
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
                            res = t.transcribe(tmp_path)
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass
                        if res["success"]:
                            st.session_state.transcript = res["text"]
                            st.success("轉錄完成")
                        else:
                            st.error(f"轉錄失敗：{res['error']}")
            if st.session_state.transcript:
                edited = st.text_area("轉錄結果（可修改）", value=st.session_state.transcript, height=200, key="rec_txt")
                st.session_state.transcript = edited

        else:
            edited = st.text_area(
                "輸入對話文字",
                value=st.session_state.transcript,
                height=300,
                placeholder="銷售員：你好，請問係唔係...\n客戶：係呀...",
                key="man_txt",
            )
            st.session_state.transcript = edited
            if st.button("載入示例對話", use_container_width=True):
                st.session_state.transcript = SAMPLE_TRANSCRIPT
                st.rerun()

        call_name = st.text_input("通話備註（選填）", placeholder="例如：陳先生 2024-01-15")

        if st.button(
            "🔍 開始 AI 分析",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.transcript.strip(),
        ):
            if not st.session_state.gemini_key:
                st.error("請先在側邊欄輸入 Gemini API Key")
            else:
                with st.spinner("AI 分析中，請稍候..."):
                    a = get_analyzer()
                    a.update_api_key(st.session_state.gemini_key)
                    a.update_model(st.session_state.gemini_model)
                    result = a.analyze(st.session_state.transcript)
                st.session_state.analysis = result
                if "error" not in result:
                    st.session_state.history.append({
                        "name": call_name or f"通話 #{len(st.session_state.history)+1}",
                        "transcript": st.session_state.transcript,
                        "result": result,
                    })
                if result.get("_from_cache"):
                    st.info("⚡ 從 Cache 讀取")

    with col_result:
        st.subheader("📊 分析結果")
        analysis = st.session_state.analysis
        if analysis is None:
            st.info("分析結果會顯示在這裡")
        elif "error" in analysis:
            st.error(f"❌ {analysis['error']}")
        else:
            _render_analysis(analysis)

with tab2:
    st.subheader("📋 歷史分析記錄")
    if not st.session_state.history:
        st.info("暫無記錄")
    else:
        history = st.session_state.history
        if len(history) >= 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[h["name"] for h in history],
                y=[h["result"].get("overall_score", 0) for h in history],
                name="整體評分", mode="lines+markers",
            ))
            fig.update_layout(title="進度趨勢", yaxis=dict(range=[0, 10]), height=250)
            st.plotly_chart(fig, use_container_width=True)
        for record in reversed(history):
            r = record["result"]
            score = r.get("overall_score", 0)
            prob = r.get("success_probability", 0)
            with st.expander(f"{_score_color(score)} {record['name']} · {score}/10 · {prob}%"):
                _render_analysis(r)
        if st.button("清除所有歷史記錄"):
            st.session_state.history = []
            st.rerun()

with tab3:
    st.subheader("📦 批量分析")
    st.caption("用 === 分隔多段對話")
    batch_input = st.text_area("輸入多段對話", height=250, placeholder="銷售員：你好...\n===\n銷售員：你好...")
    if st.button("🚀 開始批量分析", type="primary", disabled=not batch_input.strip()):
        if not st.session_state.gemini_key:
            st.error("請先在側邊欄輸入 Gemini API Key")
        else:
            segs = [s.strip() for s in batch_input.split("===") if s.strip()]
            st.info(f"共 {len(segs)} 段")
            a = get_analyzer()
            a.update_api_key(st.session_state.gemini_key)
            a.update_model(st.session_state.gemini_model)
            results = []
            prog = st.progress(0)
            for i, seg in enumerate(segs):
                with st.spinner(f"分析第 {i+1}/{len(segs)} 段..."):
                    results.append(a.analyze(seg))
                    prog.progress((i+1)/len(segs))
            st.success(f"完成！共 {len(results)} 段")
            rows = [{"#": i+1, "整體評分": r.get("overall_score", 0), "成功率": f"{r.get('success_probability',0)}%", "情緒": r.get("customer_sentiment",{}).get("overall","-")} if "error" not in r else {"#": i+1, "整體評分": "錯誤", "成功率": "-", "情緒": "-"} for i, r in enumerate(results)]
            st.dataframe(rows, use_container_width=True)
            for i, (seg, r) in enumerate(zip(segs, results)):
                if "error" not in r:
                    st.session_state.history.append({"name": f"批量#{len(st.session_state.history)+1}", "transcript": seg, "result": r})
            for i, (seg, r) in enumerate(zip(segs, results)):
                with st.expander(f"第 {i+1} 段"):
                    if "error" in r:
                        st.error(r["error"])
                    else:
                        _render_analysis(r)
