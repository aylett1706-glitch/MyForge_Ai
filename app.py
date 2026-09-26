import os
import gradio as gr
import requests

# ── Configuration ────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PREAMBLE = """You are Forge AI, an intelligent assistant for an Australian disability support platform.
You are careful, factual, and professional. Answer truthfully and concisely using verifiable information.
If you are not certain, say so plainly — do not invent regulations, legislation, medical advice, or data.
Use markdown formatting: **bold**, bullet points, and clear sections where helpful.
Always note that critical decisions should be verified with qualified professionals."""

PROMPT_CATEGORIES = [
    {
        "label": "Participants & Care",
        "prompts": [
            "Summarise today's shift notes and progress notes. List key highlights and anything requiring follow-up.",
            "Give me a wellbeing snapshot. Include mood trends, health concerns, and quality-of-life observations.",
            "Identify participants with active alerts or overdue assessments. Summarise key risks and recommend actions.",
            "Draft a goal progress report for the current month."
        ]
    },
    {
        "label": "Compliance & Risk",
        "prompts": [
            "Review recent incidents. Flag any that may involve restrictive practices or mandatory reporting obligations.",
            "Assess readiness against NDIS Practice Standards. Identify gaps and give a readiness score out of 10.",
            "Summarise worker compliance status. Who has expiring credentials or overdue training?",
            "List any potential mandatory reporting obligations that may apply to recent incidents."
        ]
    },
    {
        "label": "Funding & Finance",
        "prompts": [
            "Forecast which participants are at risk of running out of funding before their plan end date.",
            "Summarise budget utilisation. Flag those over 80% with >3 months remaining and those under 40%.",
            "Check for billing anomalies — unsubmitted claims, duplicate charges, or shifts without matching timesheets."
        ]
    },
    {
        "label": "Rostering & Operations",
        "prompts": [
            "Identify any shifts this week that are unassigned or have no confirmed worker.",
            "Analyse worker workloads. Who is overworked and who has capacity?",
            "Generate a comprehensive end-of-day handover summary.",
            "List participants with NDIS plan reviews due in the next 60 days."
        ]
    }
]

# ── AI Call Function ─────────────────────────────────────────
def get_ai_response(user_message, history_messages):
    """
    history_messages: list of {"role": "user"/"assistant", "content": str}
    in Gradio's "messages" chat format.
    """
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY not set. Add it in Space → Settings → Secrets."

    messages = [{"role": "system", "content": SYSTEM_PREAMBLE}]
    # keep the last 6 turns (12 messages) of prior context
    messages.extend(history_messages[-12:])
    messages.append({"role": "user", "content": user_message})

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 768
            },
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        return f"⚠️ API Error {resp.status_code}: {resp.text[:80]}"
    except Exception as e:
        return f"⚠️ Connection error: {str(e)}"

# ── Build UI ──────────────────────────────────────────────────
with gr.Blocks(title="ForgeSync AI") as demo:
    gr.Markdown("""
    # 🧠 ForgeSync AI
    Your intelligent assistant for NDIS care operations — replies in ~1–2 seconds
    """)

    # Quick Prompts
    selected_prompt = gr.State("")
    with gr.Accordion("💡 Quick Prompts", open=True):
        for cat in PROMPT_CATEGORIES:
            gr.Markdown(f"**{cat['label']}**")
            with gr.Row():
                for p in cat["prompts"]:
                    gr.Button(p, size="sm").click(
                        lambda text=p: text, outputs=selected_prompt
                    )

    # Chat — Gradio 6.x uses the "messages" format natively; no type= argument needed/accepted
    chatbot = gr.Chatbot(height=450)

    msg = gr.Textbox(
        placeholder="Ask anything about participants, compliance, funding, rostering...",
        label="Your message"
    )

    def respond(message, history, quick_text):
        text_to_send = quick_text if quick_text else message
        if not text_to_send.strip():
            return "", history, ""

        history = history + [{"role": "user", "content": text_to_send}]
        reply = get_ai_response(text_to_send, history[:-1])
        history = history + [{"role": "assistant", "content": reply}]

        return "", history, ""

    msg.submit(
        respond,
        inputs=[msg, chatbot, selected_prompt],
        outputs=[msg, chatbot, selected_prompt]
    )

    gr.Markdown("*Always verify critical decisions with qualified professionals.*")

# ── Launch ────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Render sets PORT; falls back to 7860 locally
    demo.launch(server_name="0.0.0.0", server_port=port)
