import os, time, json
from openai import OpenAI

def summarize_monthly(prompt, retries=3):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI summary disabled: set OPENAI_API_KEY to enable monthly summaries."

    client = OpenAI(api_key=api_key)
    system = (
        "You summarize complaint analytics. Return concise plain-English business insights. "
        "Do not invent metrics. Mention uncertainty when data is limited."
    )
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
                input=[{"role":"system","content":system},
                       {"role":"user","content":prompt}],
                max_output_tokens=250,
            )
            text = response.output_text.strip()
            if 20 <= len(text) <= 2500:
                return text
        except Exception:
            if attempt == retries - 1:
                return "Summary generation failed after retries."
            time.sleep(1.5 * (attempt + 1))
    return "Summary generation failed."
