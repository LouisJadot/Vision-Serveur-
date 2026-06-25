import re
import base64
import cv2
from openai import OpenAI

from config import OPENROUTER_API_KEY, VLM_MODEL

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

PROMPT = """Look at this image and return ONLY a JSON array of the distinct object categories you see.
Use short English descriptions starting with "a " (e.g. "a chair", "a laptop", "a water bottle").
Return ONLY the JSON array, no explanation, no markdown.
Example: ["a chair","a laptop","a water bottle"]"""


def generate_labels(frame) -> list[str]:
    """
    Send the frame to the VLM and return a list of labels.
    Returns an empty list in case of error.
    """
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()

    try:
        res = client.chat.completions.create(
            model=VLM_MODEL,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
        )
        raw = res.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error API VLM: {e}")
        return []

    # Cleaning: remove ```json ... ``` blocks if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Extract the first JSON array found
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if not match:
        print(f"VLM response not parseable: {raw!r}")
        return []

    try:
        labels = __import__("json").loads(match.group(0))
        if isinstance(labels, list):
            return [str(l).strip() for l in labels if str(l).strip()]
    except Exception as e:
        print(f"Error parsing VLM JSON: {e} — raw: {raw!r}")

    return []