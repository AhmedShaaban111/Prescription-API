"""
check_models.py
---------------
Lists all embedding models available for your Gemini API key,
then tests a small embed_content call to confirm which one works.
Run this before using drug_pipeline.py to find the correct model name.
"""

import os
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

print("=" * 60)
print("Available embedding models on your API key:")
print("=" * 60)

embed_models = []
for m in client.models.list():
    # filter to models that support embedding
    supported = getattr(m, "supported_actions", None) or []
    if "embedContent" in str(supported) or "embed" in m.name.lower():
        print(f"  {m.name}")
        embed_models.append(m.name)

if not embed_models:
    print("  (none found with 'embed' in name — listing ALL models:)")
    for m in client.models.list():
        print(f"  {m.name}")
    embed_models = [m.name for m in client.models.list()]

print("\n" + "=" * 60)
print("Testing embed_content on each candidate...")
print("=" * 60)

TEST_TEXT = ["aspirin 500mg"]

candidates = [
    "text-embedding-004",
    "models/text-embedding-004",
    "text-embedding-005",
    "models/text-embedding-005",
    "gemini-embedding-exp-03-07",
    "models/gemini-embedding-exp-03-07",
]

# also add whatever was listed above
for m in embed_models:
    if m not in candidates:
        candidates.append(m)

for model_name in candidates:
    try:
        result = client.models.embed_content(model=model_name, contents=TEST_TEXT)
        dim = len(result.embeddings[0].values)
        print(f"  [OK]  '{model_name}'  ->  embedding dim={dim}")
    except Exception as e:
        short = str(e)[:120]
        print(f"  [X]   '{model_name}'  ->  {short}")

print("\nUse the [OK] model name in drug_pipeline.py -> GeminiEmbeddingFunction")
