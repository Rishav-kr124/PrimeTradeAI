import google.generativeai as genai
import streamlit as st

# Setup the key
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

print("------------------------------------------------")
print("SEARCHING FOR AVAILABLE MODELS...")
print("------------------------------------------------")

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ FOUND: {m.name}")
except Exception as e:
    print(f"❌ ERROR: {e}")

print("------------------------------------------------")