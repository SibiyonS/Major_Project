#from google import genai

'''client = genai.Client(api_key="AIzaSyCPNNSVmnxrNDZ95fCnOmpIetyvyO6DB6w")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain AI in one sentence"
)

print(response.text)'''

# scripts/google_api.py

# scripts/google_api.py

from google import genai


def list_available_models():
    try:
        # 🔑 Direct API key (ONLY for testing)
        client = genai.Client(api_key="AIzaSyCPNNSVmnxrNDZ95fCnOmpIetyvyO6DB6w")

        print("\n📡 Fetching available Gemini models...\n")

        models = client.models.list()

        count = 0
        for m in models:
            count += 1
            print(f"{count}. Model Name : {m.name}")
            print(f"   Display Name : {getattr(m, 'display_name', 'N/A')}")
            print(f"   Methods      : {getattr(m, 'supported_generation_methods', 'N/A')}")
            print("-" * 60)

        if count == 0:
            print("⚠️ No models found. Check API access.")

    except Exception as e:
        print("\n❌ Error while fetching models:")
        print(e)


if __name__ == "__main__":
    list_available_models()

#AIzaSyCPNNSVmnxrNDZ95fCnOmpIetyvyO6DB6w