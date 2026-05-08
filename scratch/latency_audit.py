import time
import ollama

def time_query(model_name):
    print(f"Timing {model_name}...")
    prompt = "System Facts:\n- The apple tree originally comes from Kazakhstan.\n\nUser Query: Where does the apple tree originally come from?"
    
    # Warm up
    ollama.chat(model=model_name, messages=[{"role": "user", "content": "hi"}])
    
    start = time.time()
    response = ollama.chat(
        model=model_name, 
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0}
    )
    end = time.time()
    
    latency = end - start
    content = response.get('message', {}).get('content', '').strip()
    print(f"  Result: {content}")
    print(f"  Latency: {latency:.2f}s")
    return latency

models = ["smollm2:360m", "qwen3:0.6b", "llama3.2:1b", "llama3.2:latest"]
results = {}

for m in models:
    try:
        results[m] = time_query(m)
    except Exception as e:
        print(f"Error timing {m}: {e}")

print("\nFINAL LATENCY AUDIT:")
for m, l in results.items():
    print(f"{m:<15} | {l:.2f}s")
