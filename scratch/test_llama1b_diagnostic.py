import ollama
import sys
import time

def test_model(model_name):
    questions = [
        {
            "query": "What does the Transformer dispense with entirely?",
            "context": "the Transformer relies entirely on an attention mechanism to draw global dependencies between input and output, dispensing with recurrence and convolutions entirely.",
            "label": "Q16 (Pluralization)"
        },
        {
            "query": "How many days was the model trained for on eight GPUs?",
            "context": "the model was trained for 3.5 days on eight GPUs.",
            "label": "Q18 (Units)"
        },
        {
            "query": "What attention mechanism outperforms dot product attention without scaling for larger values of dk?",
            "context": "additive attention outperforms dot product attention without scaling for larger values of dk.",
            "label": "Q19 (Synthesis)"
        }
    ]
    
    print(f"--- Testing {model_name} ---")
    for q in questions:
        print(f"[{q['label']}] Query: {q['query']}")
        start_time = time.time()
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': f"Answer in one short sentence.\nContext: {q['context']}\nQuestion: {q['query']}"}],
                options={'temperature': 0.0, 'num_predict': 100}
            )
            elapsed = time.time() - start_time
            print(f"Response: '{response['message']['content'].strip()}'")
            print(f"Latency: {elapsed:.2f}s")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 10)

if __name__ == "__main__":
    test_model('llama3.2:1b')
