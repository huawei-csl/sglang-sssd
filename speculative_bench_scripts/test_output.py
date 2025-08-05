import threading
import openai

port = 34597
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

def fetch_completion(prompt: str, results: list, index: int) -> None:
    """Send a non-streaming request and save the full reply."""
    response = client.chat.completions.create(
        model="/storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
        stream=False,          # ← no streaming
    )
    # Store the assistant’s reply in the correct slot
    results[index] = response.choices[0].message.content.strip()

# The prompts we want to run
prompts = [
    "List 3 countries and their capitals, and explain why you chose them.",
    "Explain the significance of the moon landing in 1969.",
    "Who are you?",
]

results = [None] * len(prompts)      # placeholder for replies
threads = []

# Launch each prompt in its own thread
for i, prompt in enumerate(prompts):
    t = threading.Thread(target=fetch_completion, args=(prompt, results, i))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

# Print every reply, neatly separated
print("\n=== Completed Responses ===")
for prompt, reply in zip(prompts, results):
    print(f"\n--- Prompt: {prompt} ---\n{reply}\n")
