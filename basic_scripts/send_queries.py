# import openai

# port = 34083
# client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

# # Use stream=True for streaming responses
# response = client.chat.completions.create(
#     model="/storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/",
#     messages=[
#         {"role": "user", "content": "List 3 countries and their capitals, and explain why you chose them."},
#     ],
#     temperature=0,
#     max_tokens=10,
#     stream=True,
# )

# # Handle the streaming output
# for chunk in response:
#     if chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end="", flush=True)
# print()

# PARALLEL REQUESTS

import threading
import openai
import time

port = 38986
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

def stream_request(prompt):
    response = client.chat.completions.create(
        model="/storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
        stream=True,
    )

    print(f"\n--- Streaming for prompt: {prompt} ---")
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n--- Done ---")

# Define two prompts
prompt1 = "List 3 countries and their capitals, and explain why you chose them."
prompt2 = "Explain the significance of the moon landing in 1969."
prompt3 = "Who are you?"

# Launch both requests in separate threads
thread1 = threading.Thread(target=stream_request, args=(prompt1,))
thread2 = threading.Thread(target=stream_request, args=(prompt2,))
thread3 = threading.Thread(target=stream_request, args=(prompt3,))

thread1.start()
thread2.start()

time.sleep(1)
thread3.start()

# Wait for both to finish
thread1.join()
thread2.join()
thread3.join()