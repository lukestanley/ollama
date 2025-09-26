import time
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="")
def time_request(sample_count, use_stream, label, content):
    print(f"Starting {label}...")
    start = time.perf_counter()
    payload = {"model":"gemma3:270m-it-qat","messages":[{"role":"user","content":content}],"max_tokens":160,"n":sample_count,"stream":use_stream, "temperature":0.1}
    if use_stream:
        seen_indexes = set()
        for chunk in client.chat.completions.create(**payload):
            for ch in chunk.choices:
                idx = getattr(ch, "index", None)
                label_text = "?" if idx is None else idx
                if label_text not in seen_indexes:
                    if seen_indexes:
                        print()
                    print(f"[{label_text}] ", end="", flush=True)
                    seen_indexes.add(label_text)
                text = getattr(ch.delta, "content", None)
                if text:
                    print(text, end="", flush=True)
        print()
    if not use_stream:
        resp = client.chat.completions.create(**payload)
        for choice in resp.choices:
            idx = getattr(choice, "index", None)
            label_text = "?" if idx is None else idx
            text = choice.message.content or ""
            print(f"[{label_text}] {text}")
    duration = time.perf_counter() - start
    print(f"\nDone {label} in {duration:.3f}s")
    return duration
print("Warm up request (n=1, non-stream)...")
time_request(1, False, "warm-up", "Count from 1 to 3")
print("Warm up done.\n")
content = "Count from 1 to 5"
time_single_non_stream = time_request(1, False, "non-stream n=1", content)
time_parallel_non_stream = time_request(3, False, "non-stream n=3", content)
time_parallel_stream = time_request(3, True, "stream n=3", content)
parallel_ratio = time_parallel_non_stream / time_single_non_stream
parity_ratio = time_parallel_stream / time_parallel_non_stream
print("Results:")
print(f"n=1 non-stream: {time_single_non_stream:.3f}s")
print(f"n=3 non-stream: {time_parallel_non_stream:.3f}s (n3/n1 = {parallel_ratio:.3f})")
print(f"n=3 stream:     {time_parallel_stream:.3f}s (stream/non-stream n3 = {parity_ratio:.3f})")
reasons = []
if parallel_ratio > 1.8:
    reasons.append("Parallel non-stream too slow compared to n=1.")
if parity_ratio > 1.15:
    reasons.append("Stream slower than non-stream for n=3.")
if reasons:
    print("\n".join(reasons))
    print("RESULT: FAIL")
if not reasons:
    print("RESULT: PASS")
