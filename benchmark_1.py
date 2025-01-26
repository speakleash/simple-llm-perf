import os
import time
import concurrent.futures
from openai import OpenAI
import sys
import statistics
import argparse


parser = argparse.ArgumentParser(description='Benchmark VLLM OpenAI client')
parser.add_argument('--base-url', type=str, required=True, help='VLLM server base URL')
parser.add_argument('--api-key', type=str, default="EMPTY", help='API key (default: EMPTY)')
parser.add_argument('--model', type=str, default="gpt-4o", help='Model name')

args = parser.parse_args()

BASE_URL = args.base_url
MODEL= args.model
API_KEY= args.api_key
THREAD_COUNTS = [1, 5, 10, 20]

base_dir = os.path.dirname(os.path.abspath(__file__))
example_text_path = os.path.join(base_dir, 'benchmark_1.txt')

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def summarize_text(text: str):
    messages = [
        {"role": "system", "content": "Jesteś pomocnym asystentem."},
        {"role": "user", "content": f"Proszę o streszczenie następującego tekstu:\n\n {text}"}
    ]
    
    stream_start_time = time.time()
    first_token_time = None
    
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        temperature=0.0,
        max_tokens=2000,
        top_p=1
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            if first_token_time is None:
                first_token_time = time.time() - stream_start_time
            break  
    
    no_stream_start_time = time.time()
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=False,
        temperature=0.0,
        max_tokens=2000,
        top_p=1
    )
    
    total_time = time.time() - no_stream_start_time
    
    return {
        "first_token_time": first_token_time,
        "total_time": total_time,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "tokens_per_second": response.usage.completion_tokens/total_time,
        "text": response.choices[0].message.content
    }

def run_benchmark(text: str, num_threads: int = 3):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(summarize_text, text) for _ in range(num_threads)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    first_token_times = [r['first_token_time'] for r in results]
    total_times = [r['total_time'] for r in results]
    prompt_tokens = [r['prompt_tokens'] for r in results]
    completion_tokens = [r['completion_tokens'] for r in results]
    tokens_per_second = [r['tokens_per_second'] for r in results]
    
    stats = {
        'first_token_time': {
            'mean': statistics.mean(first_token_times),
            'min': min(first_token_times),
            'max': max(first_token_times),
            'stdev': statistics.stdev(first_token_times) if len(first_token_times) > 1 else 0
        },
        'total_time': {
            'mean': statistics.mean(total_times),
            'min': min(total_times),
            'max': max(total_times),
            'stdev': statistics.stdev(total_times) if len(total_times) > 1 else 0
        },
        'prompt_tokens': {
            'mean': statistics.mean(prompt_tokens),
            'min': min(prompt_tokens),
            'max': max(prompt_tokens)
        },
        'completion_tokens': {
            'mean': statistics.mean(completion_tokens),
            'min': min(completion_tokens),
            'max': max(completion_tokens)
        },
        'tokens_per_second': {
            'mean': statistics.mean(tokens_per_second),
            'min': min(tokens_per_second),
            'max': max(tokens_per_second),
            'stdev': statistics.stdev(tokens_per_second) if len(tokens_per_second) > 1 else 0
        }
    }
    
    return stats


if os.path.exists(example_text_path):
    with open(example_text_path, 'r') as file:
        example_text = file.read()
else:
    print(f"File {example_text_path} does not exist")
    sys.exit(1)

lines = example_text.split('\n')
all = lines
half = lines[:len(lines)//2]
quarter = lines[:len(lines)//4]


texts = {
    "25%": '\n'.join(quarter),
    "50%": '\n'.join(half),
    "100%": '\n'.join(all)
}

for num_threads in THREAD_COUNTS:
    print(f"\nRozpoczynam testy dla {num_threads} wątków...")
    
    results_path = os.path.join(base_dir, f'test_{num_threads}.tsv')
    with open(results_path, 'w') as f:
        f.write("input_tokens\toutput_tokens\ttokens_per_second\ttime_to_first_token\tthreads\n")
    
    for name, text in texts.items():
        print(f"Test dla {name} tekstu...")
        stats = run_benchmark(text, num_threads=num_threads)
        
        # Zapisz wyniki do pliku
        with open(results_path, 'a') as f:
            f.write(f"{stats['prompt_tokens']['mean']:.0f}\t"
                    f"{stats['completion_tokens']['mean']:.0f}\t"
                    f"{stats['tokens_per_second']['mean']:.2f}\t"
                    f"{stats['first_token_time']['mean']:.2f}\t"
                    f"{num_threads}\n")

print(f"\nTesty zakończone. Wyniki zostały zapisane w plikach test_X.tsv")
