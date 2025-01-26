import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

import tiktoken
from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table


@dataclass
class BenchmarkResult:
    time_to_first_token: float
    tokens_per_second: float
    total_tokens: int
    input_tokens: int
    total_time: float

def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in the input text."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

async def generate_completion(
    client: AsyncOpenAI,
    prompt: str,
    model: str = "gpt-4o",
) -> BenchmarkResult:
    # Count input tokens
    input_tokens = count_tokens(prompt, model)
    
    start_time = time.time()
    first_token_received = False
    first_token_time = 0
    token_count = 0
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=2000,
    )
    
    async for chunk in response:
        if not first_token_received:
            first_token_time = time.time() - start_time
            first_token_received = True
            
        if chunk.choices[0].delta.content is not None:
            token_count += 1
    
    total_time = time.time() - start_time
    tokens_per_second = token_count / total_time if total_time > 0 else 0
    
    return BenchmarkResult(
        time_to_first_token=first_token_time,
        tokens_per_second=tokens_per_second,
        total_tokens=token_count,
        input_tokens=input_tokens,
        total_time=total_time
    )

async def run_parallel_benchmark(
    client: AsyncOpenAI,
    prompt: str,
    num_parallel: int,
    model: str = "gpt-4o",
) -> list[BenchmarkResult]:
    tasks = [
        generate_completion(client, prompt, model)
        for _ in range(num_parallel)
    ]
    return await asyncio.gather(*tasks)

def display_results(parallel_results: dict[int, list[BenchmarkResult]]):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    
    table.add_column("Parallel Requests")
    table.add_column("Input Tokens")
    table.add_column("Output Tokens")
    table.add_column("Time to First Token (s)")
    table.add_column("Tokens/Second")
    table.add_column("Total Time (s)")
    
    for num_parallel, results in parallel_results.items():
        avg_first_token = statistics.mean(r.time_to_first_token for r in results)
        avg_tokens_per_sec = statistics.mean(r.tokens_per_second for r in results)
        total_output_tokens = sum(r.total_tokens for r in results)
        input_tokens = results[0].input_tokens  # Same for all requests
        avg_total_time = statistics.mean(r.total_time for r in results)
        
        table.add_row(
            str(num_parallel),
            str(input_tokens),
            str(total_output_tokens),
            f"{avg_first_token:.3f}",
            f"{avg_tokens_per_sec:.1f}",
            f"{avg_total_time:.3f}"
        )
    
    console.print(table)

async def main():
    parser = argparse.ArgumentParser(description='Benchmark VLLM OpenAI client')
    parser.add_argument('--base-url', type=str, required=True, help='VLLM server base URL')
    parser.add_argument('--api-key', type=str, default="EMPTY", help='API key (default: EMPTY)')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Model name')
    parser.add_argument('--max-parallel', type=int, default=10, help='Number of parallel requests')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations')
    args = parser.parse_args()

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )
    
    # We need to make a request to warm up the client
    _ = await client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "Hello, world!"}],
        temperature=0.0,
    )

    # Test prompt that will generate a moderate length response
    prompt = """Please explain the concept of quantum entanglement and its implications 
    for quantum computing. Provide specific examples and current research applications."""

    parallel_results = {}
    
    print(f"\nRunning benchmark with {args.max_parallel} parallel requests...")
    iteration_results = []
    
    for i in range(args.iterations):
        print(f"Iteration {i+1}/{args.iterations}")
        results = await run_parallel_benchmark(
            client,
            prompt,
            args.max_parallel,
            args.model
        )
        iteration_results.extend(results)
        
    parallel_results[args.max_parallel] = iteration_results
    
    print("\nBenchmark Results:")
    display_results(parallel_results)

if __name__ == "__main__":
    asyncio.run(main())
