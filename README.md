# Benchmark Scripts

## Roadmap

Currently there are two separate benchmark scripts that will be merged into a single `benchmark.py` in the future.

- [ ] Merge benchmark_1.py and benchmark_2.py into a single `benchmark.py`
- [ ] Add more benchmarks and tasks (generate, summarize, translate, etc.)
- [ ] Save all results and metrics to a file

## Running the benchmarks

### Benchmark 1

Benchmark for testing model responses (author: Sekon):

```bash
python benchmark_1.py --base-url <api url> --model <model name> --api-key <api key>
```

### Benchmark 2

Benchmark for testing model responses (author: Ginterhauser):

```bash
python benchmark_2.py --base-url <api url> --model <model name> --api-key <api key> --max-parallel=<number of threads>
```
