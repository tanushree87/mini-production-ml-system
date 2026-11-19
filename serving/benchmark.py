"""
benchmark.py
Fires 500 POST requests to /predict and reports latency percentiles.
Samples real rows from the test partition to get realistic payloads.

Run (with the API already running on port 8000):
    python serving/benchmark.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL    = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"
N_REQUESTS   = 500
RANDOM_SEED  = 42
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "artifacts", "benchmark_results.json"
)

FEATURE_COLS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def build_payloads_from_raw(n: int) -> list:
    """
    Load raw CSV, take the last 15% (test partition) in temporal order,
    and sample n rows. Sends raw Amount/Time so the API scaler is exercised.
    """
    import os
    root     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df       = pd.read_csv(os.path.join(root, "data", "creditcard.csv"))
    df       = df.sort_values("Time").reset_index(drop=True)

    test_start = int(len(df) * 0.85)
    test_df    = df.iloc[test_start:].copy()

    sample = test_df[FEATURE_COLS].sample(n=n, random_state=RANDOM_SEED)
    return sample.to_dict(orient="records")


def check_health():
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        r.raise_for_status()
        print(f"[benchmark] /health → {r.json()}")
    except Exception as e:
        print(f"[benchmark] ERROR: API not reachable. Start uvicorn first.\n  {e}")
        sys.exit(1)


def run_benchmark(payloads: list) -> list:
    latencies_ms = []
    errors       = 0

    print(f"[benchmark] Firing {len(payloads)} requests → {API_URL}")
    for i, payload in enumerate(payloads, 1):
        try:
            t0 = time.perf_counter()
            r  = requests.post(API_URL, json=payload, timeout=10)
            t1 = time.perf_counter()

            if r.status_code == 200:
                latencies_ms.append((t1 - t0) * 1000)
            else:
                errors += 1
                print(f"[benchmark] Request {i} failed: {r.status_code} {r.text}")

        except Exception as e:
            errors += 1
            print(f"[benchmark] Request {i} exception: {e}")

        if i % 10 == 0:
            avg_so_far = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
            print(f"[benchmark] {i:>3}/{len(payloads)} | "
                  f"avg so far: {avg_so_far:.1f} ms | "
                  f"errors: {errors}", flush=True)

    print(f"[benchmark] Done. Errors: {errors}/{len(payloads)}")
    return latencies_ms


def report(latencies_ms: list):
    arr = np.array(latencies_ms)

    results = {
        "n_requests":      len(arr),
        "n_errors":        N_REQUESTS - len(arr),
        "latency_ms": {
            "min":  round(float(np.min(arr)),  3),
            "p50":  round(float(np.percentile(arr, 50)), 3),
            "p95":  round(float(np.percentile(arr, 95)), 3),
            "p99":  round(float(np.percentile(arr, 99)), 3),
            "max":  round(float(np.max(arr)),  3),
            "mean": round(float(np.mean(arr)), 3),
            "std":  round(float(np.std(arr)),  3),
        },
        "throughput_rps": round(1000 / float(np.mean(arr)), 2),
    }

    print("\n" + "="*50)
    print("  LATENCY BENCHMARK RESULTS (ms)")
    print("="*50)
    print(f"  Requests sent      : {results['n_requests']}")
    print(f"  Errors             : {results['n_errors']}")
    print(f"  Min latency        : {results['latency_ms']['min']:.3f} ms")
    print(f"  p50  (median)      : {results['latency_ms']['p50']:.3f} ms")
    print(f"  p95               : {results['latency_ms']['p95']:.3f} ms")
    print(f"  p99               : {results['latency_ms']['p99']:.3f} ms")
    print(f"  Max latency        : {results['latency_ms']['max']:.3f} ms")
    print(f"  Mean latency       : {results['latency_ms']['mean']:.3f} ms")
    print(f"  Throughput         : {results['throughput_rps']:.1f} req/sec")
    print("="*50)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[benchmark] Results saved → {RESULTS_PATH}")
    return results


if __name__ == "__main__":

    # Health check first
    check_health()

    # Build payloads and run
    payloads     = build_payloads_from_raw(N_REQUESTS)
    latencies_ms = run_benchmark(payloads)
    report(latencies_ms)