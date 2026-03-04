#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

VERSION_RE = re.compile(r"^\d+(?:\.\d+)*$")


def parse_version(version: str) -> tuple[int, ...]:
    if not VERSION_RE.match(version):
        raise ValueError(f"Unsupported version format: {version}")
    return tuple(int(p) for p in version.split("."))


def get_available_versions(package: str) -> set[str]:
    with subprocess.Popen(
        [sys.executable, "-m", "pip", "index", "versions", package],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to list versions for {package}: {stderr.strip()}")

    versions: set[str] = set()
    for token in stdout.replace(",", " ").split():
        token = token.strip()
        if VERSION_RE.match(token):
            versions.add(token)

    if not versions:
        raise RuntimeError(f"No parseable versions found for {package} from pip output.")
    return versions


def get_matching_versions(min_version: str, max_versions: int) -> list[str]:
    min_v = parse_version(min_version)
    jax_versions = get_available_versions("jax")
    jaxlib_versions = get_available_versions("jaxlib")
    candidates = []
    for v in (jax_versions & jaxlib_versions):
        if parse_version(v) >= min_v:
            candidates.append(v)
    candidates.sort(key=parse_version)
    if not candidates:
        raise RuntimeError(f"No common jax/jaxlib versions found >= {min_version}.")
    selected = candidates[-max_versions:]
    return selected


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def install_version_pair(version: str, cwd: Path) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        f"jax=={version}",
        f"jaxlib=={version}",
    ]
    proc = run(cmd, cwd)
    if proc.returncode != 0:
        return False, proc.stderr.strip()
    return True, proc.stdout.strip()


def run_benchmark(cwd: Path, size: int, dtype: str, device: str, gpu_id: int) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "examples/timing_bench.py",
        "--size",
        str(size),
        "--dtype",
        dtype,
        "--device",
        device,
    ]
    if device == "gpu":
        cmd.extend(["--gpu-id", str(gpu_id)])

    proc = run(cmd, cwd)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or proc.stdout.strip()
    return True, proc.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark jaxtomo timings across recent jax/jaxlib versions."
    )
    parser.add_argument("--min-version", type=str, default="0.4.30")
    parser.add_argument("--max-versions", type=int, default=4)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("timing_matrix_results.json"),
        help="JSON report file to write.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_path = args.output
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    print("Discovering matching jax/jaxlib versions...", flush=True)
    versions = get_matching_versions(args.min_version, args.max_versions)
    print(f"Selected versions: {', '.join(versions)}", flush=True)

    results: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo_root),
        "min_version": args.min_version,
        "max_versions": args.max_versions,
        "selected_versions": versions,
        "size": args.size,
        "dtype": args.dtype,
        "device": args.device,
        "gpu_id": args.gpu_id if args.device == "gpu" else None,
        "versions": {},
    }

    for i, version in enumerate(versions, start=1):
        print(f"[{i}/{len(versions)}] Installing jax/jaxlib=={version}...", flush=True)
        version_result: dict[str, object] = {}
        ok, install_out = install_version_pair(version, repo_root)
        if not ok:
            version_result["status"] = "install_failed"
            version_result["error"] = install_out
            results["versions"][version] = version_result
            print(f"[{i}/{len(versions)}] install failed for {version}", flush=True)
            continue

        print(f"[{i}/{len(versions)}] Running benchmark for {version}...", flush=True)
        ok, bench_out = run_benchmark(
            repo_root, args.size, args.dtype, args.device, args.gpu_id
        )
        if not ok:
            version_result["status"] = "benchmark_failed"
            version_result["error"] = bench_out
            results["versions"][version] = version_result
            print(f"[{i}/{len(versions)}] benchmark failed for {version}", flush=True)
            continue

        try:
            parsed = json.loads(bench_out)
        except json.JSONDecodeError:
            version_result["status"] = "benchmark_failed"
            version_result["error"] = f"non_json_output: {bench_out}"
            results["versions"][version] = version_result
            print(f"[{i}/{len(versions)}] benchmark output parse failed for {version}", flush=True)
            continue

        version_result["status"] = "ok"
        version_result["result"] = parsed
        results["versions"][version] = version_result
        print(f"[{i}/{len(versions)}] done {version}", flush=True)

    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote timing report to {output_path}", flush=True)


if __name__ == "__main__":
    main()
