#!/usr/bin/env python3
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

hosts = sys.argv[1:]
if not hosts:
    sys.exit(f"USAGE: {sys.argv[0]} [host1] [host2] ...")

ts = subprocess.run(
    ["tailscale", "status"], check=True, text=True, capture_output=True
).stdout.splitlines()
ip = {line.split()[1]: line.split()[0] for line in ts if len(line.split()) >= 2}
ips = [ip[h] for h in hosts]
devs = [[h, ip[h]] for h in hosts]


def get_models(a: str) -> set[str]:
    try:
        r = urlopen(f"http://{a}:8000/models", timeout=5)  # pyright: ignore[reportAny]
        return set(json.loads(r.read()))  # pyright: ignore[reportAny]
    except Exception:
        return set()


with ThreadPoolExecutor(len(ips)) as ex:
    models = set[str].intersection(*ex.map(get_models, ips))


def run(h: str, a: str, body: bytes):
    try:
        r = urlopen(  # pyright: ignore[reportAny]
            Request(
                f"http://{a}:8000/run_test",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            ),
            timeout=300,
        )
        for line in r.read().decode(errors="replace").splitlines():  # pyright: ignore[reportAny]
            print(f"\n{h}@{a}: {line}", flush=True)
    except Exception:
        print(f"\n{h}@{a}: request to {h} failed", flush=True)
        raise


print(f"Starting test with {models}")
for model in models:
    body = json.dumps({"devs": devs, "model_id": model}).encode()
    with ThreadPoolExecutor(max_workers=len(hosts)) as ex:
        for _ in ex.map(run, hosts, ips, itertools.repeat(body)):
            pass
