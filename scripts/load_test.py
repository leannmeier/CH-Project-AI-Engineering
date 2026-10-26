"""Prueba de concurrencia: dispara N peticiones a la vez y mide cuánto tarda cada job.

Uso (con la API levantada):
    python scripts/load_test.py                # 5 peticiones, sin aprobar
    python scripts/load_test.py --approve      # aprueba automáticamente los jobs que esperan aprobación
    python scripts/load_test.py -n 5 --stagger 2

Ojo con los límites del plan gratuito de Groq: varias corridas simultáneas del sistema multi-agente
pueden pasarse del límite de tokens por minuto (429). Si pasa, el job queda en FAILED con
error_type "transient" y eso también es parte de lo que se quiere observar. Usá --stagger para
espaciar las peticiones si querés evitarlo.
"""

import argparse
import asyncio
import time

import httpx

TERMINAL = {"DONE", "FAILED", "REJECTED"}

QUERIES = [
    "Resume en tres puntos qué es el protocolo MQTT.",
    "Busca qué es una base de datos vectorial.",
    "Explica brevemente qué es el patrón supervisor en sistemas multi-agente.",
    "Busca qué es Redis Stack y para qué sirve.",
    "Resume qué es la observabilidad en aplicaciones de IA.",
]


async def run_one(client: httpx.AsyncClient, index: int, query: str, approve: bool, timeout: float, poll: float) -> dict:
    started = time.perf_counter()
    response = await client.post("/tasks", json={"query": query})
    response.raise_for_status()
    job_id = response.json()["job_id"]
    enqueue_seconds = time.perf_counter() - started  # el POST debe responder casi al instante

    approved_sent = False
    status, job = "PENDING", {}
    while True:
        job = (await client.get(f"/tasks/{job_id}")).json()
        status = job["status"]

        if status == "WAITING_APPROVAL" and approve and not approved_sent:
            await client.post(f"/tasks/{job_id}/approve", json={"approved": True, "comment": "auto (load test)"})
            approved_sent = True
        elif status in TERMINAL or (status == "WAITING_APPROVAL" and not approve):
            break

        if time.perf_counter() - started > timeout:
            status = "TIMEOUT"
            break
        await asyncio.sleep(poll)

    return {
        "n": index,
        "job_id": job_id[:8],
        "status": status,
        "enqueue_s": enqueue_seconds,
        "total_s": time.perf_counter() - started,
        "error": (job.get("error") or "")[:80],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba de concurrencia de la API multi-agente")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("-n", type=int, default=5, help="cantidad de peticiones simultáneas")
    parser.add_argument("--approve", action="store_true", help="aprobar automáticamente los jobs en WAITING_APPROVAL")
    parser.add_argument("--stagger", type=float, default=0.0, help="segundos de espera entre peticiones")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--poll", type=float, default=1.0)
    args = parser.parse_args()

    print(f"Lanzando {args.n} peticiones a {args.base_url} (approve={args.approve}, stagger={args.stagger}s)...\n")
    wall_start = time.perf_counter()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=30.0) as client:
        tasks = []
        for i in range(args.n):
            query = QUERIES[i % len(QUERIES)]
            tasks.append(asyncio.create_task(run_one(client, i + 1, query, args.approve, args.timeout, args.poll)))
            if args.stagger and i < args.n - 1:
                await asyncio.sleep(args.stagger)
        results = await asyncio.gather(*tasks)

    print(f"{'#':<3} {'job':<9} {'estado':<17} {'POST (s)':>9} {'total (s)':>10}  error")
    for r in results:
        print(f"{r['n']:<3} {r['job_id']:<9} {r['status']:<17} {r['enqueue_s']:>9.3f} {r['total_s']:>10.1f}  {r['error']}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nResumen: {counts} | tiempo total: {time.perf_counter() - wall_start:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
