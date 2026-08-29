"""
api.py
FastAPI REST backend for the optimizer.
Run: uvicorn api:app --reload --port 8000
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from traffic_model import TrafficNetwork
from optimizer import HybridRouteOptimizer
import time

app = FastAPI(title="SIH Traffic Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load network
_network = None

def get_network():
    global _network
    if _network is None:
        _network = TrafficNetwork(dist=3000)
    return _network


class RouteRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    pop_size: int = 50
    ngen: int = 30
    k_candidates: int = 8


class RouteResponse(BaseModel):
    route_nodes: List[int]
    distance_km: float
    time_min: float
    congestion: float
    composite_cost: float
    solve_time_sec: float
    generations: int
    convergence_history: List[float]


@app.get("/health")
def health():
    return {"status": "ok", "network_loaded": _network is not None}


@app.post("/optimize", response_model=RouteResponse)
def optimize_route(req: RouteRequest):
    """Main optimization endpoint."""
    network = get_network()
    origin = (req.origin_lat, req.origin_lon)
    destination = (req.dest_lat, req.dest_lon)

    # Get candidate routes
    candidates = network.k_shortest_paths(origin, destination, k=req.k_candidates)

    if len(candidates) < 2:
        return RouteResponse(
            route_nodes=[], distance_km=0, time_min=0,
            congestion=0, composite_cost=0, solve_time_sec=0,
            generations=0, convergence_history=[]
        )

    def cost_func(indices):
        total = 0.0
        for idx in indices:
            m = network.route_metrics(candidates[idx])
            total += m['time_min'] + m['distance_km'] * 0.5 + m['congestion'] * 10
        return total

    start = time.time()
    opt = HybridRouteOptimizer(
        candidate_routes=candidates,
        cost_func=cost_func,
        pop_size=req.pop_size,
        ngen=req.ngen
    )
    result = opt.optimize()
    elapsed = time.time() - start

    best_route = result['best_route']
    metrics = network.route_metrics(best_route)

    return RouteResponse(
        route_nodes=best_route,
        distance_km=metrics['distance_km'],
        time_min=metrics['time_min'],
        congestion=metrics['congestion'],
        composite_cost=result['best_cost'],
        solve_time_sec=round(elapsed, 3),
        generations=result['generations'],
        convergence_history=result['history']
    )


@app.post("/inject_congestion")
def inject_congestion(lat: float = Query(...), lon: float = Query(...),
                      severity: float = Query(2.5)):
    """Inject a congestion event (for demo)."""
    network = get_network()
    affected = network.simulate_congestion_wave((lat, lon), severity=severity)
    return {"affected_edges": affected, "severity": severity}


@app.post("/reset")
def reset():
    """Reset all congestion."""
    from traffic_model import TrafficState
    global _network
    if _network:
        _network.congestion = TrafficState()
        _network._apply_congestion()
    return {"status": "reset"}   