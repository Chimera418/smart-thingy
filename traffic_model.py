"""
traffic_model.py
Loads OSM road network and simulates dynamic congestion.
"""
import osmnx as ox
import networkx as nx
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class TrafficState:
    """Holds current congestion multipliers per edge."""
    multipliers: Dict[Tuple, float] = field(default_factory=dict)
    timestamp: float = 0.0

    def get_multiplier(self, u, v, key):
        """Return congestion multiplier for edge (u,v,key)."""
        return self.multipliers.get((u, v, key), 1.0)

    def inject_congestion(self, edge_list, severity=2.0):
        """Inject a congestion event on given edges."""
        for (u, v, k) in edge_list:
            self.multipliers[(u, v, k)] = severity
        self.timestamp = time.time()


class TrafficNetwork:
    """Wraps OSMnx graph with congestion-aware routing."""

    def __init__(self, place: str = "Bengaluru, Karnataka, India",
                 network_type: str = "drive", dist: int = 5000):
        print(f"Loading road network for {place}...")
        # Use a central point for smaller graph (faster demo)
        G = ox.graph_from_point(
            (12.9716, 77.5946),  # Bengaluru center
            dist=dist,
            network_type=network_type
        )
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        self.G = G
        self.congestion = TrafficState()
        self._apply_congestion()

    def _apply_congestion(self):
        """Apply current congestion multipliers to edge travel_time."""
        for u, v, k, data in self.G.edges(keys=True, data=True):
            base_time = data.get('travel_time', data.get('length', 100) / 10)
            mult = self.congestion.get_multiplier(u, v, k)
            data['travel_time'] = base_time * mult

    def get_nodes_coords(self) -> np.ndarray:
        """Return (lat, lon) array for all nodes."""
        return np.array([(data['y'], data['x'])
                         for _, data in self.G.nodes(data=True)])

    def nearest_node(self, lat: float, lon: float):
        """Find nearest node to a lat/lon point."""
        return ox.distance.nearest_nodes(self.G, X=lon, Y=lat)

    def shortest_path(self, origin: Tuple, destination: Tuple,
                      weight: str = "travel_time") -> List[int]:
        """Get shortest path using current (congested) weights."""
        orig_node = self.nearest_node(*origin)
        dest_node = self.nearest_node(*destination)
        return nx.shortest_path(self.G, orig_node, dest_node, weight=weight)

    def k_shortest_paths(self, origin: Tuple, destination: Tuple,
                         k: int = 5, weight: str = "travel_time") -> List[List[int]]:
        """Get K shortest paths (candidate routes for metaheuristic)."""
        orig_node = self.nearest_node(*origin)
        dest_node = self.nearest_node(*destination)
        return list(nx.shortest_simple_paths(
            self.G, orig_node, dest_node, weight=weight
        ))[:k]

    def route_metrics(self, path: List[int]) -> Dict[str, float]:
        """Compute distance, time, congestion score for a route."""
        edges = list(zip(path[:-1], path[1:]))
        total_dist = 0.0
        total_time = 0.0
        congestion_score = 0.0

        for u, v in edges:
            # Get edge data (handle multi-edges)
            for k, data in self.G.edges[u, v].items():
                total_dist += data.get('length', 0)
                total_time += data.get('travel_time', 0)
                congestion_score += self.congestion.get_multiplier(u, v, k) - 1.0
                break  # take first edge if multi-edge

        return {
            'distance_km': total_dist / 1000.0,
            'time_min': total_time / 60.0,
            'congestion': congestion_score,
            'composite': total_time / 60.0 + congestion_score * 5  # weighted
        }

    def simulate_congestion_wave(self, center: Tuple, radius_m: float = 1000,
                                  severity: float = 2.5):
        """Simulate a congestion event (e.g., accident) near a point."""
        center_node = self.nearest_node(*center)
        affected_edges = []
        for u, v, k, data in self.G.edges(keys=True, data=True):
            if u == center_node or v == center_node:
                affected_edges.append((u, v, k))
            # Also check neighbors within radius
            elif data.get('length', 0) < radius_m:
                u_coord = self.G.nodes[u]
                v_coord = self.G.nodes[v]
                dist_u = ((u_coord['y'] - center[0])**2 +
                         (u_coord['x'] - center[1])**2) ** 0.5
                if dist_u < radius_m / 111000:  # rough lat/lon to meters
                    affected_edges.append((u, v, k))

        self.congestion.inject_congestion(affected_edges, severity)
        self._apply_congestion()
        return len(affected_edges)   