 #!/usr/bin/env python3
"""
graph_store.py -- minimal query library for a typed {nodes, edges} JSON graph.

Declared interface: load, neighbors(id, depth), holes(), subgraph(ids). Deliberately thin --
this is the query surface only; no extraction or build logic lives here.

The graph file path comes from the first CLI argument or the GRAPH_STORE environment variable,
defaulting to ./GRAPH_STORE.json. Any file with the same {nodes, edges} shape works.
"""
import json
import os
from collections import defaultdict, deque

DEFAULT_PATH = os.environ.get("GRAPH_STORE", "GRAPH_STORE.json")


class GraphStore:
    def __init__(self, path: str = DEFAULT_PATH):
        with open(path, encoding="utf-8") as f:
            self.graph = json.load(f)
        self.nodes = {n["id"]: n for n in self.graph["nodes"]}
        self.edges = self.graph["edges"]
        self._out = defaultdict(list)
        self._in = defaultdict(list)
        for e in self.edges:
            self._out[e["src"]].append(e)
            self._in[e["dst"]].append(e)

    # ---- basic accessors ----
    def load(self, node_id: str):
        """Return the node dict for `node_id`, or None."""
        return self.nodes.get(node_id)

    def out_edges(self, node_id: str, edge_type: str = None):
        return [e for e in self._out.get(node_id, []) if edge_type is None or e["type"] == edge_type]

    def in_edges(self, node_id: str, edge_type: str = None):
        return [e for e in self._in.get(node_id, []) if edge_type is None or e["type"] == edge_type]

    def out_degree(self, node_id: str) -> int:
        return len(set(e["dst"] for e in self._out.get(node_id, [])))

    def in_degree(self, node_id: str) -> int:
        return len(set(e["src"] for e in self._in.get(node_id, [])))

    def degree(self, node_id: str) -> int:
        """Undirected degree (distinct neighbors either direction)."""
        return len(self._neighbor_ids(node_id))

    def _neighbor_ids(self, node_id: str):
        return {e["dst"] for e in self._out.get(node_id, [])} | {e["src"] for e in self._in.get(node_id, [])}

    # ---- neighbors(id, depth) -- the radar's inner circle primitive ----
    def neighbors(self, node_id: str, depth: int = 1):
        """
        BFS over UNDIRECTED edges out to `depth` hops. Returns a dict
        {hop_distance: [node_id, ...]} (hop 0 = the seed itself), plus the
        edge that reached each node (for provenance/evidence display).
        """
        if node_id not in self.nodes:
            return {}
        visited = {node_id: 0}
        order = {0: [node_id]}
        reached_via = {}
        q = deque([node_id])
        while q:
            cur = q.popleft()
            d = visited[cur]
            if d >= depth:
                continue
            for e in self._out.get(cur, []) + self._in.get(cur, []):
                nb = e["dst"] if e["src"] == cur else e["src"]
                if nb in visited:
                    continue
                visited[nb] = d + 1
                order.setdefault(d + 1, []).append(nb)
                reached_via[nb] = e
                q.append(nb)
        self._last_reached_via = reached_via # inspectable after a call
        return order

    # ---- holes() ----
    def holes(self, status: str = None):
        """
        All type=hole nodes, optionally filtered by status
        (open|superseded|...). Returns full node dicts.
        """
        return [n for n in self.nodes.values() if n["type"] == "hole" and (status is None or n["status"] == status)]

    # ---- subgraph(ids) ----
    def subgraph(self, ids):
        """
        Induced subgraph: the given node ids + all edges where BOTH
        endpoints are in the set. Returns a fresh {nodes, edges} dict in
        the same schema (a valid stand-alone GRAPH_STORE-shaped object).
        """
        idset = set(ids)
        sub_nodes = [self.nodes[i] for i in idset if i in self.nodes]
        sub_edges = [e for e in self.edges if e["src"] in idset and e["dst"] in idset]
        return {"schema": self.graph.get("schema"), "nodes": sub_nodes, "edges": sub_edges}

    # ---- dashboard helpers (used by the metrics report, not required by
    # the declared 4-method interface but kept here to stay a "tiny query
    # lib" rather than scattering ad hoc analysis scripts) ----
    def weakly_connected_components(self, edge_types=None):
        """
        Connectivity over an UNDIRECTED view of (optionally a subset
        of) edge types. Default: all edge types.
        """
        adj = defaultdict(set)
        use_edges = self.edges if edge_types is None else [e for e in self.edges if e["type"] in edge_types]
        for e in use_edges:
            adj[e["src"]].add(e["dst"])
            adj[e["dst"]].add(e["src"])
        seen = set()
        components = []
        for nid in self.nodes:
            if nid in seen:
                continue
            stack, comp = [nid], set()
            while stack:
                cur = stack.pop()
                if cur in comp:
                    continue
                comp.add(cur)
                stack.extend(adj.get(cur, set()) - comp)
            seen |= comp
            components.append(comp)
        return sorted(components, key=len, reverse=True)

    def hubs(self, top_n: int = 20):
        ranked = sorted(self.nodes, key=lambda nid: -self.degree(nid))
        return [(nid, self.degree(nid)) for nid in ranked[:top_n]]

    def frontier(self, max_degree: int = 1, recent_dates=()):
        """
        Low-degree recent nodes -- fresh work that hasn't been wired
        into the graph yet, i.e. exactly the densification target.
        """
        return [n["id"] for n in self.nodes.values()
                if n.get("date") in recent_dates and self.degree(n["id"]) <= max_degree]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    store = GraphStore(path)
    print(f"nodes={len(store.nodes)} edges={len(store.edges)} "
          f"components={len(store.weakly_connected_components())} holes={len(store.holes())}")
