import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class NodeKey:
    node_type: str
    node_key: str


@dataclass(frozen=True)
class GraphEdge:
    src_type: str
    src_id: str
    dst_type: str
    dst_id: str
    edge_type: str
    weight: int


class SapGraphDB:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path).resolve())
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _fetch_node_label(self, node_type: str, node_key: str) -> Optional[str]:
        row = self.conn.execute(
            """
            SELECT label FROM graph_nodes
            WHERE node_type=? AND node_key=?;
            """,
            (node_type, node_key),
        ).fetchone()
        return row["label"] if row else None

    def get_node_metadata(self, node_type: str, node_key: str) -> Dict[str, Any]:
        row = self.conn.execute(
            """
            SELECT metadata_json FROM graph_node_metadata
            WHERE node_type=? AND node_key=?;
            """,
            (node_type, node_key),
        ).fetchone()
        if not row:
            return {}
        return json.loads(row["metadata_json"]) if row["metadata_json"] else {}

    def get_node_label(self, node_type: str, node_key: str) -> str:
        return self._fetch_node_label(node_type, node_key) or node_key

    def count_nodes(self, node_type: Optional[str] = None) -> int:
        """Count total nodes or nodes of a specific type."""
        if node_type:
            row = self.conn.execute(
                "SELECT COUNT(*) as count FROM graph_nodes WHERE node_type=?;",
                (node_type,),
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) as count FROM graph_nodes;").fetchone()
        return row["count"]

    def expand_neighborhood(
        self,
        start_type: str,
        start_key: str,
        depth: int = 2,
        max_nodes: int = 200,
        include_reverse: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Returns (nodes, edges) for visualization, with BFS-like expansion.
        """
        start = NodeKey(start_type, start_key)
        visited: Set[NodeKey] = {start}
        frontier: Set[NodeKey] = {start}
        all_edges: List[GraphEdge] = []

        for _level in range(depth):
            if not frontier:
                break
            next_frontier: Set[NodeKey] = set()

            frontier_list = list(frontier)
            for nk in frontier_list:
                # forward edges
                rows = self.conn.execute(
                    """
                    SELECT src_type, src_id, dst_type, dst_id, edge_type, weight
                    FROM graph_edges
                    WHERE src_type=? AND src_id=?;
                    """,
                    (nk.node_type, nk.node_key),
                ).fetchall()
                for r in rows:
                    e = GraphEdge(
                        src_type=r["src_type"],
                        src_id=r["src_id"],
                        dst_type=r["dst_type"],
                        dst_id=r["dst_id"],
                        edge_type=r["edge_type"],
                        weight=int(r["weight"] or 1),
                    )
                    all_edges.append(e)
                    dst = NodeKey(e.dst_type, e.dst_id)
                    if dst not in visited:
                        if len(visited) >= max_nodes:
                            continue
                        visited.add(dst)
                        next_frontier.add(dst)

                if include_reverse:
                    rows = self.conn.execute(
                        """
                        SELECT src_type, src_id, dst_type, dst_id, edge_type, weight
                        FROM graph_edges
                        WHERE dst_type=? AND dst_id=?;
                        """,
                        (nk.node_type, nk.node_key),
                    ).fetchall()
                    for r in rows:
                        e = GraphEdge(
                            src_type=r["src_type"],
                            src_id=r["src_id"],
                            dst_type=r["dst_type"],
                            dst_id=r["dst_id"],
                            edge_type=r["edge_type"],
                            weight=int(r["weight"] or 1),
                        )
                        all_edges.append(e)
                        src = NodeKey(e.src_type, e.src_id)
                        if src not in visited:
                            if len(visited) >= max_nodes:
                                continue
                            visited.add(src)
                            next_frontier.add(src)

            frontier = next_frontier
            if len(visited) >= max_nodes:
                break

        nodes_out: List[Dict[str, Any]] = []
        for nk in visited:
            nodes_out.append(
                {
                    "node_type": nk.node_type,
                    "node_key": nk.node_key,
                    "label": self.get_node_label(nk.node_type, nk.node_key),
                    "metadata": self.get_node_metadata(nk.node_type, nk.node_key),
                }
            )

        # Keep only edges where both ends are in visited set
        edges_out: List[Dict[str, Any]] = []
        for e in all_edges:
            if NodeKey(e.src_type, e.src_id) in visited and NodeKey(e.dst_type, e.dst_id) in visited:
                edges_out.append(
                    {
                        "src_type": e.src_type,
                        "src_id": e.src_id,
                        "dst_type": e.dst_type,
                        "dst_id": e.dst_id,
                        "edge_type": e.edge_type,
                        "weight": e.weight,
                    }
                )

        # Deduplicate edges (could be many duplicates from item-level edges)
        seen: Set[Tuple[str, str, str, str, str]] = set()
        dedup_edges_out: List[Dict[str, Any]] = []
        for ed in edges_out:
            k = (ed["src_type"], ed["src_id"], ed["dst_type"], ed["dst_id"], ed["edge_type"])
            if k in seen:
                continue
            seen.add(k)
            dedup_edges_out.append(ed)

        return nodes_out, dedup_edges_out

