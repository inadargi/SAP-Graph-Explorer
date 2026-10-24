import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _jsonl_iter(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _infer_schema_from_first_record(data_dir: Path) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Returns (table_name, columns) where columns is [(col_name, sqlite_type), ...].
    We keep all values as TEXT for robustness.
    """
    table_name = data_dir.name
    first_jsonl = next(data_dir.glob("*.jsonl"), None)
    if first_jsonl is None:
        raise RuntimeError(f"No jsonl files found in {data_dir}")

    for rec in _jsonl_iter(first_jsonl):
        keys = list(rec.keys())
        cols = [(k, "TEXT") for k in keys]
        return table_name, cols

    raise RuntimeError(f"Empty jsonl in {first_jsonl}")


def _normalize_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _create_table(conn: sqlite3.Connection, table_name: str, cols: List[Tuple[str, str]]) -> None:
    col_sql = ", ".join([f'"{name}" {typ}' for name, typ in cols])
    conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_sql});')


def _insert_jsonl(conn: sqlite3.Connection, table_name: str, cols: List[str], path: Path, chunk_size: int = 2000) -> None:
    placeholders = ", ".join(["?"] * len(cols))
    col_sql = ", ".join([f'"{c}"' for c in cols])
    insert_sql = f'INSERT INTO "{table_name}" ({col_sql}) VALUES ({placeholders});'

    batch: List[Tuple[Any, ...]] = []
    for rec in _jsonl_iter(path):
        row = tuple(_normalize_value(rec.get(c)) for c in cols)
        batch.append(row)
        if len(batch) >= chunk_size:
            conn.executemany(insert_sql, batch)
            batch.clear()
    if batch:
        conn.executemany(insert_sql, batch)


def _ingest_table_dir(conn: sqlite3.Connection, data_dir: Path) -> None:
    table_name, cols_with_types = _infer_schema_from_first_record(data_dir)
    cols = [c for c, _ in cols_with_types]
    _create_table(conn, table_name, cols_with_types)

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    for p in jsonl_files:
        _insert_jsonl(conn, table_name, cols, p)


def _exec(conn: sqlite3.Connection, sql: str) -> None:
    conn.executescript(sql)


def _build_edges_and_nodes(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS graph_edges;")
    conn.execute("DROP TABLE IF EXISTS graph_nodes;")
    conn.execute("DROP TABLE IF EXISTS graph_node_metadata;")

    conn.execute(
        """
        CREATE TABLE graph_edges (
            edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            src_type TEXT NOT NULL,
            src_id TEXT NOT NULL,
            dst_type TEXT NOT NULL,
            dst_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            weight INTEGER DEFAULT 1
        );
        """
    )
    conn.execute("CREATE INDEX idx_graph_edges_src ON graph_edges(src_type, src_id);")
    conn.execute("CREATE INDEX idx_graph_edges_dst ON graph_edges(dst_type, dst_id);")
    conn.execute("CREATE INDEX idx_graph_edges_type ON graph_edges(edge_type);")

    conn.execute(
        """
        CREATE TABLE graph_nodes (
            node_id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_type TEXT NOT NULL,
            node_key TEXT NOT NULL,
            label TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE UNIQUE INDEX uidx_graph_nodes ON graph_nodes(node_type, node_key);")

    conn.execute(
        """
        CREATE TABLE graph_node_metadata (
            node_type TEXT NOT NULL,
            node_key TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            PRIMARY KEY (node_type, node_key)
        );
        """
    )

    def ensure_node(node_type: str, node_key: str, label: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO graph_nodes(node_type, node_key, label)
            VALUES (?, ?, ?);
            """,
            (node_type, node_key, label),
        )
        if metadata is None:
            metadata = {}
        conn.execute(
            """
            INSERT OR REPLACE INTO graph_node_metadata(node_type, node_key, metadata_json)
            VALUES (?, ?, ?);
            """,
            (node_type, node_key, json.dumps(metadata, ensure_ascii=False)),
        )

    # --- Customers ---
    bp_exists = _table_exists(conn, "business_partners")
    if bp_exists:
        for rec in conn.execute(
            """
            SELECT businessPartner AS k, COALESCE(businessPartnerFullName, businessPartnerName, '') AS label
            FROM business_partners;
            """
        ):
            ensure_node("Customer", str(rec[0]), str(rec[1]) or str(rec[0]), metadata={"customerId": str(rec[0])})

    # --- Products ---
    if _table_exists(conn, "products"):
        # Prefer EN descriptions when available
        desc_join = ""
        if _table_exists(conn, "product_descriptions"):
            desc_join = """
                LEFT JOIN product_descriptions pd
                    ON pd.product = p.product AND pd.language = 'EN'
            """
        for rec in conn.execute(
            f"""
            SELECT p.product AS k,
                   COALESCE(pd.productDescription, p.productOldId, p.productType, '') AS label
            FROM products p
            {desc_join};
            """
        ):
            ensure_node("Product", str(rec[0]), str(rec[1]) or str(rec[0]), metadata={"productId": str(rec[0])})

    # --- Plants ---
    if _table_exists(conn, "plants"):
        for rec in conn.execute(
            """
            SELECT plant AS k, COALESCE(plantName, plant, '') AS label
            FROM plants;
            """
        ):
            ensure_node("Plant", str(rec[0]), str(rec[1]) or str(rec[0]), metadata={"plantId": str(rec[0])})

    # --- Document nodes ---
    for tname, node_type in [
        ("sales_order_headers", "SalesOrder"),
        ("outbound_delivery_headers", "Delivery"),
        ("billing_document_headers", "BillingDocument"),
        ("journal_entry_items_accounts_receivable", "JournalEntry"),
    ]:
        if _table_exists(conn, tname):
            if node_type == "JournalEntry":
                src_col = "accountingDocument"
            elif node_type == "SalesOrder":
                src_col = "salesOrder"
            elif node_type == "Delivery":
                src_col = "deliveryDocument"
            else:  # BillingDocument
                src_col = "billingDocument"

            # label best-effort: use doc id as label
            for rec in conn.execute(f'SELECT DISTINCT "{src_col}" FROM "{tname}";'):
                if rec[0] is None:
                    continue
                ensure_node(node_type, str(rec[0]), str(rec[0]), metadata={"id": str(rec[0])})

    # --- Edges: SalesOrder -> Customer ---
    if _table_exists(conn, "sales_order_headers"):
        conn.execute(
            """
            INSERT INTO graph_edges(src_type, src_id, dst_type, dst_id, edge_type, weight)
            SELECT 'SalesOrder' AS src_type, so.salesOrder AS src_id,
                   'Customer' AS dst_type, so.soldToParty AS dst_id,
                   'sold_to' AS edge_type, 1 AS weight
            FROM sales_order_headers so
            WHERE so.soldToParty IS NOT NULL;
            """
        )

    # --- Edges: SalesOrder -> Delivery ---
    if _table_exists(conn, "outbound_delivery_items"):
        conn.execute(
            """
            INSERT INTO graph_edges(src_type, src_id, dst_type, dst_id, edge_type, weight)
            SELECT 'SalesOrder' AS src_type,
                   di.referenceSdDocument AS src_id,
                   'Delivery' AS dst_type,
                   di.deliveryDocument AS dst_id,
                   'delivered_as' AS edge_type,
                   1 AS weight
            FROM outbound_delivery_items di
            WHERE di.referenceSdDocument IS NOT NULL AND di.deliveryDocument IS NOT NULL;
            """
        )

    # --- Edges: Delivery -> BillingDocument ---
    if _table_exists(conn, "billing_document_items"):
        conn.execute(
            """
            INSERT INTO graph_edges(src_type, src_id, dst_type, dst_id, edge_type, weight)
            SELECT 'Delivery' AS src_type,
                   bi.referenceSdDocument AS src_id,
                   'BillingDocument' AS dst_type,
                   bi.billingDocument AS dst_id,
                   'billed_for' AS edge_type,
                   1 AS weight
            FROM billing_document_items bi
            WHERE bi.referenceSdDocument IS NOT NULL AND bi.billingDocument IS NOT NULL;
            """
        )

    # --- Edges: BillingDocument -> JournalEntry ---
    if _table_exists(conn, "billing_document_headers") and _table_exists(conn, "journal_entry_items_accounts_receivable"):
        conn.execute(
            """
            INSERT INTO graph_edges(src_type, src_id, dst_type, dst_id, edge_type, weight)
            SELECT 'BillingDocument' AS src_type,
                   b.billingDocument AS src_id,
                   'JournalEntry' AS dst_type,
                   b.accountingDocument AS dst_id,
                   'journal_entry' AS edge_type,
                   1 AS weight
            FROM billing_document_headers b
            WHERE b.accountingDocument IS NOT NULL AND b.billingDocument IS NOT NULL;
            """
        )

    # --- Edges: BillingDocument -> Product ---
    if _table_exists(conn, "billing_document_items"):
        conn.execute(
            """
            INSERT INTO graph_edges(src_type, src_id, dst_type, dst_id, edge_type, weight)
            SELECT 'BillingDocument' AS src_type,
                   bi.billingDocument AS src_id,
                   'Product' AS dst_type,
                   bi.material AS dst_id,
                   'billed_product' AS edge_type,
                   COUNT(1) AS weight
            FROM billing_document_items bi
            WHERE bi.material IS NOT NULL AND bi.billingDocument IS NOT NULL
            GROUP BY bi.billingDocument, bi.material;
            """
        )

    # --- Edges: SalesOrder -> Product ---
    if _table_exists(conn, "sales_order_items"):
        conn.execute(
            """
            INSERT INTO graph_edges(src_type, src_id, dst_type, dst_id, edge_type, weight)
            SELECT 'SalesOrder' AS src_type,
                   soi.salesOrder AS src_id,
                   'Product' AS dst_type,
                   soi.material AS dst_id,
                   'ordered_product' AS edge_type,
                   COUNT(1) AS weight
            FROM sales_order_items soi
            WHERE soi.material IS NOT NULL AND soi.salesOrder IS NOT NULL
            GROUP BY soi.salesOrder, soi.material;
            """
        )

    # --- Edges: SalesOrder -> Plant ---
    if _table_exists(conn, "sales_order_items"):
        conn.execute(
            """
            INSERT INTO graph_edges(src_type, src_id, dst_type, dst_id, edge_type, weight)
            SELECT 'SalesOrder' AS src_type,
                   soi.salesOrder AS src_id,
                   'Plant' AS dst_type,
                   soi.productionPlant AS dst_id,
                   'fulfills_in' AS edge_type,
                   COUNT(1) AS weight
            FROM sales_order_items soi
            WHERE soi.productionPlant IS NOT NULL AND soi.salesOrder IS NOT NULL
            GROUP BY soi.salesOrder, soi.productionPlant;
            """
        )


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table_name,),
    ).fetchone()
    return row is not None


def build_db(data_dir: Path, db_path: Path, rebuild: bool) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild and db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    # Ingest tables found under data/<table>/*.jsonl
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # skip empty
        if not any(subdir.glob("*.jsonl")):
            continue
        if _table_exists(conn, subdir.name):
            # If we are not rebuilding, keep existing tables
            if not rebuild:
                continue
            conn.execute(f'DROP TABLE IF EXISTS "{subdir.name}";')
        print(f"[build_db] Ingesting {subdir.name} ...")
        _ingest_table_dir(conn, subdir)

    print("[build_db] Building graph edges + nodes ...")
    _build_edges_and_nodes(conn)

    conn.commit()
    conn.close()
    print(f"[build_db] Done: {db_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="Folder containing jsonl table directories")
    ap.add_argument("--db", default="data/sap_graph.sqlite", help="SQLite output path")
    ap.add_argument("--rebuild", action="store_true", help="Rebuild DB from scratch")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    db_path = Path(args.db).resolve()
    build_db(data_dir, db_path, rebuild=args.rebuild)


if __name__ == "__main__":
    main()

