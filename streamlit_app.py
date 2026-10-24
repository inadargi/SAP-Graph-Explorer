import json
import os
from dotenv import load_dotenv  
load_dotenv() 
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from pyvis.network import Network

from src.db import SapGraphDB
from src.llm_client import LLMClient
from src.queries import SapQueryEngine


DATA_DIR = Path("data")
DB_PATH = Path("data/sap_graph.sqlite")


NODE_TYPES = ["SalesOrder", "Delivery", "BillingDocument", "JournalEntry", "Customer", "Product", "Plant"]


def _ensure_db(rebuild: bool = False) -> None:
    if rebuild and DB_PATH.exists():
        DB_PATH.unlink()
    if not DB_PATH.exists():
        st.info("Building local SQLite DB from jsonl dataset (this can take a minute)...")
        cmd = ["python", "scripts/build_db.py", "--data-dir", str(DATA_DIR), "--db", str(DB_PATH), "--rebuild"]
        subprocess.run(cmd, check=True)
        st.success("DB built.")


def _render_pyvis(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
    net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff", font_color="#000000")
    net.barnes_hut()
    net.toggle_physics(True)

    type_color = {
        "SalesOrder": "#1f77b4",
        "Delivery": "#ff7f0e",
        "BillingDocument": "#2ca02c",
        "JournalEntry": "#d62728",
        "Customer": "#9467bd",
        "Product": "#8c564b",
        "Plant": "#e377c2",
    }

    def node_id(node_type: str, node_key: str) -> str:
        return f"{node_type}::{node_key}"

    for n in nodes:
        nid = node_id(n["node_type"], n["node_key"])
        title = f"{n['node_type']}: {n['node_key']}\n\n{json.dumps(n.get('metadata', {}), ensure_ascii=False)[:1500]}"
        net.add_node(
            nid,
            label=n.get("label") or n["node_key"],
            title=title,
            color=type_color.get(n["node_type"], "#333333"),
        )

    for e in edges:
        src = node_id(e["src_type"], e["src_id"])
        dst = node_id(e["dst_type"], e["dst_id"])
        net.add_edge(src, dst, label=str(e.get("edge_type", "")), value=int(e.get("weight", 1) or 1))

    html = net.generate_html()
    st.components.v1.html(html, height=640, scrolling=True)


def _heuristic_intent(question: str) -> Tuple[bool, str, Dict[str, Any]]:
    q = question.lower()

    dataset_keywords = [
        "sales order",
        "delivery",
        "billing",
        "journal",
        "accounting",
        "customer",
        "product",
        "material",
    ]
    if not any(k in q for k in dataset_keywords):
        return False, "unknown", {}

    if ("top" in q or "highest" in q or "most" in q) and ("product" in q or "material" in q) and ("billing" in q or "billing document" in q):
        return True, "top_products", {"limit": 10}

    if ("trace" in q or "flow" in q) and ("billing" in q) and ("sales order" in q or "delivery" in q):
        # attempt to extract an 8-digit-ish billing document id
        m = re.findall(r"\b\d{8,10}\b", question.replace(",", ""))
        if m:
            return True, "trace_billing_flow", {"billing_document": m[0]}
        return True, "trace_billing_flow", {"billing_document": None}

    if ("broken" in q or "incomplete" in q) and ("billed" in q or "delivery" in q):
        return True, "broken_flows", {"limit": 20}

    if ("delivered" in q or "delivery" in q) and "not billed" in q:
        return True, "broken_flows", {"limit": 20}

    # default: allow only if question explicitly mentions those key terms
    if ("billing" in q or "billed" in q) and ("delivery" in q or "deliver") and ("sales order" in q or "sales") and ("journal" in q or "accounting" in q):
        return True, "trace_billing_flow", {"billing_document": None}

    return False, "unknown", {}


def _extract_intent_with_llm(client: LLMClient, question: str) -> Dict[str, Any]:
    schema = {
        "allowed": "boolean",
        "intent": "top_products | trace_billing_flow | broken_flows | unknown",
        "params": {
            "limit": "number (optional)",
            "billing_document": "string (optional)",
        },
        "referenced_nodes": [
            {"node_type": "string", "node_key": "string"}
        ],
    }

    system = (
        "You are a strict intent router for an SAP dataset graph assistant. "
        "You MUST only answer questions about the provided dataset and its SAP-like entities: Sales Orders, Deliveries, Billing Documents, Journal Entries/Accounting Documents, Customers, and Products/Materials.\n"
        "If the user asks anything else, set allowed=false and intent=unknown.\n"
        "For allowed=true, choose exactly one intent from: "
        "top_products, trace_billing_flow, broken_flows.\n"
        "Extract any needed ids from the user question when possible.\n"
        "Return ONLY valid JSON."
    )

    user = f"""
Question:
{question}

Return JSON with this schema:
{json.dumps(schema, ensure_ascii=False)}
"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return client.chat_json(messages=messages, temperature=0)


def _generate_answer_with_llm(client: LLMClient, question: str, query_result: Dict[str, Any]) -> str:
    system = (
        "You are a grounded SAP dataset assistant. "
        "Generate a concise, data-backed answer to the user using ONLY the provided query_result JSON. "
        "Do not invent facts or reference fields not present in query_result. "
        "If query_result contains an error or empty result, explain what is missing."
    )

    user = f"""
User question:
{question}

query_result JSON:
{json.dumps(query_result, ensure_ascii=False)}
"""

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return client.chat_completion(messages=messages, temperature=0.2, max_tokens=600)


def _auto_referenced_nodes_for_intent(intent: str, params: Dict[str, Any], result: Dict[str, Any]) -> List[Dict[str, str]]:
    if intent == "top_products":
        items = result.get("top_products", None) or result
        # our engine returns list directly, but result passed here might be {"rows": ...}
        if isinstance(result, list):
            rows = result
        else:
            rows = items if isinstance(items, list) else []
        referenced = []
        for r in rows[:5]:
            referenced.append({"node_type": "Product", "node_key": str(r.get("product_id") or "")})
        return referenced

    if intent == "trace_billing_flow":
        nodes = []
        header = result.get("billingHeader") or {}
        if header.get("billingDocument"):
            nodes.append({"node_type": "BillingDocument", "node_key": str(header["billingDocument"])})
        for d in result.get("deliveries", [])[:10]:
            nodes.append({"node_type": "Delivery", "node_key": str(d)})
        if header.get("accountingDocument"):
            nodes.append({"node_type": "JournalEntry", "node_key": str(header["accountingDocument"])})
        for mapping in result.get("salesOrderMappings", [])[:5]:
            for so in mapping.get("salesOrders", [])[:3]:
                nodes.append({"node_type": "SalesOrder", "node_key": str(so)})
        return nodes

    if intent == "broken_flows":
        nodes = []
        for so_row in result.get("delivered_not_billed_sales_orders", [])[:10]:
            if so_row.get("salesOrder"):
                nodes.append({"node_type": "SalesOrder", "node_key": str(so_row["salesOrder"])})
        return nodes

    return []


def main() -> None:
    st.set_page_config(page_title="SAP Flow Graph + LLM Query", layout="wide", page_icon="📊")
    
    # Custom CSS for modern UI
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --accent: #06b6d4;
        --bg-light: #f8fafc;
        --text-dark: #1e293b;
        --text-light: #64748b;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Title styling */
    h1 {
        color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%) !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }
    
    /* Sidebar headers */
    .css-1d391kg {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        margin-top: 1.5rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px -2px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 8px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    [data-testid="stChatMessage"] {
        background-color: white !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* User message */
    [data-testid="stChatMessage"][data-testid="user"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
    }
    
    [data-testid="stChatMessage"][data-testid="user"] > div {
        color: white !important;
    }
    
    /* Chat input */
    [data-testid="stChatInput"] {
        background-color: white !important;
        border-radius: 12px !important;
        padding: 0.5rem !important;
        border: 2px solid #e2e8f0 !important;
    }
    
    [data-testid="stChatInput"] > div {
        background: transparent !important;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #dbeafe, #bfdbfe) !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 8px !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0) !important;
        border-left: 4px solid #22c55e !important;
        border-radius: 8px !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7, #fde68a) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2, #fecaca) !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 8px !important;
    }
    
    /* Captions */
    .stCaption {
        color: #64748b !important;
        font-style: italic !important;
    }
    
    /* Subheaders */
    h2, h3 {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* Divider */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent) !important;
        margin: 2rem 0 !important;
    }
    
    /* Metrics and cards */
    [data-testid="stMetric"] {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("SAP Flow Graph + LLM Query (Dataset Grounded)")

    _ensure_db(rebuild=False)

    client = LLMClient()
    
    # Theme selector
    st.sidebar.markdown("### 🎨 Theme")
    theme = st.sidebar.radio("Select theme", ["Light", "Dark"], index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 LLM Status")
    
    # Apply dark theme CSS if selected
    if theme == "Dark":
        st.markdown("""
        <style>
        /* Dark theme overrides */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        }
        
        h1 {
            background: linear-gradient(90deg, #818cf8, #a78bfa) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
            border-right: 1px solid #334155 !important;
        }
        
        .css-1d391kg, h2, h3 {
            color: #f1f5f9 !important;
        }
        
        [data-testid="stChatMessage"] {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            color: #f1f5f9 !important;
        }
        
        [data-testid="stChatInput"] {
            background-color: #1e293b !important;
            border: 2px solid #334155 !important;
        }
        
        .stCaption {
            color: #94a3b8 !important;
        }
        
        [data-testid="stMetric"] {
            background: #1e293b !important;
            border: 1px solid #334155 !important;
            color: #f1f5f9 !important;
        }
        
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select {
            background-color: #1e293b !important;
            color: #f1f5f9 !important;
            border-color: #334155 !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {
            border-color: #818cf8 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    if client.enabled:
        st.sidebar.success("✅ LLM enabled (LLM_API_KEY set).")
        st.sidebar.caption(f"Model: `{os.getenv('LLM_MODEL','') or 'default'}`")
    else:
        st.sidebar.warning("⚠️ LLM disabled (set `LLM_API_KEY` to enable natural-language querying).")
        st.sidebar.caption("Graph expansion still works; chat uses heuristics + templates.")

    db = SapGraphDB(str(DB_PATH))
    engine = SapQueryEngine(db.conn)
    
    # Welcome card with stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Sales Orders", f"{db.count_nodes('SalesOrder'):,}")
    with col2:
        st.metric("🚚 Deliveries", f"{db.count_nodes('Delivery'):,}")
    with col3:
        st.metric("📄 Billing Docs", f"{db.count_nodes('BillingDocument'):,}")
    with col4:
        st.metric("👥 Customers", f"{db.count_nodes('Customer'):,}")
    
    st.markdown("---")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "graph_auto_start" not in st.session_state:
        st.session_state.graph_auto_start = None

    # --- Graph panel ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🕸️ Graph Explorer")
    start_type = st.sidebar.selectbox("Start node type", NODE_TYPES, index=0, key="start_type")
    start_key = st.sidebar.text_input("Start node key (id)", key="start_key", value="", placeholder="e.g., 740507")
    depth = st.sidebar.slider("Expansion depth", 1, 4, 2)

    if st.sidebar.button("🔍 Expand neighbors", use_container_width=True):
        if not start_key.strip():
            st.sidebar.error("Please provide a node key/id.")
        else:
            nodes, edges = db.expand_neighborhood(start_type, start_key.strip(), depth=depth)
            st.subheader("🕸️ Graph Visualization")
            _render_pyvis(nodes, edges)

    # Auto expansion after chat
    if st.session_state.graph_auto_start:
        auto = st.session_state.graph_auto_start
        st.sidebar.info(f"🎯 Auto-expanding: {auto['node_type']} / {auto['node_key']}")
        nodes, edges = db.expand_neighborhood(auto["node_type"], auto["node_key"], depth=depth)
        st.subheader("🕸️ Graph Visualization (Auto-highlight)")
        _render_pyvis(nodes, edges)
        st.session_state.graph_auto_start = None

    st.markdown("---")

    # --- Chat panel ---
    st.subheader("💬 Ask questions about the SAP flow")
    st.caption("Examples: “Trace billing document 90504248”, “Which products have the highest number of billing documents?”, “Which sales orders are delivered but not billed?”")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Type your question...")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Interpreting and querying dataset..."):
                # 1) Intent routing + guardrails
                if client.enabled:
                    try:
                        intent_obj = _extract_intent_with_llm(client, user_question)
                    except Exception as e:
                        intent_obj = {"allowed": False, "intent": "unknown", "params": {}, "referenced_nodes": []}
                else:
                    allowed, intent, params = _heuristic_intent(user_question)
                    intent_obj = {"allowed": allowed, "intent": intent, "params": params, "referenced_nodes": []}

                allowed = bool(intent_obj.get("allowed"))
                intent = intent_obj.get("intent", "unknown")
                params = intent_obj.get("params") or {}

                if not allowed or intent == "unknown":
                    answer = (
                        "This system is designed to answer questions related to the provided SAP dataset only "
                        "(Sales Orders, Deliveries, Billing Documents, Journal Entries, Customers, Products/Materials)."
                    )
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    return

                # 2) Execute grounded query
                query_result: Any
                try:
                    if intent == "top_products":
                        limit = int(params.get("limit") or 10)
                        rows = engine.top_products_by_billing_documents(limit=limit)
                        query_result = {"top_products": rows}
                        referenced = [{"node_type": "Product", "node_key": str(r.get("product_id"))} for r in rows[:5]]
                    elif intent == "trace_billing_flow":
                        billing_document = params.get("billing_document")
                        if not billing_document:
                            answer = "Please specify the billing document number (e.g. “90504248”)."
                            st.markdown(answer)
                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                            return
                        query_result = engine.trace_billing_document_flow(str(billing_document))
                        referenced = _auto_referenced_nodes_for_intent(intent, params, query_result)
                    elif intent == "broken_flows":
                        limit = int(params.get("limit") or 20)
                        query_result = engine.broken_flows(limit=limit)
                        referenced = _auto_referenced_nodes_for_intent(intent, params, query_result)
                    else:
                        answer = "I couldn't map your question to a supported dataset intent."
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        return
                except Exception as e:
                    query_result = {"error": "query_failed", "details": str(e)}
                    referenced = []

                # 3) Produce final answer
                if client.enabled:
                    answer_text = _generate_answer_with_llm(client, user_question, query_result if isinstance(query_result, dict) else {"result": query_result})
                else:
                    # Deterministic fallback (no LLM)
                    if intent == "top_products":
                        rows = query_result["top_products"]
                        lines = ["Top products by number of billing documents:"]
                        for r in rows[:10]:
                            lines.append(f"- {r['product_name']} ({r['product_id']}): {r['billing_document_count']} billing documents")
                        answer_text = "\n".join(lines)
                    elif intent == "trace_billing_flow":
                        if "error" in query_result:
                            answer_text = "Billing document not found in dataset."
                        else:
                            bh = query_result["billingHeader"]
                            lines = [
                                f"Billing document `{bh['billingDocument']}` (cancelled={bh.get('billingDocumentIsCancelled')}).",
                                f"Accounting/Journals: `{bh.get('accountingDocument')}`",
                                f"Customer: `{bh.get('soldToParty')}`",
                                "",
                                "Deliveries referenced by this billing document:",
                            ]
                            for d in query_result.get("deliveries", [])[:20]:
                                lines.append(f"- Delivery `{d}`")
                            lines.append("")
                            lines.append("Sales order mappings via delivery:")
                            for m in query_result.get("salesOrderMappings", [])[:10]:
                                so_list = m.get("salesOrders", [])
                                lines.append(f"- Delivery `{m['deliveryDocument']}` -> Sales Orders: {', '.join(map(str, so_list))}")
                            answer_text = "\n".join(lines)
                    elif intent == "broken_flows":
                        df = query_result.get("delivered_not_billed_sales_orders", [])
                        bf = query_result.get("billed_without_delivery_billing_documents", [])
                        lines = [
                            "Broken/incomplete flow findings:",
                            "",
                            "1) Delivered but not billed (sales orders):",
                        ]
                        if not df:
                            lines.append("- None found (or not within limit).")
                        else:
                            for r in df[:20]:
                                lines.append(f"- SalesOrder {r['salesOrder']} has {r['delivered_delivery_count']} deliveries without billing")

                        lines.append("")
                        lines.append("2) Billed without delivery (billing documents):")
                        if not bf:
                            lines.append("- None found (or not within limit).")
                        else:
                            for r in bf[:20]:
                                lines.append(f"- BillingDocument {r['billingDocument']} references missing Delivery {r['missingDeliveryDocument']}")
                        answer_text = "\n".join(lines)
                    else:
                        answer_text = json.dumps(query_result, ensure_ascii=False)[:2000]

                st.markdown(answer_text)
                st.session_state.chat_history.append({"role": "assistant", "content": answer_text})

                # 4) Highlight graph nodes
                if referenced:
                    # Start from first referenced node to keep UI fast.
                    st.session_state.graph_auto_start = referenced[0]


if __name__ == "__main__":
    main()

