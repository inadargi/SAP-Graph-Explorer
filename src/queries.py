import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class SapQueryEngine:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.conn.row_factory = sqlite3.Row

    def top_products_by_billing_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Products = billing_document_items.material
        Product names (EN) come from product_descriptions.
        """
        sql = """
        SELECT
            bi.material AS product_id,
            MIN(COALESCE(pd.productDescription, p.productOldId, p.productType, bi.material)) AS product_name,
            COUNT(DISTINCT bi.billingDocument) AS billing_document_count
        FROM billing_document_items bi
        LEFT JOIN products p
            ON p.product = bi.material
        LEFT JOIN product_descriptions pd
            ON pd.product = bi.material AND pd.language = 'EN'
        GROUP BY bi.material
        ORDER BY billing_document_count DESC, product_id ASC
        LIMIT ?;
        """
        rows = self.conn.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def trace_billing_document_flow(self, billing_document: str) -> Dict[str, Any]:
        """
        Trace: BillingDocument -> Delivery -> SalesOrder -> Customer
        Also attach JournalEntry (accountingDocument) from billing_document_headers.
        """
        # Billing header
        header = self.conn.execute(
            """
            SELECT
                billingDocument,
                billingDocumentType,
                billingDocumentDate,
                billingDocumentIsCancelled,
                cancelledBillingDocument,
                totalNetAmount,
                transactionCurrency,
                companyCode,
                fiscalYear,
                accountingDocument,
                soldToParty
            FROM billing_document_headers
            WHERE billingDocument = ?
            LIMIT 1;
            """,
            (billing_document,),
        ).fetchone()

        if not header:
            return {"error": "billing_document_not_found", "billingDocument": billing_document}

        # Deliveries referenced by billing items
        deliveries = self.conn.execute(
            """
            SELECT DISTINCT
                bdi.referenceSdDocument AS deliveryDocument
            FROM billing_document_items bdi
            WHERE bdi.billingDocument = ?
              AND bdi.referenceSdDocument IS NOT NULL;
            """,
            (billing_document,),
        ).fetchall()
        delivery_docs = [r["deliveryDocument"] for r in deliveries]

        # Sales orders referenced by delivery items
        sales_orders = []
        if delivery_docs:
            placeholders = ",".join(["?"] * len(delivery_docs))
            # Map delivery -> sales order(s)
            so_rows = self.conn.execute(
                f"""
                SELECT DISTINCT
                    odi.deliveryDocument,
                    odi.referenceSdDocument AS salesOrder
                FROM outbound_delivery_items odi
                WHERE odi.deliveryDocument IN ({placeholders});
                """,
                tuple(delivery_docs),
            ).fetchall()

            # Group by deliveryDocument
            by_delivery: Dict[str, List[str]] = {}
            for r in so_rows:
                d = r["deliveryDocument"]
                s = r["salesOrder"]
                by_delivery.setdefault(d, [])
                if s not in by_delivery[d]:
                    by_delivery[d].append(s)

            for d, s_list in by_delivery.items():
                # Attach header metadata for each sales order in this delivery mapping.
                so_headers = []
                for so_id in s_list:
                    so_details = self.conn.execute(
                        """
                        SELECT
                            soh.salesOrder,
                            soh.salesOrderType,
                            soh.creationDate,
                            soh.overallDeliveryStatus,
                            soh.totalNetAmount,
                            soh.transactionCurrency,
                            soh.soldToParty
                        FROM sales_order_headers soh
                        WHERE soh.salesOrder = ?
                        LIMIT 1;
                        """,
                        (so_id,),
                    ).fetchone()
                    so_headers.append(dict(so_details) if so_details else {"salesOrder": so_id})

                sales_orders.append(
                    {
                        "deliveryDocument": d,
                        "salesOrders": s_list,
                        "salesOrderHeaders": so_headers,
                    }
                )

        # Journal entry / accounting document
        journal = None
        if header["accountingDocument"]:
            journal = self.conn.execute(
                """
                SELECT
                    accountingDocument,
                    companyCode,
                    fiscalYear,
                    glAccount,
                    customer
                FROM journal_entry_items_accounts_receivable
                WHERE accountingDocument = ?
                LIMIT 50;
                """,
                (header["accountingDocument"],),
            ).fetchall()
            journal = [dict(r) for r in journal]

        return {
            "billingHeader": dict(header),
            "deliveries": delivery_docs,
            "salesOrderMappings": sales_orders,
            "journalEntryItems": journal,
        }

    def broken_flows(self, limit: int = 20) -> Dict[str, Any]:
        """
        Broken flow patterns:
        1) Sales Orders that have deliveries but no billing for those deliveries.
        2) Billing Documents that reference deliveries missing in delivery items.

        Returns aggregated results with counts.
        """
        # 1) Delivered but not billed
        delivered_not_billed = self.conn.execute(
            """
            WITH deliveries AS (
                SELECT DISTINCT
                    referenceSdDocument AS salesOrder,
                    deliveryDocument
                FROM outbound_delivery_items
                WHERE referenceSdDocument IS NOT NULL AND deliveryDocument IS NOT NULL
            ),
            billed_deliveries AS (
                SELECT DISTINCT
                    referenceSdDocument AS deliveryDocument
                FROM billing_document_items
                WHERE referenceSdDocument IS NOT NULL
            )
            SELECT
                d.salesOrder,
                COUNT(DISTINCT d.deliveryDocument) AS delivered_delivery_count
            FROM deliveries d
            LEFT JOIN billed_deliveries b
                ON b.deliveryDocument = d.deliveryDocument
            WHERE b.deliveryDocument IS NULL
            GROUP BY d.salesOrder
            ORDER BY delivered_delivery_count DESC, d.salesOrder ASC
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()

        delivered_not_billed_rows = [dict(r) for r in delivered_not_billed]

        # 2) Billed without delivery
        billed_without_delivery = self.conn.execute(
            """
            WITH billed_deliveries AS (
                SELECT DISTINCT
                    referenceSdDocument AS deliveryDocument,
                    billingDocument
                FROM billing_document_items
                WHERE referenceSdDocument IS NOT NULL AND billingDocument IS NOT NULL
            ),
            delivery_exists AS (
                SELECT DISTINCT
                    deliveryDocument
                FROM outbound_delivery_items
                WHERE deliveryDocument IS NOT NULL
            )
            SELECT
                bd.billingDocument,
                bd.deliveryDocument AS missingDeliveryDocument
            FROM billed_deliveries bd
            LEFT JOIN delivery_exists de
                ON de.deliveryDocument = bd.deliveryDocument
            WHERE de.deliveryDocument IS NULL
            LIMIT ?;
            """,
            (limit,),
        ).fetchall()

        billed_without_delivery_rows = [dict(r) for r in billed_without_delivery]

        return {
            "delivered_not_billed_sales_orders": delivered_not_billed_rows,
            "billed_without_delivery_billing_documents": billed_without_delivery_rows,
        }

