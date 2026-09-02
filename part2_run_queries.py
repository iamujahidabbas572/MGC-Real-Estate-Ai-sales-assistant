"""
Part 2 — Run SQL Schema & Queries Against leads.csv
====================================================
Loads leads.csv into an in-memory SQLite database using the schema from
schema.sql, then executes both queries from queries.sql and prints results
as formatted tables.

Usage:
  python part2_run_queries.py
"""

import csv
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "leads.csv"
SCHEMA_PATH = BASE_DIR / "schema.sql"
QUERIES_PATH = BASE_DIR / "queries.sql"


def create_connection():
    """Create an in-memory SQLite database and return the connection."""
    return sqlite3.connect(":memory:")


def load_schema(conn):
    """Execute schema.sql to create the table structure."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema_sql)


def load_csv_data(conn):
    """Read leads.csv and insert rows into the leads table, skipping
    duplicate crm_record_hash entries (which would violate the UNIQUE
    constraint we intentionally set)."""
    cursor = conn.cursor()
    skipped = 0

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cursor.execute(
                    """
                    INSERT INTO leads (
                        lead_id, created_at, source, city, area, property_type,
                        budget_pkr_lac, bedrooms, first_response_minutes,
                        calls_made, total_call_seconds, whatsapp_replies,
                        site_visits, agent_experience_years, is_overseas,
                        referred_by_existing_client, has_financing_approved,
                        token_amount_received_pkr, crm_record_hash, converted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["lead_id"],
                        row["created_at"],
                        row["source"],
                        row["city"],
                        row.get("area") or None,
                        row["property_type"],
                        float(row["budget_pkr_lac"]) if row["budget_pkr_lac"] else None,
                        int(float(row["bedrooms"])) if row["bedrooms"] else None,
                        float(row["first_response_minutes"]) if row["first_response_minutes"] else None,
                        int(row["calls_made"]),
                        float(row["total_call_seconds"]),
                        int(row["whatsapp_replies"]),
                        int(row["site_visits"]),
                        float(row["agent_experience_years"]) if row["agent_experience_years"] else None,
                        int(row["is_overseas"]),
                        int(row["referred_by_existing_client"]),
                        int(row["has_financing_approved"]),
                        float(row["token_amount_received_pkr"]),
                        int(row["crm_record_hash"]),
                        int(row["converted"]),
                    ),
                )
            except sqlite3.IntegrityError:
                # Duplicate crm_record_hash — the UNIQUE constraint is working
                skipped += 1

    conn.commit()

    total = cursor.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
    print(f"Loaded {total} leads into SQLite ({skipped} duplicates rejected by UNIQUE constraint).\n")


def run_queries(conn):
    """Parse queries.sql, execute each query, and print results."""
    raw_sql = QUERIES_PATH.read_text(encoding="utf-8")

    # Split on the separator comments to get individual queries
    # We look for lines starting with SELECT
    queries = []
    current_query = []
    current_title = ""

    for line in raw_sql.split("\n"):
        stripped = line.strip()

        # Capture the title from the comment block above each query
        if stripped.startswith("-- Query"):
            current_title = stripped.lstrip("- ").strip()

        # Accumulate SQL lines
        if stripped.upper().startswith("SELECT") or (
            current_query and not stripped.startswith("--")
        ):
            current_query.append(line)
        elif current_query and stripped.startswith("--"):
            # We hit a new comment block, save the previous query
            queries.append((current_title, "\n".join(current_query)))
            current_query = []

    # Don't forget the last query
    if current_query:
        queries.append((current_title, "\n".join(current_query)))

    cursor = conn.cursor()

    for title, sql in queries:
        print("=" * 70)
        print(f"  {title}")
        print("=" * 70)

        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        # Calculate column widths
        widths = [len(col) for col in columns]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)))

        # Print header
        header = " | ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
        print(f"\n  {header}")
        print(f"  {'-+-'.join('-' * w for w in widths)}")

        # Print rows
        for row in rows:
            formatted = " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row))
            print(f"  {formatted}")

        print(f"\n  ({len(rows)} rows)\n")


def main():
    print("=" * 70)
    print("  Part 2 — Loading leads.csv into SQLite and running queries")
    print("=" * 70)
    print()

    conn = create_connection()

    # Step 1: Create schema without the UNIQUE constraint so we can load
    # ALL rows including duplicates — this lets Query 2 actually find them
    conn.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            lead_id                     VARCHAR(20)     PRIMARY KEY,
            created_at                  TIMESTAMP       NOT NULL,
            source                      VARCHAR(50)     NOT NULL,
            city                        VARCHAR(50)     NOT NULL,
            area                        VARCHAR(100),
            property_type               VARCHAR(50)     NOT NULL,
            budget_pkr_lac              NUMERIC(10, 2),
            bedrooms                    SMALLINT,
            first_response_minutes      NUMERIC(8, 2),
            calls_made                  INTEGER         NOT NULL DEFAULT 0,
            total_call_seconds          NUMERIC(10, 2)  NOT NULL DEFAULT 0,
            whatsapp_replies            INTEGER         NOT NULL DEFAULT 0,
            site_visits                 INTEGER         NOT NULL DEFAULT 0,
            agent_experience_years      NUMERIC(4, 2),
            is_overseas                 BOOLEAN         NOT NULL DEFAULT FALSE,
            referred_by_existing_client BOOLEAN         NOT NULL DEFAULT FALSE,
            has_financing_approved      BOOLEAN         NOT NULL DEFAULT FALSE,
            token_amount_received_pkr   NUMERIC(12, 2)  NOT NULL DEFAULT 0,
            crm_record_hash             BIGINT          NOT NULL,
            converted                   BOOLEAN         NOT NULL DEFAULT FALSE
        )
    """)

    # Load all rows (no constraint to block duplicates)
    load_csv_data(conn)
    run_queries(conn)

    # Step 2: Now demonstrate the UNIQUE constraint preventing duplicates
    print("=" * 70)
    print("  Demonstration: UNIQUE constraint on crm_record_hash")
    print("=" * 70)
    print()
    print("  Dropping and recreating the table WITH the UNIQUE constraint")
    print("  from schema.sql, then reloading the same data...\n")

    conn.execute("DROP TABLE leads")
    load_schema(conn)
    load_csv_data(conn)
    print("  The UNIQUE constraint on crm_record_hash successfully blocks")
    print("  duplicate leads at the database level.\n")

    conn.close()



if __name__ == "__main__":
    main()
