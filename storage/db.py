"""
Storage Database Module — SQLite connection management and CRUD operations.

Provides:
- initialize_db(): Creates tables from schema.sql
- insert_table_data(record): Stores parsed table records
- execute_query(sql, params): Parameterized SQL execution
- Hardcoded data seeding for key report tables

Design: SQLite for portability. Swap to PostgreSQL by changing the connection string.
"""

import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

_DB_PATH = os.getenv("SQLITE_DB_PATH", str(PROJECT_ROOT / "data" / "reports.db"))


def get_db_path() -> str:
    return _DB_PATH


@contextmanager
def get_connection():
    """Context manager for SQLite connections."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_db() -> None:
    """Create all tables from schema.sql and seed hardcoded data."""
    os.makedirs(Path(_DB_PATH).parent, exist_ok=True)

    schema_path = Path(__file__).parent / "schema.sql"
    schema_sql = schema_path.read_text()

    with get_connection() as conn:
        conn.executescript(schema_sql)
        logger.info(f"Database initialized at {_DB_PATH}")

    # Seed hardcoded data from the PDF
    _seed_data()


def _seed_data() -> None:
    """Seed the database with hardcoded data extracted from the PDF report."""
    with get_connection() as conn:
        # Check if growth_projections already seeded.  We still want to
        # ensure employment_breakdown is populated even if the main seed ran
        # previously.
        cursor = conn.execute("SELECT COUNT(*) FROM growth_projections")
        growth_count = cursor.fetchone()[0]
        already_seeded = growth_count > 0
        if already_seeded:
            logger.info("Database already seeded; ensuring employment breakdown row exists.")
            # only add employment breakdown if missing
            cur2 = conn.execute("SELECT COUNT(*) FROM employment_breakdown")
            if cur2.fetchone()[0] == 0:
                conn.execute(
                    "INSERT INTO employment_breakdown (total_employment, foreign_owned_pct, domestic_pct, us_pct) VALUES (?, ?, ?, ?)",
                    (7351, 0.71, 0.29, 0.55),
                )
                logger.info("Seeded employment breakdown row (post‑initialization)")
            return

        # ── Growth Projections (Table 7.1, Page 27) ────────────────────
        growth_data = [
            (2021, "€1,075,523,670", 7351, "current estimate"),
            (2022, "€1,183,076,038", 8086, "projection"),
            (2023, "€1,301,383,641", 8895, ""),
            (2024, "€1,431,522,005", 9784, ""),
            (2025, "€1,574,674,206", 10763, ""),
            (2026, "€1,732,141,627", 11839, ""),
            (2027, "€1,905,355,789", 13023, ""),
            (2028, "€2,095,891,368", 14325, ""),
            (2029, "€2,305,480,505", 15758, ""),
            (2030, "€2,536,028,556", 17333, ""),
        ]
        conn.executemany(
            "INSERT INTO growth_projections (year, gva, employment, note) VALUES (?, ?, ?, ?)",
            growth_data,
        )
        logger.info(f"Seeded {len(growth_data)} growth projection rows")

        # ── Regional Offices (Table 3.2, Page 15) ──────────────────────
        regional_data = [
            ("Cork", 129, 37, 92, 7.0),
            ("Galway", 39, 8, 31, 5.0),
            ("Dublin", 397, 100, 297, 4.0),
            ("Limerick", 30, 3, 27, 3.0),
            ("Belfast", 86, 31, 55, 3.0),
            ("Ireland", 734, 191, 543, 1.5),
        ]
        conn.executemany(
            "INSERT INTO regional_offices (region, total_offices, dedicated_offices, diversified_offices, per_10k_population) VALUES (?, ?, ?, ?, ?)",
            regional_data,
        )
        logger.info(f"Seeded {len(regional_data)} regional office rows")

        # ── Firm Sizes (Table 3.1, Page 14) ─────────────────────────────
        firm_data = [
            ("Large (250 FTEs or more)", "194 (89%)", "23 (11%)", 217),
            ("Medium (50 - 249 FTEs)", "17 (29%)", "41 (71%)", 58),
            ("Small (10 - 49 FTEs)", "23 (30%)", "54 (70%)", 77),
            ("Micro (1 - 9 FTEs)", "15 (11%)", "122 (89%)", 137),
            ("Total", "249 (51%)", "240 (49%)", 489),
        ]
        conn.executemany(
            "INSERT INTO firm_sizes (size_category, foreign_owned, domestic, total) VALUES (?, ?, ?, ?)",
            firm_data,
        )
        logger.info(f"Seeded {len(firm_data)} firm size rows")

        # ── Sector Summary (Table on Page 19) ──────────────────────────
        summary_data = [
            ("Total Firms", "", 489, "", ""),
            ("Total Employment", "", 7351, "", ""),
            ("Dedicated Cyber Security Services", "Dedication", 160, "33%", "3,368 employees (46%)"),
            ("Diversified Cyber Security Services", "Dedication", 329, "67%", "3,983 employees (54%)"),
            ("Foreign-owned Firms", "Origin", 249, "51%", "5,250 employees (71%)"),
            ("Domestic Firms", "Origin", 240, "49%", "2,101 employees (29%)"),
        ]
        conn.executemany(
            "INSERT INTO sector_summary (metric, category, count, percentage, detail) VALUES (?, ?, ?, ?, ?)",
            summary_data,
        )
        logger.info(f"Seeded {len(summary_data)} sector summary rows")

        # ── GVA Estimates (Table on Page 19-20) ────────────────────────
        gva_data = [
            ("Dedicated", "€75k", "€136k", 3372, "€459m"),
            ("Diversified", "€77k", "€155k", 3983, "€617m"),
            ("Total", "", "", 7355, "€1.1bn"),
        ]
        conn.executemany(
            "INSERT INTO gva_estimates (firm_type, avg_salary, gva_per_employee, employees, total_gva) VALUES (?, ?, ?, ?, ?)",
            gva_data,
        )
        logger.info(f"Seeded {len(gva_data)} GVA estimate rows")

        # ── Employment breakdown (Section 4.3 / Key Findings) ─────────────
        # totals taken from report: 7,351 employees; 71% foreign-owned; 29% domestic; 55% US-owned
        conn.execute(
            "INSERT INTO employment_breakdown (total_employment, foreign_owned_pct, domestic_pct, us_pct) VALUES (?, ?, ?, ?)",
            (7351, 0.71, 0.29, 0.55),
        )
        logger.info("Seeded employment breakdown row")


def insert_table_data(record: dict[str, Any]) -> None:
    """Insert a parsed table record into the report_tables table."""
    with get_connection() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO report_tables (table_id, page, section, table_type, raw_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                record["table_id"],
                record["page"],
                record.get("section", ""),
                record["table_type"],
                record["raw_json"],
            ),
        )


def execute_query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """
    Execute a parameterized SQL query and return results as list of dicts.

    Security: Only SELECT statements are allowed.
    """
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed for safety.")

    with get_connection() as conn:
        cursor = conn.execute(sql, params)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]


def get_table_types() -> list[str]:
    """Return all distinct table types in the report_tables table."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT DISTINCT table_type FROM report_tables ORDER BY table_type")
        return [row[0] for row in cursor.fetchall()]


def get_all_tables() -> list[str]:
    """Return names of all user tables in the database."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]


def describe_table(table_name: str) -> list[dict[str, str]]:
    """Return column info for a given table."""
    # Validate table name to prevent SQL injection
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    with get_connection() as conn:
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        return [
            {"name": row[1], "type": row[2], "nullable": not row[3], "pk": bool(row[5])}
            for row in cursor.fetchall()
        ]
