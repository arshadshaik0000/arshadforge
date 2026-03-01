-- Schema for Cyber Ireland 2022 Report structured data
-- SQLite-compatible (swap to PostgreSQL by changing driver in db.py)

CREATE TABLE IF NOT EXISTS report_tables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id TEXT UNIQUE NOT NULL,
    page INTEGER NOT NULL,
    section TEXT DEFAULT '',
    table_type TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Regional offices data (Table 3.2, Page 15)
CREATE TABLE IF NOT EXISTS regional_offices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    region TEXT NOT NULL,
    total_offices INTEGER DEFAULT 0,
    dedicated_offices INTEGER DEFAULT 0,
    diversified_offices INTEGER DEFAULT 0,
    per_10k_population REAL DEFAULT 0.0
);

-- Growth projections (Table 7.1, Page 27)
CREATE TABLE IF NOT EXISTS growth_projections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    gva TEXT DEFAULT '',
    employment INTEGER DEFAULT 0,
    note TEXT DEFAULT ''
);

-- Firm sizes by origin (Table 3.1, Page 14)
CREATE TABLE IF NOT EXISTS firm_sizes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    size_category TEXT NOT NULL,
    foreign_owned TEXT DEFAULT '',
    domestic TEXT DEFAULT '',
    total INTEGER DEFAULT 0
);

-- Sector summary (Table on Page 19)
CREATE TABLE IF NOT EXISTS sector_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric TEXT NOT NULL,
    category TEXT DEFAULT '',
    count INTEGER DEFAULT 0,
    percentage TEXT DEFAULT '',
    detail TEXT DEFAULT ''
);

-- GVA estimates (Table on Page 19-20)
CREATE TABLE IF NOT EXISTS gva_estimates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firm_type TEXT NOT NULL,
    avg_salary TEXT DEFAULT '',
    gva_per_employee TEXT DEFAULT '',
    employees INTEGER DEFAULT 0,
    total_gva TEXT DEFAULT ''
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_report_tables_type ON report_tables(table_type);
CREATE INDEX IF NOT EXISTS idx_growth_year ON growth_projections(year);
CREATE INDEX IF NOT EXISTS idx_regional_region ON regional_offices(region);

-- Employment breakdown for high‑level statistics and FDI support
CREATE TABLE IF NOT EXISTS employment_breakdown (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_employment INTEGER NOT NULL,
    foreign_owned_pct REAL DEFAULT 0.0,
    domestic_pct REAL DEFAULT 0.0,
    us_pct REAL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_employment_total ON employment_breakdown(total_employment);
