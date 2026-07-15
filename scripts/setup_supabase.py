import asyncio
import re

import asyncpg

from backend.core.config import get_settings


SUPABASE_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(20) NOT NULL CHECK (entity_type IN ('supplier', 'buyer', 'lender')),
    name VARCHAR(255) NOT NULL,
    gst_number VARCHAR(15),
    pan_number VARCHAR(10),
    bank_account VARCHAR(20),
    incorporation_date DATE,
    tier INTEGER CHECK (tier IN (1, 2, 3)),
    sector VARCHAR(100),
    country VARCHAR(50) DEFAULT 'India',
    state VARCHAR(50),
    risk_score FLOAT DEFAULT 0.0,
    is_flagged BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS invoices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    invoice_number VARCHAR(100) NOT NULL,
    supplier_id UUID REFERENCES entities(id),
    buyer_id UUID REFERENCES entities(id),
    lender_id UUID REFERENCES entities(id),
    po_number VARCHAR(100),
    grn_number VARCHAR(100),
    invoice_date DATE NOT NULL,
    due_date DATE,
    amount DECIMAL(18, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'INR',
    line_items JSONB DEFAULT '[]',
    status VARCHAR(30) DEFAULT 'pending',
    sha256_fingerprint VARCHAR(64),
    minhash_signature JSONB,
    fraud_score FLOAT DEFAULT 0.0,
    fraud_decision VARCHAR(10),
    fraud_patterns JSONB DEFAULT '[]',
    match_score FLOAT,
    cascade_depth INTEGER DEFAULT 0,
    cascade_exposure DECIMAL(18, 2) DEFAULT 0.0,
    raw_file_url TEXT,
    file_type VARCHAR(10),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fraud_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    invoice_id UUID REFERENCES invoices(id),
    case_number VARCHAR(50) UNIQUE NOT NULL,
    fraud_patterns JSONB DEFAULT '[]',
    fraud_score FLOAT NOT NULL,
    decision VARCHAR(10) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    primary_signal VARCHAR(100),
    ensemble_scores JSONB DEFAULT '{}',
    shap_values JSONB DEFAULT '{}',
    cascade_path JSONB DEFAULT '[]',
    rag_context TEXT,
    alert_narrative TEXT,
    regulation_citations JSONB DEFAULT '[]',
    analyst_decision VARCHAR(20),
    analyst_notes TEXT,
    sar_draft TEXT,
    is_confirmed_fraud BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS investigations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID REFERENCES fraud_cases(id),
    analyst_id VARCHAR(100),
    messages JSONB DEFAULT '[]',
    fraud_state JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50),
    entity_id UUID,
    invoice_id UUID,
    case_id UUID,
    actor VARCHAR(100),
    payload JSONB DEFAULT '{}',
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fingerprint_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sha256_hash VARCHAR(64) UNIQUE NOT NULL,
    invoice_number VARCHAR(100),
    lender_id UUID REFERENCES entities(id),
    amount DECIMAL(18, 2),
    supplier_id UUID REFERENCES entities(id),
    buyer_id UUID REFERENCES entities(id),
    invoice_date DATE,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_invoices_supplier ON invoices(supplier_id);
CREATE INDEX IF NOT EXISTS idx_invoices_buyer ON invoices(buyer_id);
CREATE INDEX IF NOT EXISTS idx_invoices_fraud_score ON invoices(fraud_score DESC);
CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(status);
CREATE INDEX IF NOT EXISTS idx_fingerprint_hash ON fingerprint_registry(sha256_hash);
CREATE INDEX IF NOT EXISTS idx_fraud_cases_invoice ON fraud_cases(invoice_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_entity ON audit_log(entity_id);
"""

# ---------------------------------------------------------------------------
# Connection helpers — three escalating methods
# ---------------------------------------------------------------------------

def _extract_ref(database_url: str) -> str:
    m = re.search(r"db\.([a-z0-9]+)\.supabase\.co", database_url)
    return m.group(1) if m else ""


def _build_pooler_urls(database_url: str) -> list[str]:
    """
    Build IPv4-friendly Supabase transaction-mode pooler URLs.
    Tries all common AWS regions Supabase deploys in.
    """
    ref = _extract_ref(database_url)
    if not ref:
        return []
    pw_m = re.search(r"://postgres:(.+?)@db\.", database_url)
    password = pw_m.group(1) if pw_m else ""
    regions = ["ap-south-1", "us-east-1", "us-west-1", "eu-west-1", "ap-southeast-1"]
    return [
        f"postgresql://postgres.{ref}:{password}@aws-0-{region}.pooler.supabase.com:6543/postgres"
        for region in regions
    ]


async def _try_asyncpg(database_url: str) -> bool:
    """Method 1+2: Try asyncpg via pooler URLs then the direct URL."""
    urls = _build_pooler_urls(database_url) + [database_url]
    for url in urls:
        try:
            conn = await asyncpg.connect(url, ssl="require", timeout=10)
            try:
                await conn.execute(SUPABASE_SCHEMA_SQL)
                host = url.split("@")[-1].split("/")[0]
                print(f"✓ Schema applied via asyncpg ({host})")
                return True
            finally:
                await conn.close()
        except Exception:
            continue
    return False


def _try_pgmeta_https(settings) -> bool:
    """
    Method 3: Execute DDL via Supabase's pg-meta HTTPS API.
    Works when asyncpg TCP connections are blocked or IPv6-only.
    Endpoint: POST https://<ref>.supabase.co/pg/query
    """
    try:
        import httpx
    except ImportError:
        return False

    ref = _extract_ref(settings.database_url)
    if not ref:
        return False

    url = f"https://{ref}.supabase.co/pg/query"
    headers = {
        "Authorization": f"Bearer {settings.supabase_service_key}",
        "Content-Type": "application/json",
        "apikey": settings.supabase_service_key,
    }

    # Split the schema into individual statements
    statements = [
        s.strip()
        for s in SUPABASE_SCHEMA_SQL.split(";")
        if s.strip() and not s.strip().startswith("--")
    ]

    failed_count = 0
    with httpx.Client(timeout=30) as client:
        for stmt in statements:
            try:
                resp = client.post(url, headers=headers, json={"query": stmt + ";"})
                if resp.status_code not in (200, 201):
                    failed_count += 1
            except Exception:
                failed_count += 1

    if failed_count == 0:
        print("✓ Schema applied via Supabase pg-meta HTTPS API.")
        return True

    print(f"  pg-meta API: {failed_count}/{len(statements)} statements failed.")
    return False


def _print_manual_sql() -> None:
    """Method 4: Last resort — print SQL for manual execution."""
    print("\n" + "=" * 70)
    print("MANUAL SETUP REQUIRED")
    print("=" * 70)
    print("All automatic connection methods failed.")
    print("Run the following SQL in your Supabase SQL Editor:")
    print("  https://supabase.com/dashboard/project/_/sql/new\n")
    print(SUPABASE_SCHEMA_SQL)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def setup_supabase() -> None:
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required.")

    # Method 1+2: asyncpg (pooler regions + direct)
    if await _try_asyncpg(settings.database_url):
        return

    # Method 3: pg-meta HTTPS (works when TCP/IPv6 is unavailable)
    print("asyncpg TCP unavailable — trying Supabase pg-meta HTTPS API...")
    if _try_pgmeta_https(settings):
        return

    # Method 4: Manual fallback
    _print_manual_sql()


if __name__ == "__main__":
    asyncio.run(setup_supabase())
