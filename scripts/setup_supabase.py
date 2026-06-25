import asyncio

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


async def setup_supabase() -> None:
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required to create Supabase tables.")

    connection = await asyncpg.connect(settings.database_url)
    try:
        await connection.execute(SUPABASE_SCHEMA_SQL)
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(setup_supabase())
