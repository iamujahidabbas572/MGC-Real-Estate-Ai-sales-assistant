-- ============================================================================
-- Part 2 — SQL Schema for MGC Leads CRM
-- ============================================================================
-- Design rationale:
--   A single 'leads' table is used because the CSV represents a flat CRM dump
--   with no clear one-to-many relationships that would benefit from splitting.
--   In a production system, you might normalize 'source' and 'city' into lookup
--   tables, but for ~9,000 rows the simplicity of one table outweighs the
--   marginal storage savings. The key constraint is the UNIQUE on
--   crm_record_hash, which prevents the same lead from being entered twice by
--   different agents — the exact duplicate problem visible in the raw data.
-- ============================================================================

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
    crm_record_hash             BIGINT          NOT NULL UNIQUE,
    converted                   BOOLEAN         NOT NULL DEFAULT FALSE
);

-- The UNIQUE constraint on crm_record_hash is the duplicate prevention
-- mechanism. When two agents try to enter the same lead, the second INSERT
-- will fail with a unique-constraint violation, forcing the CRM to route
-- them to the existing record instead of creating a duplicate.
