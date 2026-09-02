-- ============================================================================
-- Part 2 — SQL Queries for MGC Leads CRM
-- ============================================================================
-- Dialect: SQLite-compatible (also valid PostgreSQL / MySQL with minor tweaks).
-- These queries run directly via the included part2_run_queries.py script.
-- ============================================================================


-- -------------------------------------------------------------------------
-- Query 1: Conversion rate by lead source
-- Only sources with 200 or more leads, ordered best conversion rate first.
-- -------------------------------------------------------------------------

SELECT
    source,
    COUNT(*)                                          AS total_leads,
    SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END)   AS conversions,
    ROUND(
        100.0 * SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                                 AS conversion_rate_pct
FROM leads
GROUP BY source
HAVING COUNT(*) >= 200
ORDER BY conversion_rate_pct DESC;


-- -------------------------------------------------------------------------
-- Query 2: Find duplicate leads
-- Duplicates are identified by crm_record_hash appearing more than once.
-- In the raw data these are the same lead entered by different agents, often
-- visible as lead_id "MGC-XXXXX" and "MGC-XXXXX-B" with identical timestamps
-- and details.
--
-- Prevention strategy: the schema applies a UNIQUE constraint on
-- crm_record_hash. Any attempt to INSERT a lead whose hash already exists
-- would raise a constraint-violation error, forcing the CRM application to
-- either reject the duplicate or merge it with the existing record.
-- -------------------------------------------------------------------------

SELECT
    a.lead_id       AS lead_id_1,
    b.lead_id       AS lead_id_2,
    a.crm_record_hash,
    a.source,
    a.city,
    a.created_at
FROM leads a
JOIN leads b
    ON  a.crm_record_hash = b.crm_record_hash
    AND a.lead_id < b.lead_id
ORDER BY a.crm_record_hash;
