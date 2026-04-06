-- cadsentinel/db/schema_specs.sql
-- Spec-side tables populated by the spec ingestion service.
-- Run this once before first spec ingestion.
-- Run schema_drawings.sql first (spellcheck_runs references drawings).

-- ─────────────────────────────────────────────────────────────── --
-- spec_documents
-- One row per uploaded specification file.
-- ─────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS spec_documents (
    id            BIGSERIAL PRIMARY KEY,
    filename      TEXT NOT NULL,
    source_type   TEXT NOT NULL,              -- pdf, docx, txt
    title         TEXT,
    raw_text      TEXT,
    file_hash     TEXT UNIQUE,               -- SHA-256 for deduplication
    storage_path  TEXT,
    parse_status  TEXT NOT NULL DEFAULT 'pending',  -- pending, complete, failed
    organization_name TEXT,
    discipline    TEXT,
    uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata      JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_spec_documents_file_hash ON spec_documents (file_hash);
CREATE INDEX IF NOT EXISTS idx_spec_documents_uploaded  ON spec_documents (uploaded_at DESC);

-- ─────────────────────────────────────────────────────────────── --
-- spec_document_sections
-- One row per text chunk from a spec document.
-- ─────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS spec_document_sections (
    id               BIGSERIAL PRIMARY KEY,
    spec_document_id BIGINT NOT NULL REFERENCES spec_documents (id) ON DELETE CASCADE,
    chunk_index      INT NOT NULL,
    raw_section_text TEXT NOT NULL,
    char_start       INT,
    char_end         INT,
    source_page      INT,                    -- PDF page number if available
    section_hint     TEXT,                   -- detected heading if any
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (spec_document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_spec_sections_doc_id ON spec_document_sections (spec_document_id);

-- ─────────────────────────────────────────────────────────────── --
-- spec_rules
-- One row per extracted, normalized specification rule.
-- approved=FALSE until a human reviews and approves.
-- ─────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS spec_rules (
    id                       BIGSERIAL PRIMARY KEY,
    spec_document_id         BIGINT NOT NULL REFERENCES spec_documents (id) ON DELETE CASCADE,
    spec_code                TEXT,                        -- e.g. "3.2.1"
    spec_title               TEXT,
    original_spec_text       TEXT NOT NULL,
    normalized_rule_text     TEXT,
    rule_type                TEXT NOT NULL DEFAULT 'general',
    execution_mode           TEXT NOT NULL DEFAULT 'hybrid',
    severity_default         TEXT NOT NULL DEFAULT 'medium',
    entity_scope             TEXT[] NOT NULL DEFAULT '{}',
    expected_evidence_types  TEXT[] NOT NULL DEFAULT '{}',
    checker_prompt           TEXT,
    retrieval_recipe         JSONB NOT NULL DEFAULT '{}',
    structured_rule          JSONB NOT NULL DEFAULT '{}',
    extraction_confidence    NUMERIC(5,4),
    normalization_confidence NUMERIC(5,4),
    approved                 BOOLEAN NOT NULL DEFAULT FALSE,
    approved_by              TEXT,
    approved_at              TIMESTAMPTZ,
    rule_version             INT NOT NULL DEFAULT 1,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata                 JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_spec_rules_doc_id    ON spec_rules (spec_document_id);
CREATE INDEX IF NOT EXISTS idx_spec_rules_approved  ON spec_rules (spec_document_id, approved);
CREATE INDEX IF NOT EXISTS idx_spec_rules_rule_type ON spec_rules (rule_type);
CREATE INDEX IF NOT EXISTS idx_spec_rules_exec_mode ON spec_rules (execution_mode);

-- ─────────────────────────────────────────────────────────────── --
-- spec_rule_versions
-- Audit trail of every change to a spec rule.
-- ─────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS spec_rule_versions (
    id                   BIGSERIAL PRIMARY KEY,
    spec_rule_id         BIGINT NOT NULL REFERENCES spec_rules (id) ON DELETE CASCADE,
    version_num          INT NOT NULL,
    normalized_rule_text TEXT,
    execution_mode       TEXT NOT NULL,
    checker_prompt       TEXT,
    retrieval_recipe     JSONB NOT NULL DEFAULT '{}',
    structured_rule      JSONB NOT NULL DEFAULT '{}',
    change_reason        TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by           TEXT
);

CREATE INDEX IF NOT EXISTS idx_spec_rule_versions_rule_id ON spec_rule_versions (spec_rule_id);

-- ─────────────────────────────────────────────────────────────── --
-- spellcheck_runs
-- One top-level run: one drawing checked against one spec document.
-- ─────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS spellcheck_runs (
    id               BIGSERIAL PRIMARY KEY,
    drawing_id       BIGINT NOT NULL,
    spec_document_id BIGINT REFERENCES spec_documents (id) ON DELETE SET NULL,
    run_status       TEXT NOT NULL DEFAULT 'pending',
    total_specs      INT NOT NULL DEFAULT 0,
    specs_completed  INT NOT NULL DEFAULT 0,
    started_at       TIMESTAMPTZ,
    completed_at     TIMESTAMPTZ,
    triggered_by     TEXT,
    model_family     TEXT,
    notes            TEXT,
    metadata         JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_spellcheck_runs_drawing  ON spellcheck_runs (drawing_id);
CREATE INDEX IF NOT EXISTS idx_spellcheck_runs_spec_doc ON spellcheck_runs (spec_document_id);
CREATE INDEX IF NOT EXISTS idx_spellcheck_runs_status   ON spellcheck_runs (run_status);

-- ─────────────────────────────────────────────────────────────── --
-- spec_execution_runs
-- One row per spec rule checked within a spellcheck run.
-- ─────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS spec_execution_runs (
    id                BIGSERIAL PRIMARY KEY,
    spellcheck_run_id BIGINT NOT NULL REFERENCES spellcheck_runs (id) ON DELETE CASCADE,
    drawing_id        BIGINT NOT NULL,
    spec_rule_id      BIGINT NOT NULL REFERENCES spec_rules (id) ON DELETE CASCADE,
    rule_version      INT NOT NULL DEFAULT 1,
    execution_mode    TEXT NOT NULL,
    execution_status  TEXT NOT NULL DEFAULT 'pending',
    pass_fail         TEXT,           -- pass, fail, warning, needs_review
    severity          TEXT,
    confidence        NUMERIC(5,4),
    issue_count       INT,
    issue_summary     TEXT,
    detailed_result   JSONB NOT NULL DEFAULT '{}',
    retrieved_evidence JSONB NOT NULL DEFAULT '[]',
    llm_raw_response  JSONB,
    model_name        TEXT,
    token_input       INT,
    token_output      INT,
    latency_ms        INT,
    started_at        TIMESTAMPTZ,
    completed_at      TIMESTAMPTZ,
    metadata          JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_exec_runs_spellcheck_id ON spec_execution_runs (spellcheck_run_id);
CREATE INDEX IF NOT EXISTS idx_exec_runs_spec_rule_id  ON spec_execution_runs (spec_rule_id);
CREATE INDEX IF NOT EXISTS idx_exec_runs_pass_fail      ON spec_execution_runs (pass_fail);

-- ─────────────────────────────────────────────────────────────── --
-- spec_execution_issues
-- Granular issues found per spec check.
-- ─────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS spec_execution_issues (
    id                   BIGSERIAL PRIMARY KEY,
    spec_execution_run_id BIGINT NOT NULL REFERENCES spec_execution_runs (id) ON DELETE CASCADE,
    issue_code           TEXT,
    issue_type           TEXT,
    severity             TEXT,
    title                TEXT,
    description          TEXT NOT NULL,
    suggested_fix        TEXT,
    entity_ref           JSONB,
    evidence_ref         JSONB,
    confidence           NUMERIC(5,4),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_issues_exec_run_id ON spec_execution_issues (spec_execution_run_id);

-- ─────────────────────────────────────────────────────────────── --
-- spec_eval_scores
-- RAGAS + custom scores per spec execution run.
-- ─────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS spec_eval_scores (
    id                   BIGSERIAL PRIMARY KEY,
    spec_execution_run_id BIGINT NOT NULL REFERENCES spec_execution_runs (id) ON DELETE CASCADE,
    metric_name          TEXT NOT NULL,
    metric_value         NUMERIC(8,5),
    metric_label         TEXT,
    evaluator_name       TEXT,
    evaluator_version    TEXT,
    model_name           TEXT,
    rationale            TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata             JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_eval_scores_exec_run_id ON spec_eval_scores (spec_execution_run_id);

-- ─────────────────────────────────────────────────────────────── --
-- spellcheck_run_summaries
-- Denormalized rollup per completed spellcheck run.
-- ─────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS spellcheck_run_summaries (
    id                   BIGSERIAL PRIMARY KEY,
    spellcheck_run_id    BIGINT NOT NULL UNIQUE REFERENCES spellcheck_runs (id) ON DELETE CASCADE,
    pass_count           INT NOT NULL DEFAULT 0,
    fail_count           INT NOT NULL DEFAULT 0,
    warning_count        INT NOT NULL DEFAULT 0,
    review_count         INT NOT NULL DEFAULT 0,
    avg_confidence       NUMERIC(6,4),
    avg_retrieval_score  NUMERIC(6,4),
    avg_faithfulness_score NUMERIC(6,4),
    avg_correctness_score  NUMERIC(6,4),
    final_grade          TEXT,
    summary_text         TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);