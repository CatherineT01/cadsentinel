-- cadsentinel/db/schema_drawings.sql
-- Drawing-side tables populated by the Python ETL ingestion service.
-- Run this once against your PostgreSQL database before first ingestion.

-- ─────────────────────────────────────────────────────────────────────────── --
-- Prerequisites
-- ─────────────────────────────────────────────────────────────────────────── --

CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for text search on layer names, block names

-- ─────────────────────────────────────────────────────────────────────────── --
-- drawings
-- Top-level record per uploaded DWG file.
-- ─────────────────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS drawings (
    id                  BIGSERIAL PRIMARY KEY,
    filename            TEXT NOT NULL,
    original_path       TEXT,
    dwg_version         INT,                          -- header.version from dwg_inspect
    codepage            INT,                          -- header.codepage (1252 = CP1252)
    schema_version      TEXT NOT NULL DEFAULT '1.1.0',
    libredwg_version    JSONB NOT NULL DEFAULT '{}',
    num_entities        INT NOT NULL DEFAULT 0,
    extents_json        JSONB NOT NULL DEFAULT '{}',  -- model + paper space extents
    raw_summary_json    JSONB NOT NULL DEFAULT '{}',  -- full summary block from dwg_inspect
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata            JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_drawings_ingested_at ON drawings (ingested_at DESC);
CREATE INDEX IF NOT EXISTS idx_drawings_filename    ON drawings (filename);

-- ─────────────────────────────────────────────────────────────────────────── --
-- drawing_versions
-- One row per ingestion of a drawing. Enables deduplication via file_hash
-- and future re-ingestion tracking when a drawing is updated.
-- ─────────────────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS drawing_versions (
    id          BIGSERIAL PRIMARY KEY,
    drawing_id  BIGINT NOT NULL REFERENCES drawings (id) ON DELETE CASCADE,
    version_num INT NOT NULL DEFAULT 1,
    file_hash   TEXT NOT NULL,                        -- SHA-256 of the .dwg file
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes       TEXT,
    UNIQUE (file_hash)                                -- prevents re-ingesting same file
);

CREATE INDEX IF NOT EXISTS idx_drawing_versions_drawing_id ON drawing_versions (drawing_id);
CREATE INDEX IF NOT EXISTS idx_drawing_versions_file_hash  ON drawing_versions (file_hash);

-- ─────────────────────────────────────────────────────────────────────────── --
-- drawing_layers
-- Layer table from dwg_inspect layers[].
-- ─────────────────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS drawing_layers (
    id          BIGSERIAL PRIMARY KEY,
    drawing_id  BIGINT NOT NULL REFERENCES drawings (id) ON DELETE CASCADE,
    layer_name  TEXT NOT NULL,
    flags       INT NOT NULL DEFAULT 0,   -- frozen/locked/etc bitmask
    lineweight  INT NOT NULL DEFAULT 0,   -- in hundredths of mm; 0/255 = default/bylayer
    UNIQUE (drawing_id, layer_name)
);

CREATE INDEX IF NOT EXISTS idx_drawing_layers_drawing_id ON drawing_layers (drawing_id);
CREATE INDEX IF NOT EXISTS idx_drawing_layers_name       ON drawing_layers USING gin (layer_name gin_trgm_ops);

-- ─────────────────────────────────────────────────────────────────────────── --
-- drawing_blocks
-- Block definition summary from dwg_inspect blocks[].
-- ─────────────────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS drawing_blocks (
    id                   BIGSERIAL PRIMARY KEY,
    drawing_id           BIGINT NOT NULL REFERENCES drawings (id) ON DELETE CASCADE,
    block_name           TEXT NOT NULL,
    handle               TEXT NOT NULL,                -- hex handle string
    num_entities         INT NOT NULL DEFAULT 0,
    entity_handles_json  JSONB NOT NULL DEFAULT '[]',  -- array of child entity handles
    UNIQUE (drawing_id, handle)
);

CREATE INDEX IF NOT EXISTS idx_drawing_blocks_drawing_id ON drawing_blocks (drawing_id);
CREATE INDEX IF NOT EXISTS idx_drawing_blocks_name       ON drawing_blocks USING gin (block_name gin_trgm_ops);

-- ─────────────────────────────────────────────────────────────────────────── --
-- drawing_entities
-- One row per entity from dwg_inspect entities[].
-- Structural types (BLOCK, ENDBLK, SEQEND) are excluded by the ETL.
-- ─────────────────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS drawing_entities (
    id            BIGSERIAL PRIMARY KEY,
    drawing_id    BIGINT NOT NULL REFERENCES drawings (id) ON DELETE CASCADE,
    entity_type   TEXT NOT NULL,                       -- e.g. LINE, MTEXT, DIMENSION_LINEAR
    category      TEXT NOT NULL,                       -- curve | text | dimension | insert | image | other
    layer         TEXT,
    handle        TEXT,                                -- hex handle string
    owner_handle  TEXT,                                -- hex handle of owning block/modelspace
    dwg_index     INT,                                 -- obj->index from LibreDWG
    text_content  TEXT,                                -- for TEXT/MTEXT/ATTRIB; NULL for geometry-only
    geometry_json JSONB                                -- full geometry sub-object from dwg_inspect
);

CREATE INDEX IF NOT EXISTS idx_drawing_entities_drawing_id  ON drawing_entities (drawing_id);
CREATE INDEX IF NOT EXISTS idx_drawing_entities_category    ON drawing_entities (drawing_id, category);
CREATE INDEX IF NOT EXISTS idx_drawing_entities_type        ON drawing_entities (drawing_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_drawing_entities_layer       ON drawing_entities (drawing_id, layer);
CREATE INDEX IF NOT EXISTS idx_drawing_entities_handle      ON drawing_entities (handle);
CREATE INDEX IF NOT EXISTS idx_drawing_entities_text        ON drawing_entities USING gin (text_content gin_trgm_ops)
    WHERE text_content IS NOT NULL;

-- ─────────────────────────────────────────────────────────────────────────── --
-- drawing_dimensions
-- Denormalised dimension table for fast deterministic unit checking.
-- Populated from entities[] filtered by category='dimension'.
-- ─────────────────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS drawing_dimensions (
    id                  BIGSERIAL PRIMARY KEY,
    drawing_id          BIGINT NOT NULL REFERENCES drawings (id) ON DELETE CASCADE,
    dim_type            TEXT NOT NULL,                 -- DIMENSION_LINEAR, DIMENSION_RADIUS, etc.
    handle              TEXT,
    layer               TEXT,
    measured_value      DOUBLE PRECISION,              -- act_measurement from LibreDWG
    user_text           TEXT,                          -- override text if set by drafter
    text_position_json  JSONB,                         -- {x, y, z} of dimension text label
    geometry_json       JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_drawing_dimensions_drawing_id ON drawing_dimensions (drawing_id);
CREATE INDEX IF NOT EXISTS idx_drawing_dimensions_type       ON drawing_dimensions (drawing_id, dim_type);
CREATE INDEX IF NOT EXISTS idx_drawing_dimensions_layer      ON drawing_dimensions (drawing_id, layer);

-- ─────────────────────────────────────────────────────────────────────────── --
-- drawing_title_block
-- One row per drawing; stores the best title block candidate detected.
-- ─────────────────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS drawing_title_block (
    id                    BIGSERIAL PRIMARY KEY,
    drawing_id            BIGINT NOT NULL REFERENCES drawings (id) ON DELETE CASCADE,
    found                 BOOLEAN NOT NULL DEFAULT FALSE,
    block_name            TEXT,
    handle                TEXT,
    layer                 TEXT,
    attributes_json       JSONB NOT NULL DEFAULT '{}',   -- {TAG: value, ...} dict
    geometry_json         JSONB NOT NULL DEFAULT '{}',   -- ins_pt, scale, rotation
    candidates_json       JSONB NOT NULL DEFAULT '[]',   -- all candidates for audit
    detection_confidence  NUMERIC(5, 4) NOT NULL DEFAULT 0,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (drawing_id)
);

CREATE INDEX IF NOT EXISTS idx_drawing_title_block_drawing_id ON drawing_title_block (drawing_id);

-- ─────────────────────────────────────────────────────────────────────────── --
-- drawing_text_chunks
-- One row per text entity; embedding filled by the async embedding worker.
-- This is what pgvector semantic search queries.
-- ─────────────────────────────────────────────────────────────────────────── --

CREATE TABLE IF NOT EXISTS drawing_text_chunks (
    id             BIGSERIAL PRIMARY KEY,
    drawing_id     BIGINT NOT NULL REFERENCES drawings (id) ON DELETE CASCADE,
    entity_handle  TEXT,                               -- back-reference to drawing_entities.handle
    entity_type    TEXT,                               -- TEXT, MTEXT, ATTRIB, etc.
    layer          TEXT,
    chunk_text     TEXT NOT NULL,
    position_json  JSONB,                              -- ins_pt {x, y, z} for spatial proximity
                                                                                                                            
    embedded_at    TIMESTAMPTZ,
    embedded_model TEXT
);

CREATE INDEX IF NOT EXISTS idx_drawing_text_chunks_drawing_id ON drawing_text_chunks (drawing_id);
CREATE INDEX IF NOT EXISTS idx_drawing_text_chunks_handle     ON drawing_text_chunks (entity_handle);
                        