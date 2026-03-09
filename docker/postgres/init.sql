-- ══════════════════════════════════════════════════════════
-- init.sql — Initialisation PostgreSQL NOTAM
-- ══════════════════════════════════════════════════════════

-- Extensions utiles
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- Recherche textuelle

-- Schéma dédié
CREATE SCHEMA IF NOT EXISTS notam;

-- Commentaire sur la base
COMMENT ON DATABASE notam_db IS 'NOTAM Classification System — Production Database';

-- Index pour les recherches textuelles sur les prédictions
-- (créés après que SQLAlchemy ait créé les tables)