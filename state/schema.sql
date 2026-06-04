-- =============================================================================
-- L.O.V.E. Relationship Support Agent — SQLite Schema
-- RSM 8430 Group 18
-- =============================================================================
-- Run via state/store.py init_db(). Tables are created IF NOT EXISTS so
-- re-running is safe.
-- =============================================================================

-- Sessions table: one row per chat session.
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Messages table: full conversation history.
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content     TEXT NOT NULL,
    intent      TEXT,                       -- router-assigned intent (nullable)
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Action state: tracks in-progress multi-turn action (e.g. build_plan slots).
-- At most one active action per session.
CREATE TABLE IF NOT EXISTS action_state (
    session_id      TEXT PRIMARY KEY,
    current_intent  TEXT NOT NULL,
    slots_json      TEXT NOT NULL DEFAULT '{}',  -- JSON dict of collected slots
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Saved plans: persisted conversation plans that users can retrieve later.
CREATE TABLE IF NOT EXISTS saved_plans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    label       TEXT NOT NULL DEFAULT 'default',
    plan_json   TEXT NOT NULL,               -- JSON dict of the full plan
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- User profile memory: one row per session, stores inferred profile as JSON.
CREATE TABLE IF NOT EXISTS user_profiles (
    session_id    TEXT PRIMARY KEY,
    profile_json  TEXT NOT NULL DEFAULT '{}',
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Index for fast message retrieval by session.
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

-- Index for fast plan lookup by session.
CREATE INDEX IF NOT EXISTS idx_plans_session ON saved_plans(session_id, created_at);
