/**
 * SQLite persistence layer using Node.js built-in node:sqlite (DatabaseSync).
 * Stores experiments, training rounds, security events, and the SHA-256 hash-linked audit ledger.
 * Every SQL statement uses parameterized queries strictly.
 */

import fs from 'node:fs';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import type { AuditRecord } from './ledger';

export interface StoredRound {
  round_number: number;
  timestamp: string;
  distribution: string;
  accepted_updates: number;
  rejected_updates: number;
  global_accuracy: number;
  global_precision: number;
  global_recall: number;
  global_f1: number;
  global_roc_auc: number;
  global_loss: number;
  confusion_matrix: { tp: number; tn: number; fp: number; fn: number };
  hospital_contributions: Array<{
    hospital_id: string;
    hospital_name: string;
    num_samples: number;
    weight_share: number;
    local_accuracy: number;
    local_loss: number;
    delta_norm: number;
    accepted: boolean;
  }>;
}

export interface StoredSecurityEvent {
  id: number;
  timestamp: string;
  hospital_id: string;
  claimed_sender: string;
  training_round: number;
  scenario: string;
  hmac_valid: boolean;
  validation_passed: boolean;
  decision: 'ACCEPTED' | 'REJECTED';
  layer_failed: string;
  reason: string;
  delta_norm: number;
  cosine_similarity: number | null;
  signature_preview: string;
}

export interface StoredActivity {
  id: number;
  timestamp: string;
  category: 'federated' | 'hospital' | 'security' | 'ledger' | 'system';
  title: string;
  detail: string;
  severity: 'info' | 'success' | 'warning' | 'alert';
}

export class ClearLedgerDatabase {
  private db: DatabaseSync;
  public readonly dbPath: string;

  constructor(customPath?: string) {
    const baseDir = path.resolve(process.cwd(), 'results');
    const resolvedPath =
      customPath === ':memory:'
        ? ':memory:'
        : customPath
          ? path.resolve(customPath)
          : path.resolve(baseDir, 'clearledger.db');

    if (resolvedPath !== ':memory:') {
      const parentDir = path.dirname(resolvedPath);
      fs.mkdirSync(parentDir, { recursive: true });
    }

    this.dbPath = resolvedPath;
    this.db = new DatabaseSync(resolvedPath);
    this.initializeSchema();
  }

  private initializeSchema(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS experiment_meta (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        initialized INTEGER NOT NULL DEFAULT 0,
        distribution TEXT NOT NULL DEFAULT 'balanced',
        seed INTEGER NOT NULL DEFAULT 42,
        local_epochs INTEGER NOT NULL DEFAULT 12,
        learning_rate REAL NOT NULL DEFAULT 0.12,
        current_round INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'not_initialized',
        updated_at TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS training_rounds (
        round_number INTEGER PRIMARY KEY,
        timestamp TEXT NOT NULL,
        distribution TEXT NOT NULL,
        accepted_updates INTEGER NOT NULL,
        rejected_updates INTEGER NOT NULL,
        global_accuracy REAL NOT NULL,
        global_precision REAL NOT NULL,
        global_recall REAL NOT NULL,
        global_f1 REAL NOT NULL,
        global_roc_auc REAL NOT NULL,
        global_loss REAL NOT NULL,
        confusion_matrix_json TEXT NOT NULL,
        hospital_contributions_json TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS security_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        hospital_id TEXT NOT NULL,
        claimed_sender TEXT NOT NULL,
        training_round INTEGER NOT NULL,
        scenario TEXT NOT NULL,
        hmac_valid INTEGER NOT NULL,
        validation_passed INTEGER NOT NULL,
        decision TEXT NOT NULL,
        layer_failed TEXT NOT NULL,
        reason TEXT NOT NULL,
        delta_norm REAL NOT NULL,
        cosine_similarity REAL,
        signature_preview TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS audit_ledger (
        record_id INTEGER PRIMARY KEY,
        timestamp TEXT NOT NULL,
        hospital_id TEXT NOT NULL,
        training_round INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        payload_summary TEXT NOT NULL,
        payload_hash TEXT NOT NULL,
        previous_hash TEXT NOT NULL,
        current_hash TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS system_activity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        category TEXT NOT NULL,
        title TEXT NOT NULL,
        detail TEXT NOT NULL,
        severity TEXT NOT NULL
      );
    `);

    const row = this.db
      .prepare('SELECT id FROM experiment_meta WHERE id = 1')
      .get();
    if (!row) {
      this.db
        .prepare(
          `INSERT INTO experiment_meta (id, initialized, distribution, seed, local_epochs, learning_rate, current_round, status, updated_at)
           VALUES (1, 0, 'balanced', 42, 12, 0.12, 0, 'not_initialized', ?)`
        )
        .run(new Date().toISOString());
    }
  }

  saveExperimentMeta(meta: {
    initialized: boolean;
    distribution: string;
    seed: number;
    local_epochs: number;
    learning_rate: number;
    current_round: number;
    status: string;
  }): void {
    this.db
      .prepare(
        `UPDATE experiment_meta
         SET initialized = ?, distribution = ?, seed = ?, local_epochs = ?, learning_rate = ?, current_round = ?, status = ?, updated_at = ?
         WHERE id = 1`
      )
      .run(
        meta.initialized ? 1 : 0,
        meta.distribution,
        meta.seed,
        meta.local_epochs,
        meta.learning_rate,
        meta.current_round,
        meta.status,
        new Date().toISOString()
      );
  }

  getExperimentMeta(): {
    initialized: boolean;
    distribution: string;
    seed: number;
    local_epochs: number;
    learning_rate: number;
    current_round: number;
    status: string;
    updated_at: string;
  } {
    const row = this.db
      .prepare('SELECT * FROM experiment_meta WHERE id = 1')
      .get() as Record<string, unknown>;
    return {
      initialized: Number(row.initialized) === 1,
      distribution: String(row.distribution),
      seed: Number(row.seed),
      local_epochs: Number(row.local_epochs),
      learning_rate: Number(row.learning_rate),
      current_round: Number(row.current_round),
      status: String(row.status),
      updated_at: String(row.updated_at),
    };
  }

  insertRound(round: StoredRound): void {
    this.db
      .prepare(
        `INSERT OR REPLACE INTO training_rounds (
          round_number, timestamp, distribution, accepted_updates, rejected_updates,
          global_accuracy, global_precision, global_recall, global_f1, global_roc_auc, global_loss,
          confusion_matrix_json, hospital_contributions_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        round.round_number,
        round.timestamp,
        round.distribution,
        round.accepted_updates,
        round.rejected_updates,
        round.global_accuracy,
        round.global_precision,
        round.global_recall,
        round.global_f1,
        round.global_roc_auc,
        round.global_loss,
        JSON.stringify(round.confusion_matrix),
        JSON.stringify(round.hospital_contributions)
      );
  }

  getRounds(): StoredRound[] {
    const rows = this.db
      .prepare('SELECT * FROM training_rounds ORDER BY round_number ASC')
      .all() as Array<Record<string, unknown>>;

    return rows.map((r) => ({
      round_number: Number(r.round_number),
      timestamp: String(r.timestamp),
      distribution: String(r.distribution),
      accepted_updates: Number(r.accepted_updates),
      rejected_updates: Number(r.rejected_updates),
      global_accuracy: Number(r.global_accuracy),
      global_precision: Number(r.global_precision),
      global_recall: Number(r.global_recall),
      global_f1: Number(r.global_f1),
      global_roc_auc: Number(r.global_roc_auc),
      global_loss: Number(r.global_loss),
      confusion_matrix: JSON.parse(String(r.confusion_matrix_json)),
      hospital_contributions: JSON.parse(String(r.hospital_contributions_json)),
    }));
  }

  insertSecurityEvent(event: Omit<StoredSecurityEvent, 'id'>): StoredSecurityEvent {
    const result = this.db
      .prepare(
        `INSERT INTO security_events (
          timestamp, hospital_id, claimed_sender, training_round, scenario,
          hmac_valid, validation_passed, decision, layer_failed, reason,
          delta_norm, cosine_similarity, signature_preview
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        event.timestamp,
        event.hospital_id,
        event.claimed_sender,
        event.training_round,
        event.scenario,
        event.hmac_valid ? 1 : 0,
        event.validation_passed ? 1 : 0,
        event.decision,
        event.layer_failed,
        event.reason,
        event.delta_norm,
        event.cosine_similarity,
        event.signature_preview
      );

    return {
      ...event,
      id: Number(result.lastInsertRowid),
    };
  }

  getSecurityEvents(limit = 100): StoredSecurityEvent[] {
    const rows = this.db
      .prepare('SELECT * FROM security_events ORDER BY id DESC LIMIT ?')
      .all(limit) as Array<Record<string, unknown>>;

    return rows.map((r) => ({
      id: Number(r.id),
      timestamp: String(r.timestamp),
      hospital_id: String(r.hospital_id),
      claimed_sender: String(r.claimed_sender),
      training_round: Number(r.training_round),
      scenario: String(r.scenario),
      hmac_valid: Number(r.hmac_valid) === 1,
      validation_passed: Number(r.validation_passed) === 1,
      decision: String(r.decision) as 'ACCEPTED' | 'REJECTED',
      layer_failed: String(r.layer_failed),
      reason: String(r.reason),
      delta_norm: Number(r.delta_norm),
      cosine_similarity:
        r.cosine_similarity === null || r.cosine_similarity === undefined
          ? null
          : Number(r.cosine_similarity),
      signature_preview: String(r.signature_preview),
    }));
  }

  appendAuditRecord(record: AuditRecord): void {
    this.db
      .prepare(
        `INSERT INTO audit_ledger (
          record_id, timestamp, hospital_id, training_round, event_type,
          payload_summary, payload_hash, previous_hash, current_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        record.record_id,
        record.timestamp,
        record.hospital_id,
        record.training_round,
        record.event_type,
        record.payload_summary,
        record.payload_hash,
        record.previous_hash,
        record.current_hash
      );
  }

  getAuditLedger(): AuditRecord[] {
    const rows = this.db
      .prepare('SELECT * FROM audit_ledger ORDER BY record_id ASC')
      .all() as Array<Record<string, unknown>>;

    return rows.map((r) => ({
      record_id: Number(r.record_id),
      timestamp: String(r.timestamp),
      hospital_id: String(r.hospital_id),
      training_round: Number(r.training_round),
      event_type: String(r.event_type),
      payload_summary: String(r.payload_summary),
      payload_hash: String(r.payload_hash),
      previous_hash: String(r.previous_hash),
      current_hash: String(r.current_hash),
    }));
  }

  insertActivity(entry: Omit<StoredActivity, 'id'>): void {
    this.db
      .prepare(
        `INSERT INTO system_activity (timestamp, category, title, detail, severity)
         VALUES (?, ?, ?, ?, ?)`
      )
      .run(entry.timestamp, entry.category, entry.title, entry.detail, entry.severity);
  }

  getRecentActivity(limit = 20): StoredActivity[] {
    const rows = this.db
      .prepare('SELECT * FROM system_activity ORDER BY id DESC LIMIT ?')
      .all(limit) as Array<Record<string, unknown>>;

    return rows.map((r) => ({
      id: Number(r.id),
      timestamp: String(r.timestamp),
      category: String(r.category) as StoredActivity['category'],
      title: String(r.title),
      detail: String(r.detail),
      severity: String(r.severity) as StoredActivity['severity'],
    }));
  }

  resetAll(): void {
    this.db.exec(`
      DELETE FROM training_rounds;
      DELETE FROM security_events;
      DELETE FROM audit_ledger;
      DELETE FROM system_activity;
    `);
    this.saveExperimentMeta({
      initialized: false,
      distribution: 'balanced',
      seed: 42,
      local_epochs: 12,
      learning_rate: 0.12,
      current_round: 0,
      status: 'not_initialized',
    });
  }

  close(): void {
    this.db.close();
  }
}
