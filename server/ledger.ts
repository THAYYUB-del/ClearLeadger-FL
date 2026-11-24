/**
 * Local SHA-256 hash-linked audit ledger.
 * Note: This is a tamper-evident cryptographic hash chain stored locally in SQLite,
 * not a decentralized blockchain network with distributed consensus.
 */

import crypto from 'node:crypto';

export const GENESIS_PREV_HASH = '0'.repeat(64);

export interface AuditRecord {
  record_id: number;
  timestamp: string;
  hospital_id: string;
  training_round: number;
  event_type: string;
  payload_summary: string;
  payload_hash: string;
  previous_hash: string;
  current_hash: string;
}

export interface LedgerVerificationReport {
  valid: boolean;
  total_records: number;
  checked_records: number;
  broken_record_id: number | null;
  failure_reason: string | null;
  verified_at: string;
}

export interface TamperTestReport {
  test_executed_at: string;
  live_ledger_untouched: boolean;
  live_ledger_valid: boolean;
  target_record_id: number;
  original_summary: string;
  tampered_summary: string;
  stored_hash: string;
  recomputed_hash: string;
  verification_after_tamper: LedgerVerificationReport;
}

export function sha256Hex(input: string): string {
  return crypto.createHash('sha256').update(input).digest('hex');
}

export function computeRecordHash(fields: {
  record_id: number;
  timestamp: string;
  hospital_id: string;
  training_round: number;
  event_type: string;
  payload_summary: string;
  payload_hash: string;
  previous_hash: string;
}): string {
  const canonical = JSON.stringify({
    event_type: fields.event_type,
    hospital_id: fields.hospital_id,
    payload_hash: fields.payload_hash,
    payload_summary: fields.payload_summary,
    previous_hash: fields.previous_hash,
    record_id: fields.record_id,
    timestamp: fields.timestamp,
    training_round: fields.training_round,
  });
  return sha256Hex(canonical);
}

export function createAuditRecord(
  nextRecordId: number,
  previousHash: string,
  params: {
    hospital_id: string;
    training_round: number;
    event_type: string;
    payload_summary: string;
    payload_data: unknown;
  }
): AuditRecord {
  const timestamp = new Date().toISOString();
  const payload_hash = sha256Hex(JSON.stringify(params.payload_data));
  const current_hash = computeRecordHash({
    record_id: nextRecordId,
    timestamp,
    hospital_id: params.hospital_id,
    training_round: params.training_round,
    event_type: params.event_type,
    payload_summary: params.payload_summary,
    payload_hash,
    previous_hash: previousHash,
  });

  return {
    record_id: nextRecordId,
    timestamp,
    hospital_id: params.hospital_id,
    training_round: params.training_round,
    event_type: params.event_type,
    payload_summary: params.payload_summary,
    payload_hash,
    previous_hash: previousHash,
    current_hash,
  };
}

export function verifyLedgerChain(records: AuditRecord[]): LedgerVerificationReport {
  const verifiedAt = new Date().toISOString();
  if (records.length === 0) {
    return {
      valid: true,
      total_records: 0,
      checked_records: 0,
      broken_record_id: null,
      failure_reason: null,
      verified_at: verifiedAt,
    };
  }

  for (let i = 0; i < records.length; i++) {
    const rec = records[i];
    const expectedPrev = i === 0 ? GENESIS_PREV_HASH : records[i - 1].current_hash;

    if (rec.previous_hash !== expectedPrev) {
      return {
        valid: false,
        total_records: records.length,
        checked_records: i + 1,
        broken_record_id: rec.record_id,
        failure_reason: `Chain linkage mismatch at Record #${rec.record_id}: previous_hash does not match prior record's current_hash.`,
        verified_at: verifiedAt,
      };
    }

    const recomputed = computeRecordHash({
      record_id: rec.record_id,
      timestamp: rec.timestamp,
      hospital_id: rec.hospital_id,
      training_round: rec.training_round,
      event_type: rec.event_type,
      payload_summary: rec.payload_summary,
      payload_hash: rec.payload_hash,
      previous_hash: rec.previous_hash,
    });

    if (recomputed !== rec.current_hash) {
      return {
        valid: false,
        total_records: records.length,
        checked_records: i + 1,
        broken_record_id: rec.record_id,
        failure_reason: `Hash mismatch at Record #${rec.record_id}: stored SHA-256 digest does not match recomputed record hash (content altered).`,
        verified_at: verifiedAt,
      };
    }
  }

  return {
    valid: true,
    total_records: records.length,
    checked_records: records.length,
    broken_record_id: null,
    failure_reason: null,
    verified_at: verifiedAt,
  };
}

/**
 * Controlled non-destructive tampering demonstration:
 * Clones the current ledger in memory, alters a historical record's summary/payload,
 * and proves verifyLedgerChain catches the tampering without modifying SQLite.
 */
export function runControlledTamperTest(
  liveRecords: AuditRecord[],
  targetRecordId?: number
): TamperTestReport {
  if (liveRecords.length === 0) {
    throw new Error('Audit ledger is empty. Initialize an experiment first to create audit records.');
  }

  const cloned: AuditRecord[] = liveRecords.map((r) => ({ ...r }));
  let targetIdx =
    targetRecordId !== undefined
      ? cloned.findIndex((r) => r.record_id === targetRecordId)
      : Math.min(cloned.length - 1, Math.max(0, cloned.length - 2));

  if (targetIdx === -1) targetIdx = cloned.length - 1;

  const target = cloned[targetIdx];
  const originalSummary = target.payload_summary;
  const tamperedSummary = `${originalSummary} [ALTERED_WEIGHT_OVERRIDE=+99.4]`;
  target.payload_summary = tamperedSummary;

  const recomputedHash = computeRecordHash({
    record_id: target.record_id,
    timestamp: target.timestamp,
    hospital_id: target.hospital_id,
    training_round: target.training_round,
    event_type: target.event_type,
    payload_summary: target.payload_summary,
    payload_hash: target.payload_hash,
    previous_hash: target.previous_hash,
  });

  const verificationAfterTamper = verifyLedgerChain(cloned);
  const liveCheck = verifyLedgerChain(liveRecords);

  return {
    test_executed_at: new Date().toISOString(),
    live_ledger_untouched: true,
    live_ledger_valid: liveCheck.valid,
    target_record_id: target.record_id,
    original_summary: originalSummary,
    tampered_summary: tamperedSummary,
    stored_hash: target.current_hash,
    recomputed_hash: recomputedHash,
    verification_after_tamper: verificationAfterTamper,
  };
}
