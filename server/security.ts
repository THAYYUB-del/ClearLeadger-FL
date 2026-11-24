/**
 * Cryptographic update authentication (HMAC-SHA256) and separate
 * malicious-update validation (norm & directional/loss screening).
 *
 * Architectural principle:
 * - A valid HMAC proves possession of the hospital's secret key and payload
 *   integrity in transit; it does NOT prove the update is honest.
 * - Malicious update validation inspects parameter deltas independently of HMAC.
 */

import crypto from 'node:crypto';
import type { HospitalId } from './dataset';
import { cosineSimilarity, l2Norm, type ModelParameters } from './ml';

export interface SignedModelUpdate {
  hospital_id: HospitalId;
  claimed_sender: string;
  training_round: number;
  num_samples: number;
  parameters: ModelParameters;
  timestamp: string;
  nonce: string;
  hmac_signature: string;
}

export interface UpdateVerificationResult {
  hmac_valid: boolean;
  validation_passed: boolean;
  accepted: boolean;
  decision: 'ACCEPTED' | 'REJECTED';
  layer_failed: 'NONE' | 'AUTHENTICATION' | 'MALICIOUS_VALIDATION';
  reason: string;
  delta_norm: number;
  cosine_similarity: number | null;
  signature_preview: string;
}

const VALID_HOSPITALS: readonly HospitalId[] = [
  'hospital_a',
  'hospital_b',
  'hospital_c',
];

const ENV_KEY_MAP: Record<HospitalId, string> = {
  hospital_a: 'HOSPITAL_A_SECRET',
  hospital_b: 'HOSPITAL_B_SECRET',
  hospital_c: 'HOSPITAL_C_SECRET',
};

// Ephemeral in-memory fallback keys generated once per server process when env vars are unset
const EPHEMERAL_KEYS: Record<HospitalId, string> = {
  hospital_a: crypto.randomBytes(32).toString('hex'),
  hospital_b: crypto.randomBytes(32).toString('hex'),
  hospital_c: crypto.randomBytes(32).toString('hex'),
};

export function getHospitalSecret(hospitalId: HospitalId): string {
  const envVar = ENV_KEY_MAP[hospitalId];
  const val = process.env[envVar];
  if (val && !val.startsWith('replace-with-') && val.trim().length >= 16) {
    return val.trim();
  }
  return EPHEMERAL_KEYS[hospitalId];
}

export function canonicalizeUpdatePayload(payload: {
  hospital_id: string;
  training_round: number;
  num_samples: number;
  weights: number[];
  bias: number;
  timestamp: string;
  nonce: string;
}): string {
  return JSON.stringify({
    bias: Number(payload.bias.toFixed(8)),
    hospital_id: payload.hospital_id,
    nonce: payload.nonce,
    num_samples: payload.num_samples,
    timestamp: payload.timestamp,
    training_round: payload.training_round,
    weights: payload.weights.map((w) => Number(w.toFixed(8))),
  });
}

export function signModelUpdate(
  hospitalId: HospitalId,
  trainingRound: number,
  numSamples: number,
  parameters: ModelParameters,
  overrideSecret?: string,
  claimedSender?: string
): SignedModelUpdate {
  const timestamp = new Date().toISOString();
  const nonce = crypto.randomBytes(12).toString('hex');
  const secret = overrideSecret ?? getHospitalSecret(hospitalId);

  const canonical = canonicalizeUpdatePayload({
    hospital_id: claimedSender ?? hospitalId,
    training_round: trainingRound,
    num_samples: numSamples,
    weights: parameters.weights,
    bias: parameters.bias,
    timestamp,
    nonce,
  });

  const hmac_signature = crypto
    .createHmac('sha256', secret)
    .update(canonical)
    .digest('hex');

  return {
    hospital_id: hospitalId,
    claimed_sender: claimedSender ?? hospitalId,
    training_round: trainingRound,
    num_samples: numSamples,
    parameters,
    timestamp,
    nonce,
    hmac_signature,
  };
}

export function verifyHmacSignature(update: SignedModelUpdate): boolean {
  if (!VALID_HOSPITALS.includes(update.claimed_sender as HospitalId)) {
    return false;
  }

  const secret = getHospitalSecret(update.claimed_sender as HospitalId);
  const canonical = canonicalizeUpdatePayload({
    hospital_id: update.claimed_sender,
    training_round: update.training_round,
    num_samples: update.num_samples,
    weights: update.parameters.weights,
    bias: update.parameters.bias,
    timestamp: update.timestamp,
    nonce: update.nonce,
  });

  const expectedHex = crypto
    .createHmac('sha256', secret)
    .update(canonical)
    .digest('hex');

  const expectedBuf = Buffer.from(expectedHex, 'hex');
  const providedBuf = Buffer.from(
    /^[0-9a-fA-F]{64}$/.test(update.hmac_signature)
      ? update.hmac_signature
      : '0'.repeat(64),
    'hex'
  );

  if (expectedBuf.length !== providedBuf.length) return false;
  return crypto.timingSafeEqual(expectedBuf, providedBuf);
}

/**
 * Evaluates a signed hospital model update through BOTH independent layers:
 * 1. Cryptographic authentication (HMAC-SHA256)
 * 2. Statistical / behavioral poisoning validation (L2 norm bound & directional cosine alignment)
 */
export function inspectAndValidateUpdate(
  update: SignedModelUpdate,
  globalBefore: ModelParameters,
  referenceDelta?: number[],
  options: { maxDeltaNorm?: number; minCosineSimilarity?: number } = {}
): UpdateVerificationResult {
  const maxDeltaNorm = options.maxDeltaNorm ?? 4.5;
  const minCosineSimilarity = options.minCosineSimilarity ?? -0.15;

  const deltaVec = [
    ...update.parameters.weights.map((w, j) => w - globalBefore.weights[j]),
    update.parameters.bias - globalBefore.bias,
  ];
  const deltaNorm = Number(l2Norm(deltaVec).toFixed(4));
  const refNorm = referenceDelta ? l2Norm(referenceDelta) : 0;
  const effectiveMaxNorm =
    referenceDelta && refNorm > 1e-8
      ? Math.min(maxDeltaNorm, Math.max(0.85, refNorm * 3.5))
      : maxDeltaNorm;
  const cosSim =
    referenceDelta && refNorm > 1e-8 && deltaNorm > 1e-8
      ? Number(cosineSimilarity(deltaVec, referenceDelta).toFixed(4))
      : null;

  const signaturePreview = `${update.hmac_signature.slice(0, 16)}...`;

  // Layer 1: Cryptographic Authentication
  const hmacValid = verifyHmacSignature(update);
  if (!hmacValid) {
    return {
      hmac_valid: false,
      validation_passed: false,
      accepted: false,
      decision: 'REJECTED',
      layer_failed: 'AUTHENTICATION',
      reason:
        'HMAC-SHA256 verification failed. Signature does not match the claimed hospital secret key or payload was modified in transit.',
      delta_norm: deltaNorm,
      cosine_similarity: cosSim,
      signature_preview: signaturePreview,
    };
  }

  // Layer 2: Behavioral / Malicious Update Validation (independent of valid HMAC)
  const hasNonFinite =
    !Number.isFinite(update.parameters.bias) ||
    update.parameters.weights.some((w) => !Number.isFinite(w));

  if (hasNonFinite) {
    return {
      hmac_valid: true,
      validation_passed: false,
      accepted: false,
      decision: 'REJECTED',
      layer_failed: 'MALICIOUS_VALIDATION',
      reason:
        'Valid HMAC-SHA256 authenticated sender, but parameter vector contains NaN/Infinity values (rejected by update validator).',
      delta_norm: deltaNorm,
      cosine_similarity: cosSim,
      signature_preview: signaturePreview,
    };
  }

  if (deltaNorm > effectiveMaxNorm) {
    return {
      hmac_valid: true,
      validation_passed: false,
      accepted: false,
      decision: 'REJECTED',
      layer_failed: 'MALICIOUS_VALIDATION',
      reason: `Valid HMAC-SHA256 authenticated sender, but parameter update L2 norm (${deltaNorm.toFixed(3)}) exceeds adaptive peer bound (${effectiveMaxNorm.toFixed(2)}) — exaggerated scaling attack quarantined.`,
      delta_norm: deltaNorm,
      cosine_similarity: cosSim,
      signature_preview: signaturePreview,
    };
  }

  if (cosSim !== null && cosSim < minCosineSimilarity) {
    return {
      hmac_valid: true,
      validation_passed: false,
      accepted: false,
      decision: 'REJECTED',
      layer_failed: 'MALICIOUS_VALIDATION',
      reason: `Valid HMAC-SHA256 authenticated sender, but update direction opposes honest peer consensus (cosine similarity ${cosSim.toFixed(3)} < ${minCosineSimilarity}) — suspected label-flipping poisoning quarantined.`,
      delta_norm: deltaNorm,
      cosine_similarity: cosSim,
      signature_preview: signaturePreview,
    };
  }

  return {
    hmac_valid: true,
    validation_passed: true,
    accepted: true,
    decision: 'ACCEPTED',
    layer_failed: 'NONE',
    reason:
      'HMAC-SHA256 signature verified and update passed norm and directional integrity checks.',
    delta_norm: deltaNorm,
    cosine_similarity: cosSim,
    signature_preview: signaturePreview,
  };
}
