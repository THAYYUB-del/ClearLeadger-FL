/**
 * Federated Learning Experiment Coordinator.
 * Manages the 3 isolated hospital nodes, global FedAvg aggregation,
 * security simulation experiments, hash-linked audit ledger logging,
 * exact Linear SHAP calculations, and analytics/CSV generation.
 */

import {
  type DatasetSplit,
  type DistributionMode,
  HOSPITAL_METADATA,
  type HospitalId,
  type HospitalPartition,
  WDBC_FEATURE_NAMES,
  partitionHospitalData,
  prepareDatasetSplit,
} from './dataset';
import { ClearLedgerDatabase, type StoredRound, type StoredSecurityEvent } from './db';
import {
  GENESIS_PREV_HASH,
  createAuditRecord,
  runControlledTamperTest,
  verifyLedgerChain,
} from './ledger';
import {
  type EvaluationMetrics,
  type GlobalShapFeature,
  type IndividualPredictionExplanation,
  type ModelParameters,
  cloneParameters,
  computeGlobalShap,
  evaluateModel,
  explainPredictionShap,
  federatedAverage,
  initializeZeroModel,
  trainLocalLogisticRegression,
} from './ml';
import {
  inspectAndValidateUpdate,
  signModelUpdate,
} from './security';

export type SecurityScenario =
  | 'valid_update'
  | 'invalid_hmac'
  | 'forged_sender'
  | 'exaggerated_update'
  | 'label_flipping';

export interface HospitalNodeState {
  hospital_id: HospitalId;
  name: string;
  institution_type: string;
  region: string;
  connection_status: 'Connected' | 'Standby';
  model_status: 'Uninitialized' | 'Ready' | 'Trained Locally' | 'Synced with Global';
  training_progress: number;
  last_training_round: number;
  num_samples: number;
  benign_count: number;
  malignant_count: number;
  local_metrics: EvaluationMetrics | null;
  standalone_metrics: EvaluationMetrics | null;
  last_delta_norm: number | null;
  last_updated_at: string | null;
  privacy_guarantee: string;
}

const HOSPITAL_IDS: readonly HospitalId[] = ['hospital_a', 'hospital_b', 'hospital_c'];

function escapeCsvCell(value: string | number | null | undefined): string {
  const raw = value === null || value === undefined ? '' : String(value);
  // Prevent CSV formula injection when opened in spreadsheet applications
  const safe = /^[=+\-@\t\r]/.test(raw) ? `'${raw}` : raw;
  if (/[",\n\r]/.test(safe)) {
    return `"${safe.replace(/"/g, '""')}"`;
  }
  return safe;
}

export class FederatedCoordinator {
  public readonly db: ClearLedgerDatabase;
  private initialized = false;
  private distribution: DistributionMode = 'balanced';
  private seed = 42;
  private localEpochs = 12;
  private learningRate = 0.12;
  private currentRound = 0;

  private split: DatasetSplit | null = null;
  private partitions: Record<HospitalId, HospitalPartition> | null = null;

  private globalParams: ModelParameters | null = null;
  private globalMetrics: EvaluationMetrics | null = null;
  private centralizedParams: ModelParameters | null = null;
  private centralizedMetrics: EvaluationMetrics | null = null;

  private hospitalModels: Record<HospitalId, ModelParameters | null> = {
    hospital_a: null,
    hospital_b: null,
    hospital_c: null,
  };
  private hospitalLocalMetrics: Record<HospitalId, EvaluationMetrics | null> = {
    hospital_a: null,
    hospital_b: null,
    hospital_c: null,
  };
  private hospitalStandaloneMetrics: Record<HospitalId, EvaluationMetrics | null> = {
    hospital_a: null,
    hospital_b: null,
    hospital_c: null,
  };
  private hospitalLastRound: Record<HospitalId, number> = {
    hospital_a: 0,
    hospital_b: 0,
    hospital_c: 0,
  };
  private hospitalStatus: Record<HospitalId, HospitalNodeState['model_status']> = {
    hospital_a: 'Uninitialized',
    hospital_b: 'Uninitialized',
    hospital_c: 'Uninitialized',
  };
  private hospitalProgress: Record<HospitalId, number> = {
    hospital_a: 0,
    hospital_b: 0,
    hospital_c: 0,
  };
  private hospitalLastDeltaNorm: Record<HospitalId, number | null> = {
    hospital_a: null,
    hospital_b: null,
    hospital_c: null,
  };
  private hospitalUpdatedAt: Record<HospitalId, string | null> = {
    hospital_a: null,
    hospital_b: null,
    hospital_c: null,
  };

  constructor(dbPath?: string) {
    this.db = new ClearLedgerDatabase(dbPath);
    this.restoreFromDatabase();
  }

  /**
   * Rehydrates deterministic in-memory model weights if an experiment was previously
   * initialized and saved in SQLite across server restarts.
   */
  private restoreFromDatabase(): void {
    const meta = this.db.getExperimentMeta();
    if (!meta.initialized) return;

    this.distribution = meta.distribution === 'non_iid' ? 'non_iid' : 'balanced';
    this.seed = meta.seed;
    this.localEpochs = meta.local_epochs;
    this.learningRate = meta.learning_rate;
    this.initialized = true;

    this.split = prepareDatasetSplit(this.seed);
    this.partitions = partitionHospitalData(
      this.split.trainX,
      this.split.trainY,
      this.distribution,
      this.seed
    );
    this.globalParams = initializeZeroModel();

    // Recompute deterministic centralized & standalone models
    this.computeBaselines();

    // Replay saved number of rounds deterministically without duplicating DB records
    const savedRounds = this.db.getRounds();
    for (const r of savedRounds) {
      this.executeRoundComputationOnly(r.round_number);
    }
    this.currentRound = savedRounds.length;
    if (this.currentRound > 0 && this.globalParams && this.split) {
      this.globalMetrics = evaluateModel(
        this.globalParams,
        this.split.testX,
        this.split.testY
      );
    }
  }

  private computeBaselines(): void {
    if (!this.split || !this.partitions) return;

    // Centralized model trained on pooled training data (60 epochs comparable to 5 FL rounds)
    const centInit = initializeZeroModel();
    const centRes = trainLocalLogisticRegression(
      this.split.trainX,
      this.split.trainY,
      centInit,
      { learningRate: this.learningRate, epochs: Math.max(60, this.localEpochs * 5) }
    );
    this.centralizedParams = centRes.parameters;
    this.centralizedMetrics = evaluateModel(
      centRes.parameters,
      this.split.testX,
      this.split.testY
    );

    // Individual standalone hospital models (each trained only on its isolated local partition)
    for (const hid of HOSPITAL_IDS) {
      const part = this.partitions[hid];
      const soloRes = trainLocalLogisticRegression(
        part.trainX,
        part.trainY,
        initializeZeroModel(),
        { learningRate: this.learningRate, epochs: Math.max(60, this.localEpochs * 5) }
      );
      this.hospitalStandaloneMetrics[hid] = evaluateModel(
        soloRes.parameters,
        this.split.testX,
        this.split.testY
      );
      this.hospitalModels[hid] = cloneParameters(this.globalParams!);
      this.hospitalStatus[hid] = 'Ready';
      this.hospitalProgress[hid] = 0;
    }
  }

  private executeRoundComputationOnly(roundNumber: number): void {
    if (!this.split || !this.partitions || !this.globalParams) return;
    const baseGlobal = cloneParameters(this.globalParams);
    const updates = HOSPITAL_IDS.map((hid) => {
      const part = this.partitions![hid];
      const res = trainLocalLogisticRegression(part.trainX, part.trainY, baseGlobal, {
        learningRate: this.learningRate,
        epochs: this.localEpochs,
      });
      this.hospitalModels[hid] = res.parameters;
      this.hospitalLocalMetrics[hid] = evaluateModel(
        res.parameters,
        this.split!.testX,
        this.split!.testY
      );
      this.hospitalLastRound[hid] = roundNumber;
      this.hospitalStatus[hid] = 'Synced with Global';
      this.hospitalProgress[hid] = 100;
      this.hospitalLastDeltaNorm[hid] = Number(res.deltaNorm.toFixed(4));
      return {
        hospitalId: hid,
        numSamples: part.sampleCount,
        parameters: res.parameters,
      };
    });
    this.globalParams = federatedAverage(updates);
  }

  private appendLedgerEvent(params: {
    hospital_id: string;
    training_round: number;
    event_type: string;
    payload_summary: string;
    payload_data: unknown;
  }) {
    const existing = this.db.getAuditLedger();
    const nextId = existing.length > 0 ? existing[existing.length - 1].record_id + 1 : 1;
    const prevHash =
      existing.length > 0 ? existing[existing.length - 1].current_hash : GENESIS_PREV_HASH;
    const rec = createAuditRecord(nextId, prevHash, params);
    this.db.appendAuditRecord(rec);
    return rec;
  }

  public initializeExperiment(options?: {
    distribution?: DistributionMode;
    seed?: number;
    local_epochs?: number;
    learning_rate?: number;
  }) {
    const distribution: DistributionMode =
      options?.distribution === 'non_iid' ? 'non_iid' : 'balanced';
    const seed = Number.isFinite(options?.seed) ? Math.floor(Number(options!.seed)) : 42;
    const localEpochs = Math.min(50, Math.max(1, Math.floor(options?.local_epochs ?? 12)));
    const learningRate = Math.min(1.5, Math.max(0.005, Number(options?.learning_rate ?? 0.12)));

    this.db.resetAll();

    this.distribution = distribution;
    this.seed = seed;
    this.localEpochs = localEpochs;
    this.learningRate = learningRate;
    this.currentRound = 0;
    this.initialized = true;

    this.split = prepareDatasetSplit(seed);
    this.partitions = partitionHospitalData(
      this.split.trainX,
      this.split.trainY,
      distribution,
      seed
    );
    this.globalParams = initializeZeroModel();
    this.globalMetrics = null;

    for (const hid of HOSPITAL_IDS) {
      this.hospitalLocalMetrics[hid] = null;
      this.hospitalLastRound[hid] = 0;
      this.hospitalLastDeltaNorm[hid] = null;
      this.hospitalUpdatedAt[hid] = new Date().toISOString();
    }

    this.computeBaselines();

    this.db.saveExperimentMeta({
      initialized: true,
      distribution,
      seed,
      local_epochs: localEpochs,
      learning_rate: learningRate,
      current_round: 0,
      status: 'initialized',
    });

    this.appendLedgerEvent({
      hospital_id: 'aggregator',
      training_round: 0,
      event_type: 'EXPERIMENT_INITIALIZED',
      payload_summary: `Initialized global logistic model (30 features, ${distribution === 'balanced' ? 'Balanced IID' : 'Non-IID'} split, seed=${seed})`,
      payload_data: {
        distribution,
        seed,
        local_epochs: localEpochs,
        learning_rate: learningRate,
        train_samples: this.split.trainX.length,
        test_samples: this.split.testX.length,
        hospital_samples: {
          hospital_a: this.partitions.hospital_a.sampleCount,
          hospital_b: this.partitions.hospital_b.sampleCount,
          hospital_c: this.partitions.hospital_c.sampleCount,
        },
      },
    });

    this.db.insertActivity({
      timestamp: new Date().toISOString(),
      category: 'federated',
      title: 'Experiment Initialized',
      detail: `Partitioned ${this.split.trainX.length} training records across 3 hospitals (${distribution === 'balanced' ? 'Balanced IID' : 'Non-IID'}); ${this.split.testX.length} held-out evaluation records isolated.`,
      severity: 'info',
    });

    return this.getFederatedStatus();
  }

  public resetExperiment() {
    this.db.resetAll();
    this.initialized = false;
    this.currentRound = 0;
    this.split = null;
    this.partitions = null;
    this.globalParams = null;
    this.globalMetrics = null;
    this.centralizedParams = null;
    this.centralizedMetrics = null;

    for (const hid of HOSPITAL_IDS) {
      this.hospitalModels[hid] = null;
      this.hospitalLocalMetrics[hid] = null;
      this.hospitalStandaloneMetrics[hid] = null;
      this.hospitalLastRound[hid] = 0;
      this.hospitalStatus[hid] = 'Uninitialized';
      this.hospitalProgress[hid] = 0;
      this.hospitalLastDeltaNorm[hid] = null;
      this.hospitalUpdatedAt[hid] = null;
    }

    return {
      status: 'reset',
      message: 'Experiment state, training rounds, security events, and audit ledger cleared.',
    };
  }

  public trainSingleHospital(hospitalId: HospitalId, localEpochs?: number) {
    if (!this.initialized || !this.split || !this.partitions || !this.globalParams) {
      throw new Error('Experiment not initialized. Initialize an experiment before training.');
    }

    const part = this.partitions[hospitalId];
    if (!part) {
      throw new Error(`Unknown hospital node: ${hospitalId}`);
    }

    const epochs = Math.min(50, Math.max(1, Math.floor(localEpochs ?? this.localEpochs)));
    const startParams = this.hospitalModels[hospitalId]
      ? cloneParameters(this.hospitalModels[hospitalId]!)
      : cloneParameters(this.globalParams);

    const trainResult = trainLocalLogisticRegression(part.trainX, part.trainY, startParams, {
      learningRate: this.learningRate,
      epochs,
    });

    const evalMetrics = evaluateModel(
      trainResult.parameters,
      this.split.testX,
      this.split.testY
    );

    this.hospitalModels[hospitalId] = trainResult.parameters;
    this.hospitalLocalMetrics[hospitalId] = evalMetrics;
    this.hospitalStatus[hospitalId] = 'Trained Locally';
    this.hospitalProgress[hospitalId] = 100;
    this.hospitalLastDeltaNorm[hospitalId] = Number(trainResult.deltaNorm.toFixed(4));
    this.hospitalUpdatedAt[hospitalId] = new Date().toISOString();

    this.db.insertActivity({
      timestamp: new Date().toISOString(),
      category: 'hospital',
      title: `${part.name} Local Training Complete`,
      detail: `Trained ${epochs} local epochs on ${part.sampleCount} private records. Held-out accuracy: ${(evalMetrics.accuracy * 100).toFixed(1)}%. Raw data remained local.`,
      severity: 'success',
    });

    return {
      hospital: this.getHospitalDetail(hospitalId),
      train_loss: trainResult.trainLoss,
      train_accuracy: trainResult.trainAccuracy,
      delta_norm: Number(trainResult.deltaNorm.toFixed(4)),
      epochs_completed: epochs,
    };
  }

  public runFederatedTraining(numRounds = 1) {
    if (!this.initialized || !this.split || !this.partitions || !this.globalParams) {
      throw new Error('Experiment not initialized. Initialize an experiment before starting federated training.');
    }

    const roundsToRun = Math.min(20, Math.max(1, Math.floor(numRounds)));
    const executedRounds: StoredRound[] = [];

    for (let step = 0; step < roundsToRun; step++) {
      const nextRound = this.currentRound + 1;
      const globalBefore = cloneParameters(this.globalParams);

      // Step 1: Independent local training on each hospital's private dataset
      const localResults = HOSPITAL_IDS.map((hid) => {
        const part = this.partitions![hid];
        const res = trainLocalLogisticRegression(part.trainX, part.trainY, globalBefore, {
          learningRate: this.learningRate,
          epochs: this.localEpochs,
        });
        const evalMetrics = evaluateModel(res.parameters, this.split!.testX, this.split!.testY);
        return { hid, part, res, evalMetrics };
      });

      // Compute median/mean peer direction for directional anomaly screening
      const meanDelta = new Array<number>(globalBefore.weights.length + 1).fill(0);
      for (const item of localResults) {
        for (let j = 0; j < item.res.deltaWeights.length; j++) {
          meanDelta[j] += item.res.deltaWeights[j] / localResults.length;
        }
        meanDelta[meanDelta.length - 1] += item.res.deltaBias / localResults.length;
      }

      // Step 2: Sign each hospital update with HMAC-SHA256 & authenticate + validate
      const acceptedUpdates: Array<{
        hospitalId: HospitalId;
        numSamples: number;
        parameters: ModelParameters;
      }> = [];
      const contributions: StoredRound['hospital_contributions'] = [];
      let acceptedCount = 0;
      let rejectedCount = 0;

      const totalSamples = localResults.reduce((acc, item) => acc + item.part.sampleCount, 0);

      for (const item of localResults) {
        const signed = signModelUpdate(
          item.hid,
          nextRound,
          item.part.sampleCount,
          item.res.parameters
        );

        const verification = inspectAndValidateUpdate(signed, globalBefore, meanDelta);

        this.db.insertSecurityEvent({
          timestamp: new Date().toISOString(),
          hospital_id: item.hid,
          claimed_sender: signed.claimed_sender,
          training_round: nextRound,
          scenario: 'valid_update',
          hmac_valid: verification.hmac_valid,
          validation_passed: verification.validation_passed,
          decision: verification.decision,
          layer_failed: verification.layer_failed,
          reason: verification.reason,
          delta_norm: verification.delta_norm,
          cosine_similarity: verification.cosine_similarity,
          signature_preview: verification.signature_preview,
        });

        this.hospitalModels[item.hid] = item.res.parameters;
        this.hospitalLocalMetrics[item.hid] = item.evalMetrics;
        this.hospitalLastRound[item.hid] = nextRound;
        this.hospitalStatus[item.hid] = verification.accepted
          ? 'Synced with Global'
          : 'Trained Locally';
        this.hospitalProgress[item.hid] = 100;
        this.hospitalLastDeltaNorm[item.hid] = verification.delta_norm;
        this.hospitalUpdatedAt[item.hid] = new Date().toISOString();

        if (verification.accepted) {
          acceptedCount++;
          acceptedUpdates.push({
            hospitalId: item.hid,
            numSamples: item.part.sampleCount,
            parameters: item.res.parameters,
          });

          // Record authenticated hospital update on the hash-linked audit ledger
          this.appendLedgerEvent({
            hospital_id: item.hid,
            training_round: nextRound,
            event_type: 'AUTHENTICATED_UPDATE_ACCEPTED',
            payload_summary: `${item.part.name} update verified (HMAC-SHA256 valid, n=${item.part.sampleCount}, ||Δw||=${verification.delta_norm.toFixed(3)})`,
            payload_data: {
              hospital_id: item.hid,
              round: nextRound,
              num_samples: item.part.sampleCount,
              delta_norm: verification.delta_norm,
              signature_preview: verification.signature_preview,
            },
          });
        } else {
          rejectedCount++;
        }

        contributions.push({
          hospital_id: item.hid,
          hospital_name: item.part.name,
          num_samples: item.part.sampleCount,
          weight_share: Number((item.part.sampleCount / totalSamples).toFixed(4)),
          local_accuracy: item.evalMetrics.accuracy,
          local_loss: item.evalMetrics.loss,
          delta_norm: verification.delta_norm,
          accepted: verification.accepted,
        });
      }

      // Step 3: Sample-weighted FedAvg aggregation
      this.globalParams = federatedAverage(acceptedUpdates);
      this.globalMetrics = evaluateModel(
        this.globalParams,
        this.split.testX,
        this.split.testY
      );
      this.currentRound = nextRound;

      // Sync global model back to participating hospital nodes
      for (const hid of HOSPITAL_IDS) {
        this.hospitalModels[hid] = cloneParameters(this.globalParams);
      }

      const roundRecord: StoredRound = {
        round_number: nextRound,
        timestamp: new Date().toISOString(),
        distribution: this.distribution,
        accepted_updates: acceptedCount,
        rejected_updates: rejectedCount,
        global_accuracy: this.globalMetrics.accuracy,
        global_precision: this.globalMetrics.precision,
        global_recall: this.globalMetrics.recall,
        global_f1: this.globalMetrics.f1_score,
        global_roc_auc: this.globalMetrics.roc_auc,
        global_loss: this.globalMetrics.loss,
        confusion_matrix: this.globalMetrics.confusion_matrix,
        hospital_contributions: contributions,
      };

      this.db.insertRound(roundRecord);
      this.db.saveExperimentMeta({
        initialized: true,
        distribution: this.distribution,
        seed: this.seed,
        local_epochs: this.localEpochs,
        learning_rate: this.learningRate,
        current_round: this.currentRound,
        status: 'trained',
      });

      this.appendLedgerEvent({
        hospital_id: 'aggregator',
        training_round: nextRound,
        event_type: 'FEDAVG_AGGREGATION',
        payload_summary: `Round ${nextRound} FedAvg aggregated ${acceptedCount}/3 updates -> Global Accuracy ${(this.globalMetrics.accuracy * 100).toFixed(2)}%, ROC-AUC ${this.globalMetrics.roc_auc.toFixed(4)}`,
        payload_data: {
          round: nextRound,
          accepted_updates: acceptedCount,
          global_accuracy: this.globalMetrics.accuracy,
          global_roc_auc: this.globalMetrics.roc_auc,
          global_loss: this.globalMetrics.loss,
        },
      });

      this.db.insertActivity({
        timestamp: new Date().toISOString(),
        category: 'federated',
        title: `Round ${nextRound} FedAvg Completed`,
        detail: `Aggregated ${acceptedCount} authenticated updates. Global accuracy reached ${(this.globalMetrics.accuracy * 100).toFixed(2)}% (loss ${this.globalMetrics.loss.toFixed(4)}).`,
        severity: 'success',
      });

      executedRounds.push(roundRecord);
    }

    return {
      rounds_executed: executedRounds.length,
      current_round: this.currentRound,
      latest_round: executedRounds[executedRounds.length - 1],
      global_metrics: this.globalMetrics,
    };
  }

  /**
   * Executes controlled security & malicious-update experiments without corrupting
   * the trained global model. Explicitly separates Layer 1 (HMAC authentication)
   * from Layer 2 (malicious-update behavioral/norm validation).
   */
  public simulateSecurityExperiment(
    scenario: SecurityScenario,
    targetHospital: HospitalId = 'hospital_b'
  ): {
    event: StoredSecurityEvent;
    explanation: string;
    honest_reference_norm: number;
  } {
    if (!this.initialized || !this.split || !this.partitions || !this.globalParams) {
      throw new Error(
        'Initialize an experiment first so hospital partitions and model weights exist for security testing.'
      );
    }

    const part = this.partitions[targetHospital];
    const baseGlobal = cloneParameters(this.globalParams);

    // Honest local update from target hospital + honest reference direction from peers
    const honestLocal = trainLocalLogisticRegression(part.trainX, part.trainY, baseGlobal, {
      learningRate: this.learningRate,
      epochs: this.localEpochs,
    });

    const peerDeltas = HOSPITAL_IDS.filter((id) => id !== targetHospital).map((peerId) => {
      const peerPart = this.partitions![peerId];
      const peerRes = trainLocalLogisticRegression(peerPart.trainX, peerPart.trainY, baseGlobal, {
        learningRate: this.learningRate,
        epochs: this.localEpochs,
      });
      return [...peerRes.deltaWeights, peerRes.deltaBias];
    });

    const referenceDelta = peerDeltas[0].map(
      (_, i) => peerDeltas.reduce((acc, d) => acc + d[i], 0) / peerDeltas.length
    );

    const roundLabel = this.currentRound + 1;
    let signedUpdate = signModelUpdate(
      targetHospital,
      roundLabel,
      part.sampleCount,
      honestLocal.parameters
    );
    let explanation = '';

    switch (scenario) {
      case 'valid_update': {
        explanation = `${part.name} signed an honest local parameter update with its real HMAC-SHA256 secret key. Both cryptographic authentication and statistical validation succeeded.`;
        break;
      }
      case 'invalid_hmac': {
        // Corrupt the HMAC signature to simulate tampering in transit or invalid key
        signedUpdate = {
          ...signedUpdate,
          hmac_signature: signedUpdate.hmac_signature.replace(/^[0-9a-f]{8}/, 'deadbeef'),
        };
        explanation = `An update claiming to be from ${part.name} carried a corrupted HMAC-SHA256 digest. Layer 1 authentication rejected it before parameter inspection.`;
        break;
      }
      case 'forged_sender': {
        // Attacker signs with a wrong/unauthorized key while impersonating target hospital
        signedUpdate = signModelUpdate(
          targetHospital,
          roundLabel,
          part.sampleCount,
          honestLocal.parameters,
          'unauthorized-attacker-forged-key-0000',
          targetHospital
        );
        explanation = `An unauthorized actor attempted to impersonate ${part.name} using an invalid secret key. Layer 1 HMAC-SHA256 verification immediately rejected the forged sender.`;
        break;
      }
      case 'exaggerated_update': {
        // Insider with VALID HMAC key scales parameter delta by 18x to hijack FedAvg
        const scaledParams: ModelParameters = {
          weights: baseGlobal.weights.map(
            (w, j) => w + honestLocal.deltaWeights[j] * 18.0
          ),
          bias: baseGlobal.bias + honestLocal.deltaBias * 18.0,
        };
        signedUpdate = signModelUpdate(
          targetHospital,
          roundLabel,
          part.sampleCount,
          scaledParams
        );
        explanation = `${part.name} signed an 18× amplified model-replacement update using its genuine secret key. Layer 1 HMAC PASSED (valid key), demonstrating that HMAC alone does not guarantee honesty — Layer 2 L2-norm screening detected and quarantined the update.`;
        break;
      }
      case 'label_flipping': {
        // Compromised node flips all local patient labels (0 <-> 1) and signs honestly with real key
        const flippedY = part.trainY.map((label) => (label === 1 ? 0 : 1));
        const poisoned = trainLocalLogisticRegression(part.trainX, flippedY, baseGlobal, {
          learningRate: this.learningRate * 1.4,
          epochs: this.localEpochs + 6,
        });
        signedUpdate = signModelUpdate(
          targetHospital,
          roundLabel,
          part.sampleCount,
          poisoned.parameters
        );
        explanation = `${part.name} trained on inverted clinical labels (Malignant ↔ Benign) and signed the update with its authentic secret key. Layer 1 HMAC PASSED, but Layer 2 directional cosine-similarity validation detected the opposing gradient vector and quarantined the update.`;
        break;
      }
    }

    const verification = inspectAndValidateUpdate(signedUpdate, baseGlobal, referenceDelta);

    const storedEvent = this.db.insertSecurityEvent({
      timestamp: new Date().toISOString(),
      hospital_id: targetHospital,
      claimed_sender: signedUpdate.claimed_sender,
      training_round: roundLabel,
      scenario,
      hmac_valid: verification.hmac_valid,
      validation_passed: verification.validation_passed,
      decision: verification.decision,
      layer_failed: verification.layer_failed,
      reason: verification.reason,
      delta_norm: verification.delta_norm,
      cosine_similarity: verification.cosine_similarity,
      signature_preview: verification.signature_preview,
    });

    if (verification.accepted) {
      this.appendLedgerEvent({
        hospital_id: targetHospital,
        training_round: roundLabel,
        event_type: 'SECURITY_VERIFIED_UPDATE',
        payload_summary: `Security test (${scenario}): ${part.name} update verified and accepted`,
        payload_data: {
          scenario,
          hospital_id: targetHospital,
          delta_norm: verification.delta_norm,
          cosine_similarity: verification.cosine_similarity,
        },
      });
    }

    this.db.insertActivity({
      timestamp: new Date().toISOString(),
      category: 'security',
      title: `Security Test: ${scenario.replace(/_/g, ' ')} (${verification.decision})`,
      detail: verification.reason,
      severity: verification.accepted ? 'success' : 'alert',
    });

    return {
      event: storedEvent,
      explanation,
      honest_reference_norm: Number(honestLocal.deltaNorm.toFixed(4)),
    };
  }

  public getHospitals(): HospitalNodeState[] {
    return HOSPITAL_IDS.map((hid) => this.getHospitalDetail(hid));
  }

  public getHospitalDetail(hospitalId: HospitalId): HospitalNodeState {
    const meta = HOSPITAL_METADATA[hospitalId];
    const part = this.partitions ? this.partitions[hospitalId] : null;
    return {
      hospital_id: hospitalId,
      name: meta.name,
      institution_type: meta.institutionType,
      region: meta.region,
      connection_status: this.initialized ? 'Connected' : 'Standby',
      model_status: this.hospitalStatus[hospitalId],
      training_progress: this.hospitalProgress[hospitalId],
      last_training_round: this.hospitalLastRound[hospitalId],
      num_samples: part ? part.sampleCount : 0,
      benign_count: part ? part.benignCount : 0,
      malignant_count: part ? part.malignantCount : 0,
      local_metrics: this.hospitalLocalMetrics[hospitalId],
      standalone_metrics: this.hospitalStandaloneMetrics[hospitalId],
      last_delta_norm: this.hospitalLastDeltaNorm[hospitalId],
      last_updated_at: this.hospitalUpdatedAt[hospitalId],
      privacy_guarantee:
        'Raw patient feature vectors remain isolated inside this hospital node. Only signed parameter weights (30 coefficients + bias) are transmitted to the aggregator.',
    };
  }

  public getFederatedStatus() {
    const rounds = this.db.getRounds();
    return {
      initialized: this.initialized,
      status: !this.initialized
        ? 'Not Initialized'
        : this.currentRound === 0
          ? 'Initialized — Ready to Train'
          : 'Trained',
      distribution: this.distribution,
      seed: this.seed,
      local_epochs: this.localEpochs,
      learning_rate: this.learningRate,
      current_round: this.currentRound,
      total_train_samples: this.split ? this.split.trainX.length : 0,
      total_test_samples: this.split ? this.split.testX.length : 0,
      global_metrics: this.globalMetrics,
      hospitals: this.getHospitals(),
      rounds_count: rounds.length,
    };
  }

  public getDashboardSummary() {
    const rounds = this.db.getRounds();
    const events = this.db.getSecurityEvents(200);
    const ledger = this.db.getAuditLedger();
    const verification = verifyLedgerChain(ledger);
    const activity = this.db.getRecentActivity(12);

    const authenticatedUpdates = events.filter((e) => e.decision === 'ACCEPTED').length;
    const securityAlerts = events.filter((e) => e.decision === 'REJECTED').length;

    return {
      experiment_initialized: this.initialized,
      federated_status: !this.initialized
        ? 'Not Initialized'
        : this.currentRound === 0
          ? 'Ready for Round 1'
          : `Active (${this.currentRound} ${this.currentRound === 1 ? 'round' : 'rounds'} completed)`,
      connected_hospitals: this.initialized ? 3 : 0,
      total_hospitals: 3,
      current_round: this.currentRound,
      global_accuracy: this.globalMetrics ? this.globalMetrics.accuracy : null,
      global_roc_auc: this.globalMetrics ? this.globalMetrics.roc_auc : null,
      global_loss: this.globalMetrics ? this.globalMetrics.loss : null,
      total_authenticated_updates: authenticatedUpdates,
      security_alerts: securityAlerts,
      ledger_records_count: ledger.length,
      ledger_integrity: {
        valid: verification.valid,
        status:
          ledger.length === 0
            ? 'Empty (Awaiting Initialization)'
            : verification.valid
              ? 'Integrity Verified'
              : 'Tampering Detected',
        verified_at: verification.verified_at,
      },
      distribution: this.distribution,
      performance_series: rounds.map((r) => ({
        round: r.round_number,
        accuracy: Number((r.global_accuracy * 100).toFixed(2)),
        f1_score: Number((r.global_f1 * 100).toFixed(2)),
        roc_auc: Number((r.global_roc_auc * 100).toFixed(2)),
        loss: r.global_loss,
      })),
      recent_activity: activity,
    };
  }

  public getLedger() {
    const records = this.db.getAuditLedger();
    const verification = verifyLedgerChain(records);
    return {
      architecture_note:
        'Local SHA-256 hash-linked audit ledger persisted in SQLite. Provides tamper-evident cryptographic chaining of aggregation events without decentralized consensus.',
      records,
      verification,
    };
  }

  public verifyLedger() {
    const records = this.db.getAuditLedger();
    const report = verifyLedgerChain(records);
    this.db.insertActivity({
      timestamp: new Date().toISOString(),
      category: 'ledger',
      title: report.valid ? 'Audit Ledger Verified' : 'Audit Ledger Integrity Failure',
      detail: report.valid
        ? `All ${report.total_records} SHA-256 hash-linked records validated successfully.`
        : report.failure_reason ?? 'Hash chain broken.',
      severity: report.valid ? 'success' : 'alert',
    });
    return report;
  }

  public runTamperTest(targetRecordId?: number) {
    const records = this.db.getAuditLedger();
    const report = runControlledTamperTest(records, targetRecordId);
    this.db.insertActivity({
      timestamp: new Date().toISOString(),
      category: 'ledger',
      title: 'Controlled Tampering Test Executed',
      detail: `Simulated mutation of Record #${report.target_record_id} in isolated sandbox; SHA-256 validator detected mismatch while live ledger remained intact.`,
      severity: 'warning',
    });
    return report;
  }

  public getGlobalExplainability(): {
    available: boolean;
    method: string;
    current_round: number;
    base_value_logit: number | null;
    test_sample_count: number;
    features: GlobalShapFeature[];
    sample_preview: Array<{
      sample_index: number;
      actual_class: 'Benign' | 'Malignant';
      predicted_class: 'Benign' | 'Malignant';
      probability_benign: number;
      mean_radius: number;
      worst_perimeter: number;
      worst_concave_points: number;
    }>;
  } {
    if (!this.initialized || !this.split || !this.globalParams || this.currentRound === 0) {
      return {
        available: false,
        method: 'Exact Interventional Linear SHAP (phi_j(x) = w_j * (x_j - E[X_train_j]))',
        current_round: this.currentRound,
        base_value_logit: null,
        test_sample_count: 0,
        features: [],
        sample_preview: [],
      };
    }

    const features = computeGlobalShap(
      this.globalParams,
      this.split.testX,
      this.split.trainStandardizedMeans
    );

    let baseLogit = this.globalParams.bias;
    for (let j = 0; j < this.globalParams.weights.length; j++) {
      baseLogit += this.globalParams.weights[j] * this.split.trainStandardizedMeans[j];
    }

    const previewCount = Math.min(20, this.split.testX.length);
    const samplePreview = [];
    for (let i = 0; i < previewCount; i++) {
      const exp = explainPredictionShap(
        this.globalParams,
        i,
        this.split.testX[i],
        this.split.testRawX[i],
        this.split.testY[i],
        this.split.trainStandardizedMeans
      );
      samplePreview.push({
        sample_index: i,
        actual_class: exp.actual_class,
        predicted_class: exp.predicted_class,
        probability_benign: exp.probability_benign,
        mean_radius: Number(this.split.testRawX[i][0].toFixed(2)),
        worst_perimeter: Number(this.split.testRawX[i][22].toFixed(2)),
        worst_concave_points: Number(this.split.testRawX[i][27].toFixed(4)),
      });
    }

    return {
      available: true,
      method: 'Exact Interventional Linear SHAP (phi_j(x) = w_j * (x_j - E[X_train_j]))',
      current_round: this.currentRound,
      base_value_logit: Number(baseLogit.toFixed(5)),
      test_sample_count: this.split.testX.length,
      features,
      sample_preview: samplePreview,
    };
  }

  public explainPrediction(sampleIndex: number): IndividualPredictionExplanation {
    if (!this.initialized || !this.split || !this.globalParams || this.currentRound === 0) {
      throw new Error(
        'Global model has not been trained yet. Run at least one federated round before generating SHAP explanations.'
      );
    }

    const idx = Math.floor(sampleIndex);
    if (!Number.isFinite(idx) || idx < 0 || idx >= this.split.testX.length) {
      throw new Error(
        `Sample index must be between 0 and ${this.split.testX.length - 1}.`
      );
    }

    return explainPredictionShap(
      this.globalParams,
      idx,
      this.split.testX[idx],
      this.split.testRawX[idx],
      this.split.testY[idx],
      this.split.trainStandardizedMeans
    );
  }

  public getAnalytics() {
    const rounds = this.db.getRounds();
    const evaluated = this.initialized && this.currentRound > 0 && this.globalMetrics !== null;

    const comparison = evaluated
      ? [
          {
            model_name: 'Centralized Baseline (Pooled)',
            scope: 'All training data pooled (privacy-breaking upper bound)',
            metrics: this.centralizedMetrics,
          },
          {
            model_name: 'Federated Global Model (FedAvg)',
            scope: `Aggregated across 3 hospitals over ${this.currentRound} ${this.currentRound === 1 ? 'round' : 'rounds'}`,
            metrics: this.globalMetrics,
          },
          ...HOSPITAL_IDS.map((hid) => ({
            model_name: `${HOSPITAL_METADATA[hid].name} (Standalone Local)`,
            scope: `Trained only on ${this.partitions?.[hid].sampleCount ?? 0} isolated local records`,
            metrics: this.hospitalStandaloneMetrics[hid],
          })),
        ]
      : [];

    return {
      evaluated,
      distribution: this.distribution,
      current_round: this.currentRound,
      dataset_info: {
        name: 'Wisconsin Breast Cancer Diagnostic (WDBC)',
        total_samples: 569,
        feature_count: WDBC_FEATURE_NAMES.length,
        train_samples: this.split ? this.split.trainX.length : 0,
        held_out_test_samples: this.split ? this.split.testX.length : 0,
        educational_notice:
          'Educational machine learning experiment only — not a clinical diagnostic system.',
      },
      global_metrics: this.globalMetrics,
      centralized_metrics: evaluated ? this.centralizedMetrics : null,
      comparison,
      rounds,
    };
  }

  public exportAnalyticsCsv(): string {
    const analytics = this.getAnalytics();
    if (!analytics.evaluated) {
      throw new Error('Cannot export CSV before running at least one federated training round.');
    }

    const lines: string[] = [];
    lines.push(
      [
        'section',
        'model_or_round',
        'distribution',
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'roc_auc',
        'loss',
        'tp',
        'tn',
        'fp',
        'fn',
      ].join(',')
    );

    for (const row of analytics.comparison) {
      if (!row.metrics) continue;
      lines.push(
        [
          escapeCsvCell('model_comparison'),
          escapeCsvCell(row.model_name),
          escapeCsvCell(analytics.distribution),
          escapeCsvCell(row.metrics.accuracy),
          escapeCsvCell(row.metrics.precision),
          escapeCsvCell(row.metrics.recall),
          escapeCsvCell(row.metrics.f1_score),
          escapeCsvCell(row.metrics.roc_auc),
          escapeCsvCell(row.metrics.loss),
          escapeCsvCell(row.metrics.confusion_matrix.tp),
          escapeCsvCell(row.metrics.confusion_matrix.tn),
          escapeCsvCell(row.metrics.confusion_matrix.fp),
          escapeCsvCell(row.metrics.confusion_matrix.fn),
        ].join(',')
      );
    }

    for (const r of analytics.rounds) {
      lines.push(
        [
          escapeCsvCell('federated_round'),
          escapeCsvCell(`Round ${r.round_number}`),
          escapeCsvCell(r.distribution),
          escapeCsvCell(r.global_accuracy),
          escapeCsvCell(r.global_precision),
          escapeCsvCell(r.global_recall),
          escapeCsvCell(r.global_f1),
          escapeCsvCell(r.global_roc_auc),
          escapeCsvCell(r.global_loss),
          escapeCsvCell(r.confusion_matrix.tp),
          escapeCsvCell(r.confusion_matrix.tn),
          escapeCsvCell(r.confusion_matrix.fp),
          escapeCsvCell(r.confusion_matrix.fn),
        ].join(',')
      );
    }

    return lines.join('\n') + '\n';
  }
}
