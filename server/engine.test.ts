import test from 'node:test';
import assert from 'node:assert/strict';
import {
  partitionHospitalData,
  prepareDatasetSplit,
  WDBC_FEATURE_NAMES,
} from './dataset';
import {
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
  verifyHmacSignature,
} from './security';
import {
  GENESIS_PREV_HASH,
  createAuditRecord,
  runControlledTamperTest,
  verifyLedgerChain,
} from './ledger';
import { FederatedCoordinator } from './coordinator';

test('dataset split and hospital separation isolate records with zero overlap', () => {
  const split = prepareDatasetSplit(42, 0.2);
  assert.equal(split.trainX.length + split.testX.length, 569);
  assert.equal(split.trainX[0].length, WDBC_FEATURE_NAMES.length);

  const balanced = partitionHospitalData(split.trainX, split.trainY, 'balanced', 42);
  assert.equal(
    balanced.hospital_a.sampleCount +
      balanced.hospital_b.sampleCount +
      balanced.hospital_c.sampleCount,
    split.trainX.length
  );
  assert.ok(Math.abs(balanced.hospital_a.sampleCount - balanced.hospital_b.sampleCount) <= 2);

  const nonIid = partitionHospitalData(split.trainX, split.trainY, 'non_iid', 42);
  const ratioA = nonIid.hospital_a.malignantCount / nonIid.hospital_a.sampleCount;
  const ratioC = nonIid.hospital_c.malignantCount / nonIid.hospital_c.sampleCount;
  assert.ok(ratioA > ratioC + 0.3, 'Non-IID split should skew class ratios across hospitals');
});

test('local logistic regression training and weighted FedAvg aggregation improve global metrics', () => {
  const split = prepareDatasetSplit(42);
  const parts = partitionHospitalData(split.trainX, split.trainY, 'balanced', 42);

  const init = initializeZeroModel();
  const resA = trainLocalLogisticRegression(parts.hospital_a.trainX, parts.hospital_a.trainY, init);
  const resB = trainLocalLogisticRegression(parts.hospital_b.trainX, parts.hospital_b.trainY, init);
  const resC = trainLocalLogisticRegression(parts.hospital_c.trainX, parts.hospital_c.trainY, init);

  assert.ok(resA.deltaNorm > 0);
  const aggregated = federatedAverage([
    { hospitalId: 'hospital_a', numSamples: parts.hospital_a.sampleCount, parameters: resA.parameters },
    { hospitalId: 'hospital_b', numSamples: parts.hospital_b.sampleCount, parameters: resB.parameters },
    { hospitalId: 'hospital_c', numSamples: parts.hospital_c.sampleCount, parameters: resC.parameters },
  ]);

  // Verify weighted average math directly on bias
  const totalN =
    parts.hospital_a.sampleCount + parts.hospital_b.sampleCount + parts.hospital_c.sampleCount;
  const expectedBias =
    (parts.hospital_a.sampleCount * resA.parameters.bias +
      parts.hospital_b.sampleCount * resB.parameters.bias +
      parts.hospital_c.sampleCount * resC.parameters.bias) /
    totalN;
  assert.ok(Math.abs(aggregated.bias - expectedBias) < 1e-10);

  const metrics = evaluateModel(aggregated, split.testX, split.testY);
  assert.ok(metrics.accuracy > 0.85, `Expected accuracy > 0.85, got ${metrics.accuracy}`);
  assert.ok(metrics.roc_auc > 0.9, `Expected ROC-AUC > 0.90, got ${metrics.roc_auc}`);
});

test('HMAC-SHA256 authentication rejects invalid signatures and forged senders', () => {
  const params = initializeZeroModel();
  params.weights[0] = 0.25;

  const valid = signModelUpdate('hospital_a', 1, 150, params);
  assert.equal(verifyHmacSignature(valid), true);

  const corrupted = { ...valid, hmac_signature: 'f'.repeat(64) };
  assert.equal(verifyHmacSignature(corrupted), false);

  const forged = signModelUpdate('hospital_a', 1, 150, params, 'wrong-key-secret-value');
  assert.equal(verifyHmacSignature(forged), false);

  const globalZero = initializeZeroModel();
  const inspectionCorrupted = inspectAndValidateUpdate(corrupted, globalZero);
  assert.equal(inspectionCorrupted.decision, 'REJECTED');
  assert.equal(inspectionCorrupted.layer_failed, 'AUTHENTICATION');
});

test('malicious-update validation distinguishes valid HMAC from exaggerated and label-flipped updates', () => {
  const coordinator = new FederatedCoordinator(':memory:');
  coordinator.initializeExperiment({ distribution: 'balanced', seed: 42 });
  coordinator.runFederatedTraining(5);

  const exaggerated = coordinator.simulateSecurityExperiment('exaggerated_update', 'hospital_b');
  assert.equal(exaggerated.event.hmac_valid, true);
  assert.equal(exaggerated.event.validation_passed, false);
  assert.equal(exaggerated.event.decision, 'REJECTED');
  assert.equal(exaggerated.event.layer_failed, 'MALICIOUS_VALIDATION');

  const labelFlip = coordinator.simulateSecurityExperiment('label_flipping', 'hospital_c');
  assert.equal(labelFlip.event.hmac_valid, true);
  assert.equal(labelFlip.event.validation_passed, false);
  assert.equal(labelFlip.event.decision, 'REJECTED');
  assert.equal(labelFlip.event.layer_failed, 'MALICIOUS_VALIDATION');

  coordinator.db.close();
});

test('hash-linked audit ledger verifies integrity and detects controlled tampering non-destructively', () => {
  const r1 = createAuditRecord(1, GENESIS_PREV_HASH, {
    hospital_id: 'aggregator',
    training_round: 0,
    event_type: 'EXPERIMENT_INITIALIZED',
    payload_summary: 'Genesis',
    payload_data: { seed: 42 },
  });
  const r2 = createAuditRecord(2, r1.current_hash, {
    hospital_id: 'hospital_a',
    training_round: 1,
    event_type: 'AUTHENTICATED_UPDATE_ACCEPTED',
    payload_summary: 'Round 1 Hospital A',
    payload_data: { round: 1 },
  });

  const validReport = verifyLedgerChain([r1, r2]);
  assert.equal(validReport.valid, true);
  assert.equal(validReport.checked_records, 2);

  const tamperReport = runControlledTamperTest([r1, r2], 1);
  assert.equal(tamperReport.verification_after_tamper.valid, false);
  assert.equal(tamperReport.verification_after_tamper.broken_record_id, 1);
  assert.equal(tamperReport.live_ledger_valid, true);
  assert.equal(tamperReport.live_ledger_untouched, true);
});

test('exact Linear SHAP values satisfy Shapley additivity and CSV export produces valid tabular rows', () => {
  const coordinator = new FederatedCoordinator(':memory:');
  coordinator.initializeExperiment({ distribution: 'balanced', seed: 42 });
  coordinator.runFederatedTraining(3);

  const globalShap = coordinator.getGlobalExplainability();
  assert.equal(globalShap.available, true);
  assert.equal(globalShap.features.length, 30);
  assert.ok(globalShap.features[0].mean_abs_shap > 0);

  const individual = coordinator.explainPrediction(0);
  assert.ok(
    Math.abs(individual.shap_sum_check - individual.predicted_logit) < 1e-3,
    `SHAP efficiency axiom mismatch: ${individual.shap_sum_check} vs ${individual.predicted_logit}`
  );

  const csv = coordinator.exportAnalyticsCsv();
  assert.ok(csv.includes('model_comparison'));
  assert.ok(csv.includes('Federated Global Model (FedAvg)'));
  assert.ok(csv.includes('Round 3'));

  coordinator.db.close();
});
