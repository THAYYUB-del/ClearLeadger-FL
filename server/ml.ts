/**
 * Pure numerical Logistic Regression, exact sample-weighted FedAvg,
 * evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix),
 * and exact Interventional Linear SHAP explanations.
 */

import { WDBC_FEATURE_NAMES } from './dataset';

export interface ModelParameters {
  weights: number[];
  bias: number;
}

export interface ConfusionMatrix {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

export interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  loss: number;
  confusion_matrix: ConfusionMatrix;
  sample_count: number;
}

export interface LocalTrainResult {
  parameters: ModelParameters;
  deltaWeights: number[];
  deltaBias: number;
  deltaNorm: number;
  trainLoss: number;
  trainAccuracy: number;
  epochsCompleted: number;
}

export interface WeightedUpdate {
  hospitalId: string;
  numSamples: number;
  parameters: ModelParameters;
}

export interface GlobalShapFeature {
  feature_index: number;
  feature_name: string;
  mean_abs_shap: number;
  weight: number;
  direction: 'benign_leaning' | 'malignant_leaning';
}

export interface IndividualShapContribution {
  feature_index: number;
  feature_name: string;
  raw_value: number;
  standardized_value: number;
  shap_value: number;
  direction: 'positive' | 'negative';
}

export interface IndividualPredictionExplanation {
  sample_index: number;
  actual_label: number;
  actual_class: 'Benign' | 'Malignant';
  predicted_label: number;
  predicted_class: 'Benign' | 'Malignant';
  probability_benign: number;
  probability_malignant: number;
  base_value_logit: number;
  predicted_logit: number;
  shap_sum_check: number;
  contributions: IndividualShapContribution[];
  top_positive: IndividualShapContribution[];
  top_negative: IndividualShapContribution[];
}

export function sigmoid(z: number): number {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

export function initializeZeroModel(numFeatures = WDBC_FEATURE_NAMES.length): ModelParameters {
  return {
    weights: new Array<number>(numFeatures).fill(0),
    bias: 0,
  };
}

export function cloneParameters(params: ModelParameters): ModelParameters {
  return {
    weights: [...params.weights],
    bias: params.bias,
  };
}

export function predictLogit(params: ModelParameters, x: number[]): number {
  let z = params.bias;
  for (let j = 0; j < params.weights.length; j++) {
    z += params.weights[j] * x[j];
  }
  return z;
}

export function predictProbabilities(params: ModelParameters, X: number[][]): number[] {
  return X.map((row) => sigmoid(predictLogit(params, row)));
}

export function l2Norm(vec: number[]): number {
  let sumSq = 0;
  for (let i = 0; i < vec.length; i++) sumSq += vec[i] * vec[i];
  return Math.sqrt(sumSq);
}

export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA < 1e-12 || normB < 1e-12) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Trains a local logistic regression model with L2 regularization using
 * gradient descent starting from `initialParams`.
 */
export function trainLocalLogisticRegression(
  X: number[][],
  y: number[],
  initialParams: ModelParameters,
  options: { learningRate?: number; epochs?: number; l2Lambda?: number } = {}
): LocalTrainResult {
  const lr = options.learningRate ?? 0.12;
  const epochs = options.epochs ?? 12;
  const l2Lambda = options.l2Lambda ?? 0.015;

  const n = X.length;
  const d = initialParams.weights.length;
  const weights = [...initialParams.weights];
  let bias = initialParams.bias;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const gradW = new Array<number>(d).fill(0);
    let gradB = 0;

    for (let i = 0; i < n; i++) {
      let z = bias;
      const xi = X[i];
      for (let j = 0; j < d; j++) z += weights[j] * xi[j];
      const err = sigmoid(z) - y[i];
      gradB += err;
      for (let j = 0; j < d; j++) gradW[j] += err * xi[j];
    }

    for (let j = 0; j < d; j++) {
      weights[j] -= lr * (gradW[j] / n + l2Lambda * weights[j]);
    }
    bias -= lr * (gradB / n);
  }

  const deltaWeights = weights.map((w, j) => w - initialParams.weights[j]);
  const deltaBias = bias - initialParams.bias;
  const deltaNorm = l2Norm([...deltaWeights, deltaBias]);

  const metrics = evaluateModel({ weights, bias }, X, y);

  return {
    parameters: { weights, bias },
    deltaWeights,
    deltaBias,
    deltaNorm,
    trainLoss: metrics.loss,
    trainAccuracy: metrics.accuracy,
    epochsCompleted: epochs,
  };
}

/**
 * Sample-weighted Federated Averaging (McMahan et al., 2017):
 * w_global = sum_k (n_k / N) * w_k, b_global = sum_k (n_k / N) * b_k
 */
export function federatedAverage(updates: WeightedUpdate[]): ModelParameters {
  if (updates.length === 0) {
    throw new Error('Cannot run FedAvg with zero accepted updates.');
  }

  const totalSamples = updates.reduce((acc, u) => acc + u.numSamples, 0);
  if (totalSamples <= 0) {
    throw new Error('Total sample count across updates must be positive.');
  }

  const numFeatures = updates[0].parameters.weights.length;
  const aggWeights = new Array<number>(numFeatures).fill(0);
  let aggBias = 0;

  for (const update of updates) {
    if (update.parameters.weights.length !== numFeatures) {
      throw new Error(`Parameter dimension mismatch from ${update.hospitalId}`);
    }
    const weightFactor = update.numSamples / totalSamples;
    for (let j = 0; j < numFeatures; j++) {
      aggWeights[j] += weightFactor * update.parameters.weights[j];
    }
    aggBias += weightFactor * update.parameters.bias;
  }

  return { weights: aggWeights, bias: aggBias };
}

/**
 * Computes exact trapezoidal ROC-AUC over predicted probabilities.
 */
export function computeRocAuc(yTrue: number[], probs: number[]): number {
  const paired = yTrue.map((label, i) => ({ label, prob: probs[i] }));
  paired.sort((a, b) => b.prob - a.prob);

  const totalPos = yTrue.filter((v) => v === 1).length;
  const totalNeg = yTrue.length - totalPos;
  if (totalPos === 0 || totalNeg === 0) return 0.5;

  let tp = 0;
  let fp = 0;
  let prevTpr = 0;
  let prevFpr = 0;
  let auc = 0;

  let i = 0;
  while (i < paired.length) {
    const score = paired[i].prob;
    while (i < paired.length && paired[i].prob === score) {
      if (paired[i].label === 1) tp++;
      else fp++;
      i++;
    }
    const tpr = tp / totalPos;
    const fpr = fp / totalNeg;
    auc += ((fpr - prevFpr) * (tpr + prevTpr)) / 2;
    prevTpr = tpr;
    prevFpr = fpr;
  }

  return Math.max(0, Math.min(1, auc));
}

export function evaluateModel(
  params: ModelParameters,
  X: number[][],
  y: number[]
): EvaluationMetrics {
  const n = X.length;
  const probs = predictProbabilities(params, X);

  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;
  let lossSum = 0;
  const eps = 1e-12;

  for (let i = 0; i < n; i++) {
    const p = Math.min(1 - eps, Math.max(eps, probs[i]));
    const yi = y[i];
    lossSum += -(yi * Math.log(p) + (1 - yi) * Math.log(1 - p));
    const pred = probs[i] >= 0.5 ? 1 : 0;
    if (pred === 1 && yi === 1) tp++;
    else if (pred === 0 && yi === 0) tn++;
    else if (pred === 1 && yi === 0) fp++;
    else fn++;
  }

  const accuracy = (tp + tn) / n;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const rocAuc = computeRocAuc(y, probs);

  return {
    accuracy: Number(accuracy.toFixed(4)),
    precision: Number(precision.toFixed(4)),
    recall: Number(recall.toFixed(4)),
    f1_score: Number(f1.toFixed(4)),
    roc_auc: Number(rocAuc.toFixed(4)),
    loss: Number((lossSum / n).toFixed(4)),
    confusion_matrix: { tp, tn, fp, fn },
    sample_count: n,
  };
}

/**
 * Computes exact Interventional Linear SHAP global importance across testX:
 * phi_j(x) = w_j * (x_j - E[X_train_j])
 * Global importance_j = mean_i |phi_j(x_i)|
 */
export function computeGlobalShap(
  params: ModelParameters,
  testX: number[][],
  trainBaselineMeans: number[]
): GlobalShapFeature[] {
  const numFeatures = params.weights.length;
  const n = testX.length;

  const features: GlobalShapFeature[] = [];
  for (let j = 0; j < numFeatures; j++) {
    let absSum = 0;
    const wj = params.weights[j];
    const muj = trainBaselineMeans[j];
    for (let i = 0; i < n; i++) {
      absSum += Math.abs(wj * (testX[i][j] - muj));
    }
    features.push({
      feature_index: j,
      feature_name: WDBC_FEATURE_NAMES[j],
      mean_abs_shap: Number((absSum / n).toFixed(5)),
      weight: Number(wj.toFixed(5)),
      direction: wj >= 0 ? 'benign_leaning' : 'malignant_leaning',
    });
  }

  features.sort((a, b) => b.mean_abs_shap - a.mean_abs_shap);
  return features;
}

/**
 * Computes exact Interventional Linear SHAP explanation for an individual
 * held-out test sample, verifying Shapley efficiency:
 * base_value + sum_j phi_j(x) == logit(x)
 */
export function explainPredictionShap(
  params: ModelParameters,
  sampleIndex: number,
  standardizedX: number[],
  rawX: number[],
  actualLabel: number,
  trainBaselineMeans: number[]
): IndividualPredictionExplanation {
  let baseLogit = params.bias;
  for (let j = 0; j < params.weights.length; j++) {
    baseLogit += params.weights[j] * trainBaselineMeans[j];
  }

  const contributions: IndividualShapContribution[] = params.weights.map((wj, j) => {
    const phi = wj * (standardizedX[j] - trainBaselineMeans[j]);
    return {
      feature_index: j,
      feature_name: WDBC_FEATURE_NAMES[j],
      raw_value: Number(rawX[j].toFixed(4)),
      standardized_value: Number(standardizedX[j].toFixed(4)),
      shap_value: Number(phi.toFixed(5)),
      direction: phi >= 0 ? 'positive' : 'negative',
    };
  });

  const shapSum = contributions.reduce((acc, c) => acc + c.shap_value, 0);
  const predictedLogit = predictLogit(params, standardizedX);
  const probBenign = sigmoid(predictedLogit);
  const predictedLabel = probBenign >= 0.5 ? 1 : 0;

  const sortedByAbs = [...contributions].sort(
    (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)
  );
  const topPositive = contributions
    .filter((c) => c.shap_value > 0)
    .sort((a, b) => b.shap_value - a.shap_value)
    .slice(0, 8);
  const topNegative = contributions
    .filter((c) => c.shap_value < 0)
    .sort((a, b) => a.shap_value - b.shap_value)
    .slice(0, 8);

  return {
    sample_index: sampleIndex,
    actual_label: actualLabel,
    actual_class: actualLabel === 1 ? 'Benign' : 'Malignant',
    predicted_label: predictedLabel,
    predicted_class: predictedLabel === 1 ? 'Benign' : 'Malignant',
    probability_benign: Number(probBenign.toFixed(4)),
    probability_malignant: Number((1 - probBenign).toFixed(4)),
    base_value_logit: Number(baseLogit.toFixed(5)),
    predicted_logit: Number(predictedLogit.toFixed(5)),
    shap_sum_check: Number((baseLogit + shapSum).toFixed(5)),
    contributions: sortedByAbs,
    top_positive: topPositive,
    top_negative: topNegative,
  };
}
