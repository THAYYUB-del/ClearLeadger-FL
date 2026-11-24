/**
 * Wisconsin Breast Cancer Diagnostic (WDBC) dataset generator & preprocessor.
 * Matches sklearn.datasets.load_breast_cancer() schema:
 * - 569 total samples (212 malignant = class 0, 357 benign = class 1)
 * - 30 continuous cytological features (10 mean, 10 SE, 10 worst)
 * - Deterministic reproducible seeding, stratified train/test split,
 *   Z-score standardization fitted strictly on training data,
 *   and isolated 3-hospital partitioning (Balanced IID & Non-IID).
 */

export const WDBC_FEATURE_NAMES: readonly string[] = [
  'mean radius',
  'mean texture',
  'mean perimeter',
  'mean area',
  'mean smoothness',
  'mean compactness',
  'mean concavity',
  'mean concave points',
  'mean symmetry',
  'mean fractal dimension',
  'radius error',
  'texture error',
  'perimeter error',
  'area error',
  'smoothness error',
  'compactness error',
  'concavity error',
  'concave points error',
  'symmetry error',
  'fractal dimension error',
  'worst radius',
  'worst texture',
  'worst perimeter',
  'worst area',
  'worst smoothness',
  'worst compactness',
  'worst concavity',
  'worst concave points',
  'worst symmetry',
  'worst fractal dimension',
] as const;

export type HospitalId = 'hospital_a' | 'hospital_b' | 'hospital_c';
export type DistributionMode = 'balanced' | 'non_iid';

export interface DatasetSplit {
  trainX: number[][];
  trainY: number[];
  testX: number[][];
  testRawX: number[][];
  testY: number[];
  featureMeans: number[];
  featureStds: number[];
  trainStandardizedMeans: number[];
}

export interface HospitalPartition {
  hospitalId: HospitalId;
  name: string;
  institutionType: string;
  region: string;
  /** Raw training records remain strictly encapsulated inside the node */
  trainX: number[][];
  trainY: number[];
  sampleCount: number;
  benignCount: number;
  malignantCount: number;
}

interface FeatureStats {
  benignMean: number;
  benignStd: number;
  malignantMean: number;
  malignantStd: number;
  sizeFactor: number;
  shapeFactor: number;
  textureFactor: number;
  minVal: number;
}

// Empirical class-conditional summary statistics matching UCI WDBC (569 samples, 30 features)
const WDBC_STATS: readonly FeatureStats[] = [
  // 0-9: mean features
  { benignMean: 12.147, benignStd: 1.78, malignantMean: 17.463, malignantStd: 3.20, sizeFactor: 0.88, shapeFactor: 0.25, textureFactor: 0.12, minVal: 6.5 },
  { benignMean: 17.915, benignStd: 3.99, malignantMean: 21.605, malignantStd: 3.78, sizeFactor: 0.18, shapeFactor: 0.15, textureFactor: 0.85, minVal: 9.7 },
  { benignMean: 78.075, benignStd: 11.81, malignantMean: 115.365, malignantStd: 21.85, sizeFactor: 0.90, shapeFactor: 0.30, textureFactor: 0.10, minVal: 43.0 },
  { benignMean: 462.79, benignStd: 134.29, malignantMean: 978.38, malignantStd: 367.94, sizeFactor: 0.92, shapeFactor: 0.20, textureFactor: 0.10, minVal: 143.0 },
  { benignMean: 0.0925, benignStd: 0.0134, malignantMean: 0.1029, malignantStd: 0.0126, sizeFactor: 0.10, shapeFactor: 0.55, textureFactor: 0.05, minVal: 0.052 },
  { benignMean: 0.0801, benignStd: 0.0337, malignantMean: 0.1452, malignantStd: 0.0540, sizeFactor: 0.38, shapeFactor: 0.78, textureFactor: 0.12, minVal: 0.019 },
  { benignMean: 0.0461, benignStd: 0.0434, malignantMean: 0.1608, malignantStd: 0.0750, sizeFactor: 0.48, shapeFactor: 0.84, textureFactor: 0.14, minVal: 0.0 },
  { benignMean: 0.0257, benignStd: 0.0159, malignantMean: 0.0880, malignantStd: 0.0344, sizeFactor: 0.62, shapeFactor: 0.82, textureFactor: 0.12, minVal: 0.0 },
  { benignMean: 0.1742, benignStd: 0.0248, malignantMean: 0.1929, malignantStd: 0.0276, sizeFactor: 0.10, shapeFactor: 0.48, textureFactor: 0.08, minVal: 0.106 },
  { benignMean: 0.0629, benignStd: 0.0067, malignantMean: 0.0627, malignantStd: 0.0076, sizeFactor: -0.25, shapeFactor: 0.42, textureFactor: 0.02, minVal: 0.049 },
  // 10-19: error features
  { benignMean: 0.2841, benignStd: 0.1126, malignantMean: 0.6091, malignantStd: 0.3450, sizeFactor: 0.72, shapeFactor: 0.28, textureFactor: 0.14, minVal: 0.11 },
  { benignMean: 1.2204, benignStd: 0.5892, malignantMean: 1.2109, malignantStd: 0.4832, sizeFactor: -0.05, shapeFactor: 0.08, textureFactor: 0.48, minVal: 0.36 },
  { benignMean: 2.0003, benignStd: 0.7712, malignantMean: 4.3239, malignantStd: 2.5685, sizeFactor: 0.74, shapeFactor: 0.32, textureFactor: 0.14, minVal: 0.75 },
  { benignMean: 21.135, benignStd: 8.843, malignantMean: 72.672, malignantStd: 61.355, sizeFactor: 0.78, shapeFactor: 0.26, textureFactor: 0.12, minVal: 6.8 },
  { benignMean: 0.0072, benignStd: 0.0031, malignantMean: 0.0068, malignantStd: 0.0029, sizeFactor: -0.15, shapeFactor: 0.22, textureFactor: 0.06, minVal: 0.0017 },
  { benignMean: 0.0214, benignStd: 0.0164, malignantMean: 0.0323, malignantStd: 0.0184, sizeFactor: 0.16, shapeFactor: 0.68, textureFactor: 0.10, minVal: 0.0022 },
  { benignMean: 0.0260, benignStd: 0.0329, malignantMean: 0.0418, malignantStd: 0.0216, sizeFactor: 0.20, shapeFactor: 0.64, textureFactor: 0.08, minVal: 0.0 },
  { benignMean: 0.0099, benignStd: 0.0057, malignantMean: 0.0151, malignantStd: 0.0055, sizeFactor: 0.32, shapeFactor: 0.60, textureFactor: 0.08, minVal: 0.0 },
  { benignMean: 0.0206, benignStd: 0.0070, malignantMean: 0.0205, malignantStd: 0.0101, sizeFactor: -0.08, shapeFactor: 0.28, textureFactor: 0.05, minVal: 0.0078 },
  { benignMean: 0.0036, benignStd: 0.0029, malignantMean: 0.0041, malignantStd: 0.0020, sizeFactor: -0.04, shapeFactor: 0.48, textureFactor: 0.04, minVal: 0.0009 },
  // 20-29: worst features
  { benignMean: 13.380, benignStd: 1.981, malignantMean: 21.135, malignantStd: 4.283, sizeFactor: 0.92, shapeFactor: 0.28, textureFactor: 0.14, minVal: 7.9 },
  { benignMean: 23.515, benignStd: 5.494, malignantMean: 29.318, malignantStd: 5.435, sizeFactor: 0.22, shapeFactor: 0.18, textureFactor: 0.88, minVal: 12.0 },
  { benignMean: 87.006, benignStd: 13.527, malignantMean: 141.370, malignantStd: 29.457, sizeFactor: 0.94, shapeFactor: 0.34, textureFactor: 0.12, minVal: 50.4 },
  { benignMean: 558.90, benignStd: 163.60, malignantMean: 1422.29, malignantStd: 597.97, sizeFactor: 0.94, shapeFactor: 0.22, textureFactor: 0.12, minVal: 185.0 },
  { benignMean: 0.1250, benignStd: 0.0200, malignantMean: 0.1448, malignantStd: 0.0219, sizeFactor: 0.16, shapeFactor: 0.56, textureFactor: 0.12, minVal: 0.071 },
  { benignMean: 0.1827, benignStd: 0.0922, malignantMean: 0.3748, malignantStd: 0.1704, sizeFactor: 0.38, shapeFactor: 0.82, textureFactor: 0.18, minVal: 0.027 },
  { benignMean: 0.1662, benignStd: 0.1404, malignantMean: 0.4506, malignantStd: 0.1815, sizeFactor: 0.48, shapeFactor: 0.86, textureFactor: 0.18, minVal: 0.0 },
  { benignMean: 0.0744, benignStd: 0.0358, malignantMean: 0.1822, malignantStd: 0.0463, sizeFactor: 0.64, shapeFactor: 0.88, textureFactor: 0.16, minVal: 0.0 },
  { benignMean: 0.2702, benignStd: 0.0417, malignantMean: 0.3235, malignantStd: 0.0747, sizeFactor: 0.16, shapeFactor: 0.52, textureFactor: 0.12, minVal: 0.156 },
  { benignMean: 0.0794, benignStd: 0.0138, malignantMean: 0.0915, malignantStd: 0.0216, sizeFactor: 0.05, shapeFactor: 0.62, textureFactor: 0.10, minVal: 0.055 },
];

export function createSeededRng(seed: number): () => number {
  let state = (seed >>> 0) || 42;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleStandardNormal(rng: () => number): number {
  let u1 = rng();
  let u2 = rng();
  while (u1 <= 1e-12) u1 = rng();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

function seededShuffle<T>(arr: T[], rng: () => number): T[] {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = copy[i];
    copy[i] = copy[j];
    copy[j] = tmp;
  }
  return copy;
}

/**
 * Generates the 569-sample, 30-feature WDBC dataset matching UCI/sklearn summary
 * moments and realistic cytological covariance structure.
 */
export function loadWdbcDataset(seed = 42): { X: number[][]; y: number[] } {
  const rng = createSeededRng(seed);
  const numMalignant = 212;
  const numBenign = 357;
  const samples: Array<{ features: number[]; label: number }> = [];

  for (let i = 0; i < numMalignant + numBenign; i++) {
    const isBenign = i >= numMalignant;
    const label = isBenign ? 1 : 0;

    // Latent clinical morphology factors (size, irregularity/concavity, tissue texture)
    const latentSize = sampleStandardNormal(rng);
    const latentShape = sampleStandardNormal(rng);
    const latentTexture = sampleStandardNormal(rng);

    // ~4% borderline overlapping clinical presentations (realistic diagnostic boundary)
    const borderlineShift = rng() < 0.045 ? (isBenign ? 1.15 : -1.15) : 0;

    const features: number[] = WDBC_STATS.map((stat) => {
      const mean = isBenign ? stat.benignMean : stat.malignantMean;
      const std = isBenign ? stat.benignStd : stat.malignantStd;
      const latent =
        stat.sizeFactor * latentSize +
        stat.shapeFactor * latentShape +
        stat.textureFactor * latentTexture;
      const factorNorm = Math.sqrt(
        stat.sizeFactor * stat.sizeFactor +
          stat.shapeFactor * stat.shapeFactor +
          stat.textureFactor * stat.textureFactor +
          0.36
      );
      const indepNoise = sampleStandardNormal(rng) * 0.6;
      const z = (latent + indepNoise) / factorNorm + borderlineShift * 0.55;
      const val = Math.max(stat.minVal, mean + z * std);
      return Number(val.toFixed(6));
    });

    samples.push({ features, label });
  }

  const shuffled = seededShuffle(samples, rng);
  return {
    X: shuffled.map((s) => s.features),
    y: shuffled.map((s) => s.label),
  };
}

/**
 * Performs stratified 80/20 train/test split and fits Z-score standardization
 * strictly on training data to prevent leakage into the held-out test set.
 */
export function prepareDatasetSplit(seed = 42, testRatio = 0.2): DatasetSplit {
  const { X, y } = loadWdbcDataset(seed);
  const rng = createSeededRng(seed + 101);

  const malignantIdx: number[] = [];
  const benignIdx: number[] = [];
  for (let i = 0; i < y.length; i++) {
    if (y[i] === 0) malignantIdx.push(i);
    else benignIdx.push(i);
  }

  const mShuffled = seededShuffle(malignantIdx, rng);
  const bShuffled = seededShuffle(benignIdx, rng);

  const mTestCount = Math.round(mShuffled.length * testRatio);
  const bTestCount = Math.round(bShuffled.length * testRatio);

  const testIndices = seededShuffle(
    [...mShuffled.slice(0, mTestCount), ...bShuffled.slice(0, bTestCount)],
    rng
  );
  const trainIndices = seededShuffle(
    [...mShuffled.slice(mTestCount), ...bShuffled.slice(bTestCount)],
    rng
  );

  const rawTrainX = trainIndices.map((idx) => X[idx]);
  const trainY = trainIndices.map((idx) => y[idx]);
  const testRawX = testIndices.map((idx) => X[idx]);
  const testY = testIndices.map((idx) => y[idx]);

  const numFeatures = WDBC_FEATURE_NAMES.length;
  const featureMeans = new Array<number>(numFeatures).fill(0);
  const featureStds = new Array<number>(numFeatures).fill(0);

  for (let j = 0; j < numFeatures; j++) {
    let sum = 0;
    for (let i = 0; i < rawTrainX.length; i++) sum += rawTrainX[i][j];
    featureMeans[j] = sum / rawTrainX.length;

    let sqSum = 0;
    for (let i = 0; i < rawTrainX.length; i++) {
      const diff = rawTrainX[i][j] - featureMeans[j];
      sqSum += diff * diff;
    }
    const std = Math.sqrt(sqSum / rawTrainX.length);
    featureStds[j] = std > 1e-8 ? std : 1.0;
  }

  const standardize = (matrix: number[][]) =>
    matrix.map((row) =>
      row.map((val, j) => (val - featureMeans[j]) / featureStds[j])
    );

  const trainX = standardize(rawTrainX);
  const testX = standardize(testRawX);

  const trainStandardizedMeans = new Array<number>(numFeatures).fill(0);
  for (let j = 0; j < numFeatures; j++) {
    let s = 0;
    for (let i = 0; i < trainX.length; i++) s += trainX[i][j];
    trainStandardizedMeans[j] = s / trainX.length;
  }

  return {
    trainX,
    trainY,
    testX,
    testRawX,
    testY,
    featureMeans,
    featureStds,
    trainStandardizedMeans,
  };
}

export const HOSPITAL_METADATA: Record<
  HospitalId,
  { name: string; institutionType: string; region: string }
> = {
  hospital_a: {
    name: 'Hospital A',
    institutionType: 'Metropolitan Oncology Center',
    region: 'North Clinical Network',
  },
  hospital_b: {
    name: 'Hospital B',
    institutionType: 'Regional Medical Center',
    region: 'Central Clinical Network',
  },
  hospital_c: {
    name: 'Hospital C',
    institutionType: 'Community Screening Hospital',
    region: 'West Clinical Network',
  },
};

/**
 * Partitions training records into three completely disjoint hospital datasets.
 * Supports both Balanced (IID) and Non-IID clinical case-mix distributions.
 */
export function partitionHospitalData(
  trainX: number[][],
  trainY: number[],
  distribution: DistributionMode,
  seed = 42
): Record<HospitalId, HospitalPartition> {
  const rng = createSeededRng(seed + 303);
  const malignantIdx: number[] = [];
  const benignIdx: number[] = [];

  for (let i = 0; i < trainY.length; i++) {
    if (trainY[i] === 0) malignantIdx.push(i);
    else benignIdx.push(i);
  }

  const mShuffled = seededShuffle(malignantIdx, rng);
  const bShuffled = seededShuffle(benignIdx, rng);

  let idxA: number[] = [];
  let idxB: number[] = [];
  let idxC: number[] = [];

  if (distribution === 'balanced') {
    // Stratified equal partition across all three hospitals
    mShuffled.forEach((idx, i) => {
      if (i % 3 === 0) idxA.push(idx);
      else if (i % 3 === 1) idxB.push(idx);
      else idxC.push(idx);
    });
    bShuffled.forEach((idx, i) => {
      if (i % 3 === 0) idxA.push(idx);
      else if (i % 3 === 1) idxB.push(idx);
      else idxC.push(idx);
    });
  } else {
    // Non-IID skewed clinical case mix:
    // Hospital A (Oncology Referral): 65% of malignant, 20% of benign
    // Hospital B (Regional Center): 25% of malignant, 40% of benign
    // Hospital C (Screening Clinic): 10% of malignant, 40% of benign
    const mA = Math.round(mShuffled.length * 0.65);
    const mB = Math.round(mShuffled.length * 0.25);
    const bA = Math.round(bShuffled.length * 0.20);
    const bB = Math.round(bShuffled.length * 0.40);

    idxA = [...mShuffled.slice(0, mA), ...bShuffled.slice(0, bA)];
    idxB = [...mShuffled.slice(mA, mA + mB), ...bShuffled.slice(bA, bA + bB)];
    idxC = [...mShuffled.slice(mA + mB), ...bShuffled.slice(bA + bB)];
  }

  const buildNode = (hospitalId: HospitalId, indices: number[]): HospitalPartition => {
    const shuffledIndices = seededShuffle(indices, rng);
    const nodeX = shuffledIndices.map((idx) => [...trainX[idx]]);
    const nodeY = shuffledIndices.map((idx) => trainY[idx]);
    const benignCount = nodeY.filter((v) => v === 1).length;
    const malignantCount = nodeY.length - benignCount;
    const meta = HOSPITAL_METADATA[hospitalId];
    return {
      hospitalId,
      name: meta.name,
      institutionType: meta.institutionType,
      region: meta.region,
      trainX: nodeX,
      trainY: nodeY,
      sampleCount: nodeY.length,
      benignCount,
      malignantCount,
    };
  };

  return {
    hospital_a: buildNode('hospital_a', idxA),
    hospital_b: buildNode('hospital_b', idxB),
    hospital_c: buildNode('hospital_c', idxC),
  };
}
