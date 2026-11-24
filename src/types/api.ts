export type PageId =
  | 'overview'
  | 'hospitals'
  | 'federated'
  | 'security'
  | 'ledger'
  | 'explainability'
  | 'analytics';

export type HospitalId = 'hospital_a' | 'hospital_b' | 'hospital_c';
export type DistributionMode = 'balanced' | 'non_iid';
export type SecurityScenario =
  | 'valid_update'
  | 'invalid_hmac'
  | 'forged_sender'
  | 'exaggerated_update'
  | 'label_flipping';

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

export interface HospitalNode {
  hospital_id: HospitalId;
  name: string;
  institution_type: string;
  region: string;
  connection_status: 'Connected' | 'Standby';
  model_status: string;
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

export interface DashboardSummary {
  experiment_initialized: boolean;
  federated_status: string;
  connected_hospitals: number;
  total_hospitals: number;
  current_round: number;
  global_accuracy: number | null;
  global_roc_auc: number | null;
  global_loss: number | null;
  total_authenticated_updates: number;
  security_alerts: number;
  ledger_records_count: number;
  ledger_integrity: {
    valid: boolean;
    status: string;
    verified_at: string;
  };
  distribution: DistributionMode;
  performance_series: Array<{
    round: number;
    accuracy: number;
    f1_score: number;
    roc_auc: number;
    loss: number;
  }>;
  recent_activity: Array<{
    id: number;
    timestamp: string;
    category: string;
    title: string;
    detail: string;
    severity: 'info' | 'success' | 'warning' | 'alert';
  }>;
}

export interface FederatedStatus {
  initialized: boolean;
  status: string;
  distribution: DistributionMode;
  seed: number;
  local_epochs: number;
  learning_rate: number;
  current_round: number;
  total_train_samples: number;
  total_test_samples: number;
  global_metrics: EvaluationMetrics | null;
  hospitals: HospitalNode[];
  rounds_count: number;
}

export interface TrainingRound {
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
  confusion_matrix: ConfusionMatrix;
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

export interface SecurityEvent {
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

export interface SecurityEventsResponse {
  summary: {
    total_events: number;
    accepted: number;
    rejected_total: number;
    rejected_authentication: number;
    rejected_malicious_validation: number;
  };
  events: SecurityEvent[];
}

export interface SecuritySimulateResponse {
  event: SecurityEvent;
  explanation: string;
  honest_reference_norm: number;
}

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

export interface LedgerResponse {
  architecture_note: string;
  records: AuditRecord[];
  verification: LedgerVerificationReport;
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

export interface GlobalShapFeature {
  feature_index: number;
  feature_name: string;
  mean_abs_shap: number;
  weight: number;
  direction: 'benign_leaning' | 'malignant_leaning';
}

export interface GlobalExplainabilityResponse {
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

export interface AnalyticsResponse {
  evaluated: boolean;
  distribution: DistributionMode;
  current_round: number;
  dataset_info: {
    name: string;
    total_samples: number;
    feature_count: number;
    train_samples: number;
    held_out_test_samples: number;
    educational_notice: string;
  };
  global_metrics: EvaluationMetrics | null;
  centralized_metrics: EvaluationMetrics | null;
  comparison: Array<{
    model_name: string;
    scope: string;
    metrics: EvaluationMetrics | null;
  }>;
  rounds: TrainingRound[];
}
