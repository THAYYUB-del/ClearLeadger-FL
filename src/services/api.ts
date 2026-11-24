import type {
  AnalyticsResponse,
  DashboardSummary,
  DistributionMode,
  FederatedStatus,
  GlobalExplainabilityResponse,
  HospitalId,
  HospitalNode,
  IndividualPredictionExplanation,
  LedgerResponse,
  LedgerVerificationReport,
  SecurityEventsResponse,
  SecurityScenario,
  SecuritySimulateResponse,
  TamperTestReport,
  TrainingRound,
} from '../types/api';

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      if (payload?.detail) message = String(payload.detail);
    } catch {
      // Ignore JSON parse errors on non-JSON error responses
    }
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export const api = {
  getHealth: () => apiRequest<{ status: string; service: string }>('/api/health'),
  getDashboard: () => apiRequest<DashboardSummary>('/api/dashboard'),
  getHospitals: () => apiRequest<{ hospitals: HospitalNode[] }>('/api/hospitals'),
  getHospitalDetail: (hospitalId: HospitalId) =>
    apiRequest<HospitalNode>(`/api/hospitals/${hospitalId}`),
  trainHospital: (hospitalId: HospitalId, localEpochs?: number) =>
    apiRequest<{
      hospital: HospitalNode;
      train_loss: number;
      train_accuracy: number;
      delta_norm: number;
      epochs_completed: number;
    }>(`/api/hospitals/${hospitalId}/train`, {
      method: 'POST',
      body: JSON.stringify(localEpochs ? { local_epochs: localEpochs } : {}),
    }),
  initializeExperiment: (params: {
    distribution: DistributionMode;
    seed?: number;
    local_epochs?: number;
    learning_rate?: number;
  }) =>
    apiRequest<FederatedStatus>('/api/experiment/initialize', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
  resetExperiment: () =>
    apiRequest<{ status: string; message: string }>('/api/experiment/reset', {
      method: 'POST',
    }),
  getFederatedStatus: () => apiRequest<FederatedStatus>('/api/federated/status'),
  getFederatedRounds: () => apiRequest<{ rounds: TrainingRound[] }>('/api/federated/rounds'),
  startFederatedTraining: (rounds: number) =>
    apiRequest<{
      rounds_executed: number;
      current_round: number;
      latest_round: TrainingRound;
    }>('/api/federated/start', {
      method: 'POST',
      body: JSON.stringify({ rounds }),
    }),
  getSecurityEvents: () => apiRequest<SecurityEventsResponse>('/api/security/events'),
  simulateSecurity: (scenario: SecurityScenario, hospitalId: HospitalId) =>
    apiRequest<SecuritySimulateResponse>('/api/security/simulate', {
      method: 'POST',
      body: JSON.stringify({ scenario, hospital_id: hospitalId }),
    }),
  getLedger: () => apiRequest<LedgerResponse>('/api/ledger'),
  verifyLedger: () =>
    apiRequest<LedgerVerificationReport>('/api/ledger/verify', { method: 'POST' }),
  runTamperTest: (targetRecordId?: number) =>
    apiRequest<TamperTestReport>('/api/ledger/tamper-test', {
      method: 'POST',
      body: JSON.stringify(
        targetRecordId !== undefined ? { target_record_id: targetRecordId } : {}
      ),
    }),
  getGlobalExplainability: () =>
    apiRequest<GlobalExplainabilityResponse>('/api/explainability/global'),
  explainPrediction: (sampleIndex: number) =>
    apiRequest<IndividualPredictionExplanation>('/api/explainability/predict', {
      method: 'POST',
      body: JSON.stringify({ sample_index: sampleIndex }),
    }),
  getAnalytics: () => apiRequest<AnalyticsResponse>('/api/analytics'),
  downloadAnalyticsCsv: async () => {
    const response = await fetch('/api/analytics/export');
    if (!response.ok) {
      let message = 'Unable to export CSV.';
      try {
        const payload = await response.json();
        if (payload?.detail) message = String(payload.detail);
      } catch {
        // ignore
      }
      throw new Error(message);
    }
    const text = await response.text();
    const blob = new Blob([text], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'clearledger-fl-analytics.csv');
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    return text;
  },
};
