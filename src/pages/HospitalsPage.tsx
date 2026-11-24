import { useEffect, useState } from 'react';
import {
  EmptyState,
  ErrorBanner,
  SectionCard,
  StatusText,
  formatPercent,
  formatShortTime,
} from '../components/UiPrimitives';
import { api } from '../services/api';
import type { HospitalId, HospitalNode } from '../types/api';

export function HospitalsPage({
  refreshKey,
  onMutated,
  onQuickInitialize,
}: {
  refreshKey: number;
  onMutated: () => void;
  onQuickInitialize: () => Promise<void>;
}) {
  const [hospitals, setHospitals] = useState<HospitalNode[]>([]);
  const [selectedId, setSelectedId] = useState<HospitalId>('hospital_a');
  const [trainingId, setTrainingId] = useState<HospitalId | null>(null);
  const [initializing, setInitializing] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadHospitals = async () => {
    try {
      const res = await api.getHospitals();
      setHospitals(res.hospitals);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load hospital nodes.');
    }
  };

  useEffect(() => {
    loadHospitals();
  }, [refreshKey]);

  const handleTrainLocal = async (hospitalId: HospitalId) => {
    setTrainingId(hospitalId);
    setError(null);
    setFeedback(null);
    try {
      const res = await api.trainHospital(hospitalId);
      setFeedback(
        `${res.hospital.name} trained locally (${res.epochs_completed} epochs · Accuracy ${formatPercent(res.hospital.local_metrics?.accuracy)}).`
      );
      await loadHospitals();
      onMutated();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Local training failed.');
    } finally {
      setTrainingId(null);
    }
  };

  const handleInit = async () => {
    setInitializing(true);
    setError(null);
    try {
      await onQuickInitialize();
      await loadHospitals();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Initialization failed.');
    } finally {
      setInitializing(false);
    }
  };

  const isInitialized = hospitals.some((h) => h.connection_status === 'Connected');
  const selectedHospital = hospitals.find((h) => h.hospital_id === selectedId) ?? hospitals[0];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Hospital Network</h2>
        <p className="mt-0.5 text-sm text-slate-500">
          Each hospital trains on its own private records and shares only model weights.
        </p>
      </div>

      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      {feedback ? (
        <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-sm text-emerald-800">
          {feedback}
        </div>
      ) : null}

      {!isInitialized ? (
        <EmptyState
          title="Hospitals are on standby"
          message="Initialize the experiment to partition the dataset across Hospital A, Hospital B, and Hospital C."
          actionLabel={initializing ? 'Initializing...' : 'Initialize Experiment'}
          onAction={handleInit}
          disabled={initializing}
        />
      ) : null}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {hospitals.map((node) => {
          const isSelected = selectedHospital?.hospital_id === node.hospital_id;
          const isTraining = trainingId === node.hospital_id;
          return (
            <div
              key={node.hospital_id}
              className={`bg-white rounded-lg p-5 border transition-colors ${
                isSelected ? 'border-teal-700' : 'border-slate-200'
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <h3 className="text-base font-semibold text-slate-900">{node.name}</h3>
                  <p className="text-xs text-slate-500">{node.institution_type}</p>
                </div>
                <StatusText
                  tone={node.connection_status === 'Connected' ? 'success' : 'neutral'}
                  label={node.connection_status}
                />
              </div>

              <dl className="mt-4 space-y-2 text-xs border-t border-slate-100 pt-4">
                <div className="flex justify-between">
                  <dt className="text-slate-500">Local Samples</dt>
                  <dd className="font-mono font-medium text-slate-900 tabular-nums">
                    {node.num_samples}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-500">Model Status</dt>
                  <dd className="font-medium text-slate-800">{node.model_status}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-500">Local Accuracy</dt>
                  <dd className="font-mono font-medium text-slate-900 tabular-nums">
                    {node.local_metrics ? formatPercent(node.local_metrics.accuracy) : 'Not trained'}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-slate-500">Last Round</dt>
                  <dd className="font-mono text-slate-700 tabular-nums">
                    {node.last_training_round > 0 ? `Round ${node.last_training_round}` : '—'}
                  </dd>
                </div>
              </dl>

              {/* Minimal progress bar */}
              <div className="mt-4">
                <div className="flex justify-between text-[11px] text-slate-500 mb-1">
                  <span>Training Progress</span>
                  <span className="font-mono tabular-nums">{node.training_progress}%</span>
                </div>
                <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-teal-700 transition-all"
                    style={{ width: `${node.training_progress}%` }}
                  />
                </div>
              </div>

              <div className="mt-5 flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => handleTrainLocal(node.hospital_id)}
                  disabled={!isInitialized || isTraining}
                  className="flex-1 px-3 py-2 text-xs font-medium text-white bg-teal-700 hover:bg-teal-800 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
                >
                  {isTraining ? 'Training...' : 'Train Local Model'}
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedId(node.hospital_id)}
                  className="px-3 py-2 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-md transition-colors whitespace-nowrap"
                >
                  Inspect
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {selectedHospital ? (
        <SectionCard
          title={`${selectedHospital.name} Details`}
          subtitle={`${selectedHospital.institution_type} · ${selectedHospital.region}`}
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
            <div className="space-y-3">
              <h3 className="text-xs font-semibold text-slate-700">Local Dataset Partition</h3>
              <dl className="grid grid-cols-3 gap-3">
                <div className="p-3 bg-slate-50 rounded-md border border-slate-200/70">
                  <dt className="text-xs text-slate-500">Total</dt>
                  <dd className="mt-1 text-lg font-semibold font-mono tabular-nums">
                    {selectedHospital.num_samples}
                  </dd>
                </div>
                <div className="p-3 bg-slate-50 rounded-md border border-slate-200/70">
                  <dt className="text-xs text-slate-500">Benign</dt>
                  <dd className="mt-1 text-lg font-semibold font-mono tabular-nums text-emerald-700">
                    {selectedHospital.benign_count}
                  </dd>
                </div>
                <div className="p-3 bg-slate-50 rounded-md border border-slate-200/70">
                  <dt className="text-xs text-slate-500">Malignant</dt>
                  <dd className="mt-1 text-lg font-semibold font-mono tabular-nums text-red-700">
                    {selectedHospital.malignant_count}
                  </dd>
                </div>
              </dl>
              <p className="text-xs text-slate-500">
                Patient records stay inside {selectedHospital.name}. Only signed model weights are sent to the aggregator.
              </p>
            </div>

            <div className="space-y-3">
              <h3 className="text-xs font-semibold text-slate-700">Held-Out Evaluation Metrics</h3>
              {selectedHospital.local_metrics ? (
                <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs">
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <dt className="text-slate-500">Accuracy</dt>
                    <dd className="font-mono font-medium tabular-nums">
                      {formatPercent(selectedHospital.local_metrics.accuracy)}
                    </dd>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <dt className="text-slate-500">Precision</dt>
                    <dd className="font-mono font-medium tabular-nums">
                      {formatPercent(selectedHospital.local_metrics.precision)}
                    </dd>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <dt className="text-slate-500">Recall</dt>
                    <dd className="font-mono font-medium tabular-nums">
                      {formatPercent(selectedHospital.local_metrics.recall)}
                    </dd>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <dt className="text-slate-500">F1 Score</dt>
                    <dd className="font-mono font-medium tabular-nums">
                      {formatPercent(selectedHospital.local_metrics.f1_score)}
                    </dd>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <dt className="text-slate-500">ROC-AUC</dt>
                    <dd className="font-mono font-medium tabular-nums">
                      {selectedHospital.local_metrics.roc_auc.toFixed(4)}
                    </dd>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <dt className="text-slate-500">Updated</dt>
                    <dd className="font-mono tabular-nums">
                      {formatShortTime(selectedHospital.last_updated_at)}
                    </dd>
                  </div>
                </dl>
              ) : (
                <p className="text-xs text-slate-500">
                  Not evaluated yet. Train this node locally or run a federated round.
                </p>
              )}
            </div>
          </div>
        </SectionCard>
      ) : null}
    </div>
  );
}
