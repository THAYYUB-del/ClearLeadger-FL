import { useEffect, useState } from 'react';
import { RoundPerformanceChart } from '../charts/PerformanceCharts';
import {
  EmptyState,
  ErrorBanner,
  MetricTile,
  SectionCard,
  StatusText,
  formatPercent,
} from '../components/UiPrimitives';
import { api } from '../services/api';
import type { DistributionMode, FederatedStatus, TrainingRound } from '../types/api';

export function FederatedPage({
  refreshKey,
  onMutated,
}: {
  refreshKey: number;
  onMutated: () => void;
}) {
  const [status, setStatus] = useState<FederatedStatus | null>(null);
  const [rounds, setRounds] = useState<TrainingRound[]>([]);
  const [distribution, setDistribution] = useState<DistributionMode>('balanced');
  const [numRounds, setNumRounds] = useState<number>(3);
  const [busyAction, setBusyAction] = useState<'init' | 'train' | 'reset' | null>(null);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      const [statusRes, roundsRes] = await Promise.all([
        api.getFederatedStatus(),
        api.getFederatedRounds(),
      ]);
      setStatus(statusRes);
      setRounds(roundsRes.rounds);
      if (statusRes.initialized) {
        setDistribution(statusRes.distribution);
      }
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load federated status.');
    }
  };

  useEffect(() => {
    loadData();
  }, [refreshKey]);

  const handleInitialize = async () => {
    setBusyAction('init');
    setError(null);
    setFeedback(null);
    try {
      const res = await api.initializeExperiment({ distribution, seed: 42 });
      setStatus(res);
      setRounds([]);
      setFeedback(
        `Experiment initialized (${distribution === 'balanced' ? 'Balanced IID' : 'Non-IID'} split · ${res.total_train_samples} train / ${res.total_test_samples} test samples).`
      );
      onMutated();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to initialize experiment.');
    } finally {
      setBusyAction(null);
    }
  };

  const handleStartTraining = async () => {
    setBusyAction('train');
    setError(null);
    setFeedback(null);
    try {
      const res = await api.startFederatedTraining(numRounds);
      setFeedback(
        `Completed ${res.rounds_executed} federated ${res.rounds_executed === 1 ? 'round' : 'rounds'}. Global accuracy is now ${formatPercent(res.latest_round.global_accuracy)}.`
      );
      await loadData();
      onMutated();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Federated training failed.');
    } finally {
      setBusyAction(null);
    }
  };

  const handleReset = async () => {
    setBusyAction('reset');
    setError(null);
    setFeedback(null);
    try {
      await api.resetExperiment();
      await loadData();
      setFeedback('Experiment reset.');
      onMutated();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset experiment.');
    } finally {
      setBusyAction(null);
    }
  };

  const latestRound = rounds.length > 0 ? rounds[rounds.length - 1] : null;
  const chartData = rounds.map((r) => ({
    round: r.round_number,
    accuracy: Number((r.global_accuracy * 100).toFixed(2)),
    f1_score: Number((r.global_f1 * 100).toFixed(2)),
    roc_auc: Number((r.global_roc_auc * 100).toFixed(2)),
    loss: r.global_loss,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Federated Learning (FedAvg)</h2>
        <p className="mt-0.5 text-sm text-slate-500">
          Initialize hospital data splits, run local training, verify updates, and aggregate into the global model.
        </p>
      </div>

      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      {feedback ? (
        <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-sm text-emerald-800">
          {feedback}
        </div>
      ) : null}

      {/* Controls Panel */}
      <SectionCard
        title="Experiment Controls"
        subtitle="Select data distribution and number of FedAvg rounds."
      >
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <label
              htmlFor="distribution-select"
              className="block text-xs font-medium text-slate-700 mb-1.5"
            >
              Data Distribution
            </label>
            <select
              id="distribution-select"
              value={distribution}
              onChange={(e) => setDistribution(e.target.value as DistributionMode)}
              className="h-9 px-3 text-sm bg-white border border-slate-300 rounded-md text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-600"
            >
              <option value="balanced">Balanced (IID)</option>
              <option value="non_iid">Skewed (Non-IID)</option>
            </select>
          </div>

          <div>
            <label
              htmlFor="rounds-select"
              className="block text-xs font-medium text-slate-700 mb-1.5"
            >
              Rounds to Run
            </label>
            <select
              id="rounds-select"
              value={numRounds}
              onChange={(e) => setNumRounds(Number(e.target.value))}
              className="h-9 px-3 text-sm bg-white border border-slate-300 rounded-md text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-600"
            >
              {[1, 2, 3, 5, 10].map((n) => (
                <option key={n} value={n}>
                  {n} {n === 1 ? 'Round' : 'Rounds'}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 ml-auto">
            <button
              type="button"
              onClick={handleInitialize}
              disabled={busyAction !== null}
              className="h-9 px-4 text-sm font-medium text-slate-800 bg-white border border-slate-300 hover:bg-slate-50 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
            >
              {busyAction === 'init' ? 'Initializing...' : 'Initialize Experiment'}
            </button>

            <button
              type="button"
              onClick={handleStartTraining}
              disabled={busyAction !== null || !status?.initialized}
              className="h-9 px-4 text-sm font-medium text-white bg-teal-700 hover:bg-teal-800 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
            >
              {busyAction === 'train' ? 'Training...' : 'Start Federated Training'}
            </button>

            <button
              type="button"
              onClick={handleReset}
              disabled={busyAction !== null}
              className="h-9 px-3.5 text-sm font-medium text-red-700 bg-white border border-red-200 hover:bg-red-50 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
            >
              {busyAction === 'reset' ? 'Resetting...' : 'Reset'}
            </button>
          </div>
        </div>
      </SectionCard>

      {/* Current Global Summary */}
      {status ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricTile label="Status" value={status.status} />
          <MetricTile
            label="Completed Rounds"
            value={String(status.current_round)}
            subtext={
              status.initialized
                ? `${status.total_train_samples} train · ${status.total_test_samples} test`
                : 'Not partitioned'
            }
          />
          <MetricTile
            label="Global Accuracy"
            value={
              status.global_metrics
                ? formatPercent(status.global_metrics.accuracy)
                : 'Not evaluated'
            }
            tone={status.global_metrics ? 'success' : 'default'}
          />
          <MetricTile
            label="Global Loss"
            value={status.global_metrics ? status.global_metrics.loss.toFixed(4) : '—'}
            subtext={
              status.global_metrics
                ? `ROC-AUC ${status.global_metrics.roc_auc.toFixed(4)}`
                : undefined
            }
          />
        </div>
      ) : null}

      {/* Chart + Latest Hospital Contributions */}
      {rounds.length > 0 ? (
        <>
          <SectionCard
            title="Global Model Progress"
            subtitle="Held-out test metrics after each FedAvg aggregation round."
          >
            <RoundPerformanceChart data={chartData} />
          </SectionCard>

          {latestRound ? (
            <SectionCard
              title={`Hospital Contributions — Round ${latestRound.round_number}`}
              subtitle="Sample-weighted FedAvg shares and local vs global accuracy."
            >
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500">
                      <th className="py-2.5 pr-4 font-medium">Hospital</th>
                      <th className="py-2.5 px-4 font-medium text-right">Samples</th>
                      <th className="py-2.5 px-4 font-medium text-right">FedAvg Weight</th>
                      <th className="py-2.5 px-4 font-medium text-right">Local Accuracy</th>
                      <th className="py-2.5 px-4 font-medium text-right">Update Norm</th>
                      <th className="py-2.5 pl-4 font-medium">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {latestRound.hospital_contributions.map((c) => (
                      <tr key={c.hospital_id} className="hover:bg-slate-50/70">
                        <td className="py-3 pr-4 font-medium text-slate-900">
                          {c.hospital_name}
                        </td>
                        <td className="py-3 px-4 text-right font-mono tabular-nums">
                          {c.num_samples}
                        </td>
                        <td className="py-3 px-4 text-right font-mono tabular-nums">
                          {formatPercent(c.weight_share)}
                        </td>
                        <td className="py-3 px-4 text-right font-mono tabular-nums">
                          {formatPercent(c.local_accuracy)}
                        </td>
                        <td className="py-3 px-4 text-right font-mono tabular-nums">
                          {c.delta_norm.toFixed(3)}
                        </td>
                        <td className="py-3 pl-4">
                          <StatusText
                            tone={c.accepted ? 'success' : 'alert'}
                            label={c.accepted ? 'Accepted' : 'Rejected'}
                          />
                        </td>
                      </tr>
                    ))}
                    <tr className="bg-slate-50 font-medium">
                      <td className="py-3 pr-4 text-slate-900">Global Aggregated Model</td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums">
                        {status?.total_train_samples ?? '—'}
                      </td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums">100.0%</td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums text-emerald-700">
                        {formatPercent(latestRound.global_accuracy)}
                      </td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums">—</td>
                      <td className="py-3 pl-4">
                        <StatusText tone="success" label="Aggregated" />
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </SectionCard>
          ) : null}

          <SectionCard title="Round History" subtitle="Saved in SQLite after every round.">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2.5 pr-4 font-medium">Round</th>
                    <th className="py-2.5 px-4 font-medium text-right">Accepted</th>
                    <th className="py-2.5 px-4 font-medium text-right">Accuracy</th>
                    <th className="py-2.5 px-4 font-medium text-right">Precision</th>
                    <th className="py-2.5 px-4 font-medium text-right">Recall</th>
                    <th className="py-2.5 px-4 font-medium text-right">F1</th>
                    <th className="py-2.5 px-4 font-medium text-right">ROC-AUC</th>
                    <th className="py-2.5 pl-4 font-medium text-right">Loss</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {rounds.map((r) => (
                    <tr key={r.round_number} className="hover:bg-slate-50/70">
                      <td className="py-2.5 pr-4 font-mono font-medium text-slate-900">
                        Round {r.round_number}
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums">
                        {r.accepted_updates}/3
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums">
                        {formatPercent(r.global_accuracy)}
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums">
                        {formatPercent(r.global_precision)}
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums">
                        {formatPercent(r.global_recall)}
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums">
                        {formatPercent(r.global_f1)}
                      </td>
                      <td className="py-2.5 px-4 text-right font-mono tabular-nums">
                        {r.global_roc_auc.toFixed(4)}
                      </td>
                      <td className="py-2.5 pl-4 text-right font-mono tabular-nums">
                        {r.global_loss.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      ) : (
        <EmptyState
          title="No federated rounds completed yet"
          message={
            status?.initialized
              ? 'Click Start Federated Training above to execute your first FedAvg round.'
              : 'Click Initialize Experiment first to prepare the three hospital nodes.'
          }
        />
      )}
    </div>
  );
}
