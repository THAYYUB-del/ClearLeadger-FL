import { useEffect, useState } from 'react';
import { RoundPerformanceChart } from '../charts/PerformanceCharts';
import {
  EmptyState,
  ErrorBanner,
  MetricTile,
  SectionCard,
  StatusText,
  formatPercent,
  formatShortTime,
} from '../components/UiPrimitives';
import { api } from '../services/api';
import type { DashboardSummary, PageId } from '../types/api';

export function OverviewPage({
  onNavigate,
  refreshKey,
  onQuickInitialize,
}: {
  onNavigate: (page: PageId) => void;
  refreshKey: number;
  onQuickInitialize: () => Promise<void>;
}) {
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    api
      .getDashboard()
      .then((res) => {
        if (active) {
          setData(res);
          setError(null);
        }
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : 'Failed to load dashboard.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [refreshKey]);

  const handleInit = async () => {
    setBusy(true);
    setError(null);
    try {
      await onQuickInitialize();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Initialization failed.');
    } finally {
      setBusy(false);
    }
  };

  if (loading && !data) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-28 bg-white border border-slate-200 rounded-lg p-5" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Federated Network Summary</h2>
          <p className="mt-0.5 text-sm text-slate-500">
            Three hospitals train a shared model without exchanging patient records.
          </p>
        </div>
        {data ? (
          <div className="text-xs text-slate-500">
            Status: <span className="font-medium text-slate-800">{data.federated_status}</span>
          </div>
        ) : null}
      </div>

      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      {data ? (
        <>
          {/* Primary KPI Row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricTile
              label="Connected Hospitals"
              value={`${data.connected_hospitals} / ${data.total_hospitals}`}
              subtext={data.experiment_initialized ? 'All sites active' : 'Ready to initialize'}
            />
            <MetricTile
              label="Current Round"
              value={String(data.current_round)}
              subtext={
                data.experiment_initialized
                  ? `Split: ${data.distribution === 'balanced' ? 'Balanced' : 'Non-IID'}`
                  : 'No experiment started'
              }
            />
            <MetricTile
              label="Global Model Accuracy"
              value={data.global_accuracy !== null ? formatPercent(data.global_accuracy) : 'Not evaluated'}
              subtext={
                data.global_roc_auc !== null
                  ? `ROC-AUC ${data.global_roc_auc.toFixed(3)}`
                  : 'Run Round 1 to evaluate'
              }
              tone={data.global_accuracy !== null ? 'success' : 'default'}
            />
            <MetricTile
              label="Ledger Integrity"
              value={data.ledger_records_count === 0 ? 'Empty' : data.ledger_integrity.valid ? 'Verified' : 'Tampered'}
              subtext={`${data.ledger_records_count} audit ${data.ledger_records_count === 1 ? 'record' : 'records'}`}
              tone={
                data.ledger_records_count === 0
                  ? 'default'
                  : data.ledger_integrity.valid
                    ? 'success'
                    : 'alert'
              }
            />
          </div>

          {/* Secondary Summary Strip */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <MetricTile
              label="Authenticated Updates"
              value={String(data.total_authenticated_updates)}
              subtext="Verified with HMAC-SHA256"
            />
            <MetricTile
              label="Security Alerts"
              value={String(data.security_alerts)}
              subtext={data.security_alerts > 0 ? 'Rejected updates logged' : 'No rejected updates'}
              tone={data.security_alerts > 0 ? 'alert' : 'default'}
            />
          </div>

          {/* Main Chart + Activity Split */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              {data.performance_series.length > 0 ? (
                <SectionCard
                  title="Training Performance"
                  subtitle="Global model metrics across federated rounds."
                  action={
                    <button
                      type="button"
                      onClick={() => onNavigate('federated')}
                      className="text-xs font-medium text-teal-700 hover:text-teal-900 whitespace-nowrap"
                    >
                      Run more rounds
                    </button>
                  }
                >
                  <RoundPerformanceChart data={data.performance_series} />
                </SectionCard>
              ) : (
                <EmptyState
                  title={
                    data.experiment_initialized
                      ? 'Ready to start training'
                      : 'No experiment running yet'
                  }
                  message={
                    data.experiment_initialized
                      ? 'Start Round 1 in Federated Learning to view global model performance.'
                      : 'Initialize the dataset split across Hospital A, B, and C to begin.'
                  }
                  actionLabel={
                    data.experiment_initialized
                      ? 'Go to Federated Learning'
                      : busy
                        ? 'Initializing...'
                        : 'Initialize Experiment'
                  }
                  onAction={
                    data.experiment_initialized ? () => onNavigate('federated') : handleInit
                  }
                  disabled={busy}
                />
              )}
            </div>

            <SectionCard title="Recent Activity" subtitle="Latest backend events.">
              {data.recent_activity.length === 0 ? (
                <p className="text-sm text-slate-500">
                  No activity recorded yet. Initialize an experiment to begin logging.
                </p>
              ) : (
                <ul className="divide-y divide-slate-100 -my-2">
                  {data.recent_activity.slice(0, 7).map((item) => (
                    <li key={item.id} className="py-3">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-medium text-slate-900 truncate">
                          {item.title}
                        </span>
                        <span className="text-[11px] font-mono text-slate-400 shrink-0 tabular-nums">
                          {formatShortTime(item.timestamp)}
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-slate-500 line-clamp-2">{item.detail}</p>
                      <div className="mt-1">
                        <StatusText
                          tone={
                            item.severity === 'alert'
                              ? 'alert'
                              : item.severity === 'warning'
                                ? 'warning'
                                : item.severity === 'success'
                                  ? 'success'
                                  : 'neutral'
                          }
                          label={item.category}
                        />
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </SectionCard>
          </div>
        </>
      ) : null}
    </div>
  );
}
