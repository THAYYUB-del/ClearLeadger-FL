import { useEffect, useState } from 'react';
import { Download } from 'lucide-react';
import { ModelComparisonChart, RoundPerformanceChart } from '../charts/PerformanceCharts';
import {
  EmptyState,
  ErrorBanner,
  MetricTile,
  SectionCard,
  formatPercent,
} from '../components/UiPrimitives';
import { api } from '../services/api';
import type { AnalyticsResponse, PageId } from '../types/api';

export function AnalyticsPage({
  refreshKey,
  onNavigate,
}: {
  refreshKey: number;
  onNavigate: (page: PageId) => void;
}) {
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [exportSuccess, setExportSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    api
      .getAnalytics()
      .then((res) => {
        if (active) {
          setData(res);
          setError(null);
        }
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : 'Failed to load analytics.');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [refreshKey]);

  const handleExportCsv = async () => {
    setExporting(true);
    setError(null);
    setExportSuccess(null);
    try {
      await api.downloadAnalyticsCsv();
      setExportSuccess('Exported clearledger-fl-analytics.csv');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'CSV export failed.');
    } finally {
      setExporting(false);
    }
  };

  const cm = data?.global_metrics?.confusion_matrix;

  const comparisonChartRows =
    data?.comparison
      .filter((row) => row.metrics !== null)
      .map((row) => ({
        name: row.model_name
          .replace(' (Pooled)', '')
          .replace(' (FedAvg)', '')
          .replace(' (Standalone Local)', ''),
        accuracy: Number((row.metrics!.accuracy * 100).toFixed(2)),
        roc_auc: Number((row.metrics!.roc_auc * 100).toFixed(2)),
        f1: Number((row.metrics!.f1_score * 100).toFixed(2)),
      })) ?? [];

  const roundChartRows =
    data?.rounds.map((r) => ({
      round: r.round_number,
      accuracy: Number((r.global_accuracy * 100).toFixed(2)),
      f1_score: Number((r.global_f1 * 100).toFixed(2)),
      roc_auc: Number((r.global_roc_auc * 100).toFixed(2)),
      loss: r.global_loss,
    })) ?? [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Model Analytics</h2>
          <p className="mt-0.5 text-sm text-slate-500">
            Held-out evaluation comparing centralized, standalone hospital, and federated models.
          </p>
        </div>

        <button
          type="button"
          onClick={handleExportCsv}
          disabled={exporting || !data?.evaluated}
          className="inline-flex items-center gap-2 h-9 px-4 text-sm font-medium text-white bg-teal-700 hover:bg-teal-800 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
        >
          <Download className="w-4 h-4 shrink-0" aria-hidden="true" />
          <span>{exporting ? 'Exporting...' : 'Export CSV'}</span>
        </button>
      </div>

      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      {exportSuccess ? (
        <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-sm text-emerald-800">
          {exportSuccess}
        </div>
      ) : null}

      {!loading && data && !data.evaluated ? (
        <EmptyState
          title="Not yet evaluated"
          message="Initialize the experiment and run at least one federated round to view real evaluation metrics."
          actionLabel="Go to Federated Learning"
          onAction={() => onNavigate('federated')}
        />
      ) : null}

      {data && data.evaluated && data.global_metrics ? (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
            <MetricTile
              label="Accuracy"
              value={formatPercent(data.global_metrics.accuracy)}
              tone="success"
            />
            <MetricTile label="Precision" value={formatPercent(data.global_metrics.precision)} />
            <MetricTile label="Recall" value={formatPercent(data.global_metrics.recall)} />
            <MetricTile label="F1 Score" value={formatPercent(data.global_metrics.f1_score)} />
            <MetricTile label="ROC-AUC" value={data.global_metrics.roc_auc.toFixed(4)} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Confusion Matrix */}
            {cm ? (
              <SectionCard
                title="Confusion Matrix"
                subtitle={`${data.dataset_info.held_out_test_samples} held-out test cases.`}
              >
                <div className="grid grid-cols-2 gap-3 text-center">
                  <div className="p-4 rounded-md bg-emerald-50 border border-emerald-200">
                    <div className="text-xs text-emerald-800 font-medium">True Benign (TP)</div>
                    <div className="mt-1 text-2xl font-semibold font-mono text-emerald-900 tabular-nums">
                      {cm.tp}
                    </div>
                  </div>
                  <div className="p-4 rounded-md bg-red-50 border border-red-200">
                    <div className="text-xs text-red-800 font-medium">False Benign (FP)</div>
                    <div className="mt-1 text-2xl font-semibold font-mono text-red-900 tabular-nums">
                      {cm.fp}
                    </div>
                  </div>
                  <div className="p-4 rounded-md bg-amber-50 border border-amber-200">
                    <div className="text-xs text-amber-800 font-medium">False Malignant (FN)</div>
                    <div className="mt-1 text-2xl font-semibold font-mono text-amber-900 tabular-nums">
                      {cm.fn}
                    </div>
                  </div>
                  <div className="p-4 rounded-md bg-slate-50 border border-slate-200">
                    <div className="text-xs text-slate-700 font-medium">True Malignant (TN)</div>
                    <div className="mt-1 text-2xl font-semibold font-mono text-slate-900 tabular-nums">
                      {cm.tn}
                    </div>
                  </div>
                </div>
              </SectionCard>
            ) : null}

            {/* Model Comparison Chart */}
            <div className="lg:col-span-2">
              <SectionCard
                title="Model Comparison"
                subtitle="Centralized baseline vs. Federated Global vs. standalone hospitals."
              >
                <ModelComparisonChart rows={comparisonChartRows} />
              </SectionCard>
            </div>
          </div>

          <SectionCard
            title="Comparison Table"
            subtitle={`Evaluated on the same ${data.dataset_info.held_out_test_samples} held-out records (${data.distribution === 'balanced' ? 'Balanced' : 'Non-IID'} split).`}
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500">
                    <th className="py-2.5 pr-4 font-medium">Model</th>
                    <th className="py-2.5 px-3 font-medium text-right">Accuracy</th>
                    <th className="py-2.5 px-3 font-medium text-right">Precision</th>
                    <th className="py-2.5 px-3 font-medium text-right">Recall</th>
                    <th className="py-2.5 px-3 font-medium text-right">F1</th>
                    <th className="py-2.5 px-3 font-medium text-right">ROC-AUC</th>
                    <th className="py-2.5 pl-3 font-medium text-right">Loss</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {data.comparison.map((row) => (
                    <tr
                      key={row.model_name}
                      className={
                        row.model_name.includes('Federated')
                          ? 'bg-teal-50/50 font-medium'
                          : 'hover:bg-slate-50/70'
                      }
                    >
                      <td className="py-3 pr-4">
                        <div className="text-slate-900 font-medium">{row.model_name}</div>
                        <div className="text-[11px] text-slate-500">{row.scope}</div>
                      </td>
                      <td className="py-3 px-3 text-right font-mono tabular-nums">
                        {formatPercent(row.metrics?.accuracy)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono tabular-nums">
                        {formatPercent(row.metrics?.precision)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono tabular-nums">
                        {formatPercent(row.metrics?.recall)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono tabular-nums">
                        {formatPercent(row.metrics?.f1_score)}
                      </td>
                      <td className="py-3 px-3 text-right font-mono tabular-nums">
                        {row.metrics ? row.metrics.roc_auc.toFixed(4) : '—'}
                      </td>
                      <td className="py-3 pl-3 text-right font-mono tabular-nums">
                        {row.metrics ? row.metrics.loss.toFixed(4) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>

          {roundChartRows.length > 0 ? (
            <SectionCard
              title="Round Performance Trend"
              subtitle="Global accuracy, ROC-AUC, and F1 score across rounds."
            >
              <RoundPerformanceChart data={roundChartRows} />
            </SectionCard>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
