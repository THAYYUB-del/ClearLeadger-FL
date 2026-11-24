import { useEffect, useState } from 'react';
import { GlobalShapChart, IndividualShapChart } from '../charts/PerformanceCharts';
import {
  EmptyState,
  ErrorBanner,
  MetricTile,
  SectionCard,
  StatusText,
  formatPercent,
} from '../components/UiPrimitives';
import { api } from '../services/api';
import type {
  GlobalExplainabilityResponse,
  IndividualPredictionExplanation,
  PageId,
} from '../types/api';

export function ExplainabilityPage({
  refreshKey,
  onNavigate,
}: {
  refreshKey: number;
  onNavigate: (page: PageId) => void;
}) {
  const [globalData, setGlobalData] = useState<GlobalExplainabilityResponse | null>(null);
  const [selectedSample, setSelectedSample] = useState<number>(0);
  const [explanation, setExplanation] = useState<IndividualPredictionExplanation | null>(null);
  const [loadingGlobal, setLoadingGlobal] = useState(true);
  const [loadingSample, setLoadingSample] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadExplanation = async (sampleIdx: number) => {
    setLoadingSample(true);
    setError(null);
    try {
      const res = await api.explainPrediction(sampleIdx);
      setExplanation(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to compute SHAP explanation.');
    } finally {
      setLoadingSample(false);
    }
  };

  useEffect(() => {
    let active = true;
    setLoadingGlobal(true);
    api
      .getGlobalExplainability()
      .then((res) => {
        if (!active) return;
        setGlobalData(res);
        setError(null);
        if (res.available) {
          loadExplanation(selectedSample);
        } else {
          setExplanation(null);
        }
      })
      .catch((err) => {
        if (active) setError(err instanceof Error ? err.message : 'Failed to load SHAP data.');
      })
      .finally(() => {
        if (active) setLoadingGlobal(false);
      });
    return () => {
      active = false;
    };
  }, [refreshKey]);

  const handleExplainClick = () => {
    loadExplanation(selectedSample);
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Explainable AI (SHAP)</h2>
        <p className="mt-0.5 text-sm text-slate-500">
          See which diagnostic features drive the global model and each individual prediction.
        </p>
      </div>

      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      {!loadingGlobal && globalData && !globalData.available ? (
        <EmptyState
          title="Model not trained yet"
          message="Run at least one federated training round to compute SHAP feature importance and prediction explanations."
          actionLabel="Go to Federated Learning"
          onAction={() => onNavigate('federated')}
        />
      ) : null}

      {globalData && globalData.available ? (
        <>
          <SectionCard
            title="Global Feature Importance (Mean |SHAP|)"
            subtitle={`Top 12 features across ${globalData.test_sample_count} held-out cases · Teal = Benign-leaning, Crimson = Malignant-leaning.`}
          >
            <GlobalShapChart features={globalData.features} />
          </SectionCard>

          <SectionCard
            title="Individual Case Explanation"
            subtitle="Inspect how each feature pushed a single held-out test prediction."
            action={
              <div className="flex items-center gap-2">
                <label htmlFor="sample-select" className="text-xs text-slate-500">
                  Test Case
                </label>
                <select
                  id="sample-select"
                  value={selectedSample}
                  onChange={(e) => setSelectedSample(Number(e.target.value))}
                  className="h-8 px-2.5 text-xs bg-white border border-slate-300 rounded-md text-slate-900"
                >
                  {globalData.sample_preview.map((s) => (
                    <option key={s.sample_index} value={s.sample_index}>
                      Case #{s.sample_index} ({s.actual_class})
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={handleExplainClick}
                  disabled={loadingSample}
                  className="h-8 px-3.5 text-xs font-medium text-white bg-teal-700 hover:bg-teal-800 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
                >
                  {loadingSample ? 'Computing...' : 'Explain Prediction'}
                </button>
              </div>
            }
          >
            {explanation ? (
              <div className="space-y-6">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <MetricTile
                    label="Predicted Class"
                    value={explanation.predicted_class}
                    subtext={
                      explanation.predicted_class === explanation.actual_class
                        ? `Matches actual (${explanation.actual_class})`
                        : `Actual: ${explanation.actual_class}`
                    }
                    tone={
                      explanation.predicted_class === explanation.actual_class
                        ? 'default'
                        : 'alert'
                    }
                  />
                  <MetricTile
                    label="Benign Probability"
                    value={formatPercent(explanation.probability_benign)}
                    subtext={`Malignant: ${formatPercent(explanation.probability_malignant)}`}
                  />
                  <MetricTile
                    label="Base Value (Logit)"
                    value={explanation.base_value_logit.toFixed(3)}
                    subtext="Training dataset baseline"
                  />
                  <MetricTile
                    label="Final Logit (Base + ΣSHAP)"
                    value={explanation.predicted_logit.toFixed(3)}
                    subtext={`Additivity check: ${explanation.shap_sum_check.toFixed(3)}`}
                  />
                </div>

                <div>
                  <h3 className="text-xs font-semibold text-slate-700 mb-3">
                    Top Feature Contributions (Case #{explanation.sample_index})
                  </h3>
                  <IndividualShapChart contributions={explanation.contributions} />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
                  <div>
                    <h4 className="text-xs font-semibold text-emerald-800 mb-2">
                      Positive Contributions (Toward Benign)
                    </h4>
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-500">
                          <th className="py-2 pr-2 font-medium">Feature</th>
                          <th className="py-2 px-2 font-medium text-right">Value</th>
                          <th className="py-2 pl-2 font-medium text-right">SHAP</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {explanation.top_positive.slice(0, 6).map((c) => (
                          <tr key={c.feature_name}>
                            <td className="py-2 pr-2 text-slate-800">{c.feature_name}</td>
                            <td className="py-2 px-2 text-right font-mono text-slate-500 tabular-nums">
                              {c.raw_value}
                            </td>
                            <td className="py-2 pl-2 text-right font-mono font-medium text-emerald-700 tabular-nums">
                              +{c.shap_value.toFixed(4)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div>
                    <h4 className="text-xs font-semibold text-red-800 mb-2">
                      Negative Contributions (Toward Malignant)
                    </h4>
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-500">
                          <th className="py-2 pr-2 font-medium">Feature</th>
                          <th className="py-2 px-2 font-medium text-right">Value</th>
                          <th className="py-2 pl-2 font-medium text-right">SHAP</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {explanation.top_negative.slice(0, 6).map((c) => (
                          <tr key={c.feature_name}>
                            <td className="py-2 pr-2 text-slate-800">{c.feature_name}</td>
                            <td className="py-2 px-2 text-right font-mono text-slate-500 tabular-nums">
                              {c.raw_value}
                            </td>
                            <td className="py-2 pl-2 text-right font-mono font-medium text-red-700 tabular-nums">
                              {c.shap_value.toFixed(4)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                  <span>Educational SHAP analysis · Wisconsin Breast Cancer dataset</span>
                  <StatusText tone="success" label="Exact Linear SHAP verified" />
                </div>
              </div>
            ) : null}
          </SectionCard>
        </>
      ) : null}
    </div>
  );
}
