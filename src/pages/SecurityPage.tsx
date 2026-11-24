import { useEffect, useState } from 'react';
import {
  ErrorBanner,
  MetricTile,
  SectionCard,
  StatusText,
  formatActorName,
  formatShortTime,
} from '../components/UiPrimitives';
import { api } from '../services/api';
import type {
  HospitalId,
  SecurityEventsResponse,
  SecurityScenario,
  SecuritySimulateResponse,
} from '../types/api';

const SCENARIOS: Array<{
  id: SecurityScenario;
  label: string;
  hint: string;
}> = [
  { id: 'valid_update', label: 'Valid Update', hint: 'Valid key + honest weights' },
  { id: 'invalid_hmac', label: 'Invalid HMAC', hint: 'Corrupted signature' },
  { id: 'forged_sender', label: 'Forged Sender', hint: 'Wrong hospital key' },
  { id: 'exaggerated_update', label: 'Exaggerated Update', hint: 'Valid key + 18x weights' },
  { id: 'label_flipping', label: 'Label Flipping', hint: 'Valid key + inverted labels' },
];

export function SecurityPage({
  refreshKey,
  onMutated,
}: {
  refreshKey: number;
  onMutated: () => void;
}) {
  const [data, setData] = useState<SecurityEventsResponse | null>(null);
  const [targetHospital, setTargetHospital] = useState<HospitalId>('hospital_b');
  const [runningScenario, setRunningScenario] = useState<SecurityScenario | null>(null);
  const [latestResult, setLatestResult] = useState<SecuritySimulateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadEvents = async () => {
    try {
      const res = await api.getSecurityEvents();
      setData(res);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load security events.');
    }
  };

  useEffect(() => {
    loadEvents();
  }, [refreshKey]);

  const runScenario = async (scenario: SecurityScenario) => {
    setRunningScenario(scenario);
    setError(null);
    try {
      const res = await api.simulateSecurity(scenario, targetHospital);
      setLatestResult(res);
      await loadEvents();
      onMutated();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Security experiment failed.');
    } finally {
      setRunningScenario(null);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Security Center</h2>
        <p className="mt-0.5 text-sm text-slate-500">
          HMAC-SHA256 proves who sent an update. Update validation checks whether the weights are safe.
        </p>
      </div>

      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      {data ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricTile label="Total Events" value={String(data.summary.total_events)} />
          <MetricTile
            label="Accepted Updates"
            value={String(data.summary.accepted)}
            tone="success"
          />
          <MetricTile
            label="Failed HMAC Auth"
            value={String(data.summary.rejected_authentication)}
            subtext="Invalid key or tampered payload"
            tone={data.summary.rejected_authentication > 0 ? 'alert' : 'default'}
          />
          <MetricTile
            label="Blocked Poisoning"
            value={String(data.summary.rejected_malicious_validation)}
            subtext="Valid HMAC, unsafe weights"
            tone={data.summary.rejected_malicious_validation > 0 ? 'warning' : 'default'}
          />
        </div>
      ) : null}

      <SectionCard
        title="Run Security Test"
        subtitle="Test authentication and malicious-update rejection without affecting the global model."
        action={
          <div className="flex items-center gap-2">
            <label htmlFor="security-hospital" className="text-xs text-slate-500">
              Target
            </label>
            <select
              id="security-hospital"
              value={targetHospital}
              onChange={(e) => setTargetHospital(e.target.value as HospitalId)}
              className="h-8 px-2.5 text-xs bg-white border border-slate-300 rounded-md text-slate-900"
            >
              <option value="hospital_a">Hospital A</option>
              <option value="hospital_b">Hospital B</option>
              <option value="hospital_c">Hospital C</option>
            </select>
          </div>
        }
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {SCENARIOS.map((s) => {
            const busy = runningScenario === s.id;
            return (
              <button
                key={s.id}
                type="button"
                onClick={() => runScenario(s.id)}
                disabled={runningScenario !== null}
                className="text-left p-3.5 rounded-md border border-slate-200 hover:border-teal-700 hover:bg-slate-50 disabled:opacity-50 transition-colors"
              >
                <div className="text-xs font-semibold text-slate-900">
                  {busy ? 'Running...' : s.label}
                </div>
                <div className="mt-1 text-[11px] text-slate-500">{s.hint}</div>
              </button>
            );
          })}
        </div>

        {latestResult ? (
          <div
            className={`mt-5 p-4 rounded-md border text-xs ${
              latestResult.event.decision === 'ACCEPTED'
                ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
                : 'bg-red-50 border-red-200 text-red-900'
            }`}
          >
            <div className="flex flex-wrap items-center justify-between gap-2 font-semibold">
              <span>
                Result: {latestResult.event.decision} ({latestResult.event.scenario.replace(/_/g, ' ')})
              </span>
              <span className="font-mono tabular-nums">
                HMAC: {latestResult.event.hmac_valid ? 'Valid' : 'Invalid'} · Validation:{' '}
                {latestResult.event.validation_passed ? 'Passed' : 'Failed'} · ||Δw||{' '}
                {latestResult.event.delta_norm.toFixed(3)}
              </span>
            </div>
            <p className="mt-1.5">{latestResult.explanation}</p>
          </div>
        ) : null}
      </SectionCard>

      <SectionCard title="Security Event History" subtitle="Every update checked by the aggregator.">
        {!data || data.events.length === 0 ? (
          <p className="text-sm text-slate-500">
            No security events recorded yet. Run a federated round or click a test scenario above.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2.5 pr-3 font-medium">Time</th>
                  <th className="py-2.5 px-3 font-medium">Hospital</th>
                  <th className="py-2.5 px-3 font-medium">Scenario</th>
                  <th className="py-2.5 px-3 font-medium">HMAC</th>
                  <th className="py-2.5 px-3 font-medium">Validation</th>
                  <th className="py-2.5 px-3 font-medium text-right">Norm</th>
                  <th className="py-2.5 pl-3 font-medium">Decision</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {data.events.map((ev) => (
                  <tr key={ev.id} className="hover:bg-slate-50/70">
                    <td className="py-2.5 pr-3 font-mono text-slate-500 tabular-nums whitespace-nowrap">
                      {formatShortTime(ev.timestamp)}
                    </td>
                    <td className="py-2.5 px-3 font-medium text-slate-900 whitespace-nowrap">
                      {formatActorName(ev.hospital_id)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-700 whitespace-nowrap">
                      {ev.scenario.replace(/_/g, ' ')}
                    </td>
                    <td className="py-2.5 px-3">
                      <StatusText
                        tone={ev.hmac_valid ? 'success' : 'alert'}
                        label={ev.hmac_valid ? 'Valid' : 'Invalid'}
                      />
                    </td>
                    <td className="py-2.5 px-3">
                      <StatusText
                        tone={ev.validation_passed ? 'success' : ev.hmac_valid ? 'warning' : 'neutral'}
                        label={
                          ev.validation_passed
                            ? 'Passed'
                            : ev.hmac_valid
                              ? 'Quarantined'
                              : 'Skipped'
                        }
                      />
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                      {ev.delta_norm.toFixed(3)}
                    </td>
                    <td className="py-2.5 pl-3">
                      <StatusText
                        tone={ev.decision === 'ACCEPTED' ? 'success' : 'alert'}
                        label={ev.decision}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </SectionCard>
    </div>
  );
}
