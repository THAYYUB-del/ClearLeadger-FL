import { useEffect, useState } from 'react';
import {
  EmptyState,
  ErrorBanner,
  MetricTile,
  SectionCard,
  StatusText,
  formatActorName,
  formatShortTime,
  shortHash,
} from '../components/UiPrimitives';
import { api } from '../services/api';
import type {
  LedgerResponse,
  LedgerVerificationReport,
  TamperTestReport,
} from '../types/api';

export function LedgerPage({
  refreshKey,
  onMutated,
}: {
  refreshKey: number;
  onMutated: () => void;
}) {
  const [ledger, setLedger] = useState<LedgerResponse | null>(null);
  const [verifyReport, setVerifyReport] = useState<LedgerVerificationReport | null>(null);
  const [tamperReport, setTamperReport] = useState<TamperTestReport | null>(null);
  const [busy, setBusy] = useState<'verify' | 'tamper' | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadLedger = async () => {
    try {
      const res = await api.getLedger();
      setLedger(res);
      setVerifyReport(res.verification);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load audit ledger.');
    }
  };

  useEffect(() => {
    loadLedger();
  }, [refreshKey]);

  const handleVerify = async () => {
    setBusy('verify');
    setError(null);
    try {
      const report = await api.verifyLedger();
      setVerifyReport(report);
      onMutated();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Verification failed.');
    } finally {
      setBusy(null);
    }
  };

  const handleTamperTest = async () => {
    setBusy('tamper');
    setError(null);
    try {
      const report = await api.runTamperTest();
      setTamperReport(report);
      await loadLedger();
      onMutated();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Tampering test failed.');
    } finally {
      setBusy(null);
    }
  };

  const records = ledger?.records ?? [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Hash-Linked Audit Ledger</h2>
          <p className="mt-0.5 text-sm text-slate-500">
            Local SHA-256 hash chain stored in SQLite for tamper-evident experiment logging.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={handleVerify}
            disabled={busy !== null}
            className="h-9 px-4 text-sm font-medium text-white bg-teal-700 hover:bg-teal-800 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
          >
            {busy === 'verify' ? 'Verifying...' : 'Verify Ledger'}
          </button>
          <button
            type="button"
            onClick={handleTamperTest}
            disabled={busy !== null || records.length === 0}
            className="h-9 px-4 text-sm font-medium text-amber-800 bg-white border border-amber-300 hover:bg-amber-50 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
          >
            {busy === 'tamper' ? 'Testing...' : 'Run Tampering Test'}
          </button>
        </div>
      </div>

      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <MetricTile
          label="Integrity Status"
          value={
            records.length === 0
              ? 'Empty'
              : verifyReport?.valid
                ? 'Verified'
                : 'Tampered'
          }
          tone={records.length === 0 ? 'default' : verifyReport?.valid ? 'success' : 'alert'}
          subtext={
            verifyReport
              ? `Checked ${verifyReport.checked_records} of ${verifyReport.total_records} records`
              : undefined
          }
        />
        <MetricTile
          label="Ledger Records"
          value={String(records.length)}
          subtext="Hash-linked SHA-256 chain"
        />
        <MetricTile
          label="Last Verified"
          value={verifyReport ? formatShortTime(verifyReport.verified_at) : '—'}
          subtext="Local SQLite audit log"
        />
      </div>

      {tamperReport ? (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-5 space-y-3 text-xs text-amber-950">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-semibold text-sm">
              Controlled Tampering Test — Tampering Detected at Record #{tamperReport.target_record_id}
            </span>
            <StatusText
              tone={tamperReport.live_ledger_valid ? 'success' : 'alert'}
              label={
                tamperReport.live_ledger_valid
                  ? 'Live ledger untouched & valid'
                  : 'Live ledger affected'
              }
            />
          </div>
          <p className="text-amber-900">
            {tamperReport.verification_after_tamper.failure_reason}
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1 font-mono text-[11px]">
            <div className="bg-white/80 border border-amber-200 rounded p-3">
              <div className="text-slate-500 font-sans mb-1">Stored Hash</div>
              <div className="break-all text-slate-800">{tamperReport.stored_hash}</div>
            </div>
            <div className="bg-white/80 border border-red-200 rounded p-3">
              <div className="text-red-700 font-sans mb-1">Recomputed Hash (Tampered Copy)</div>
              <div className="break-all text-red-800">{tamperReport.recomputed_hash}</div>
            </div>
          </div>
        </div>
      ) : null}

      {records.length === 0 ? (
        <EmptyState
          title="Audit ledger is empty"
          message="Initialize an experiment or run a federated round to generate SHA-256 hash-linked audit records."
        />
      ) : (
        <SectionCard
          title="Audit History"
          subtitle="Each record links to the previous record's SHA-256 hash."
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2.5 pr-3 font-medium">ID</th>
                  <th className="py-2.5 px-3 font-medium">Time</th>
                  <th className="py-2.5 px-3 font-medium">Hospital</th>
                  <th className="py-2.5 px-3 font-medium text-right">Round</th>
                  <th className="py-2.5 px-3 font-medium">Event</th>
                  <th className="py-2.5 px-3 font-medium">Previous Hash</th>
                  <th className="py-2.5 pl-3 font-medium">Current Hash</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {records.map((rec) => (
                  <tr key={rec.record_id} className="hover:bg-slate-50/70">
                    <td className="py-2.5 pr-3 font-mono font-medium text-slate-900 tabular-nums">
                      #{rec.record_id}
                    </td>
                    <td className="py-2.5 px-3 font-mono text-slate-500 tabular-nums whitespace-nowrap">
                      {formatShortTime(rec.timestamp)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-800 whitespace-nowrap">
                      {formatActorName(rec.hospital_id)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                      {rec.training_round}
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="font-medium text-slate-900">
                        {rec.event_type.replace(/_/g, ' ')}
                      </div>
                      <div className="text-[11px] text-slate-500 max-w-md truncate">
                        {rec.payload_summary}
                      </div>
                    </td>
                    <td
                      className="py-2.5 px-3 font-mono text-[11px] text-slate-500 whitespace-nowrap"
                      title={rec.previous_hash}
                    >
                      {shortHash(rec.previous_hash)}
                    </td>
                    <td
                      className="py-2.5 pl-3 font-mono text-[11px] text-slate-800 whitespace-nowrap"
                      title={rec.current_hash}
                    >
                      {shortHash(rec.current_hash)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}
    </div>
  );
}
