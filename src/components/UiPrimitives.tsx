import type { ReactNode } from 'react';
import { AlertCircle, AlertTriangle, CheckCircle2, Info } from 'lucide-react';

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

export function formatShortTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function shortHash(hash: string): string {
  if (!hash || hash.length <= 16) return hash;
  return `${hash.slice(0, 10)}…${hash.slice(-6)}`;
}

export function formatActorName(id: string): string {
  if (id === 'hospital_a') return 'Hospital A';
  if (id === 'hospital_b') return 'Hospital B';
  if (id === 'hospital_c') return 'Hospital C';
  if (id === 'aggregator') return 'Aggregator';
  return id.replace(/_/g, ' ');
}

export function MetricTile({
  label,
  value,
  subtext,
  tone = 'default',
}: {
  label: string;
  value: string;
  subtext?: string;
  tone?: 'default' | 'success' | 'warning' | 'alert';
}) {
  const valueTone =
    tone === 'success'
      ? 'text-emerald-700'
      : tone === 'warning'
        ? 'text-amber-700'
        : tone === 'alert'
          ? 'text-red-700'
          : 'text-slate-900';

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5">
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p className={`mt-2 text-2xl font-semibold tracking-tight tabular-nums ${valueTone}`}>
        {value}
      </p>
      {subtext ? <p className="mt-1.5 text-xs text-slate-500">{subtext}</p> : null}
    </div>
  );
}

export function StatusText({
  tone,
  label,
}: {
  tone: 'success' | 'warning' | 'alert' | 'neutral';
  label: string;
}) {
  if (tone === 'success') {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-700 whitespace-nowrap">
        <CheckCircle2 className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
        <span>{label}</span>
      </span>
    );
  }
  if (tone === 'warning') {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-medium text-amber-700 whitespace-nowrap">
        <AlertTriangle className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
        <span>{label}</span>
      </span>
    );
  }
  if (tone === 'alert') {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-medium text-red-700 whitespace-nowrap">
        <AlertCircle className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
        <span>{label}</span>
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 whitespace-nowrap">
      <Info className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
      <span>{label}</span>
    </span>
  );
}

export function EmptyState({
  title,
  message,
  actionLabel,
  onAction,
  disabled,
}: {
  title: string;
  message: string;
  actionLabel?: string;
  onAction?: () => void;
  disabled?: boolean;
}) {
  return (
    <div className="bg-white border border-slate-200 rounded-lg px-6 py-10 text-center">
      <h3 className="text-base font-semibold text-slate-900">{title}</h3>
      <p className="mt-1.5 text-sm text-slate-500 max-w-md mx-auto">{message}</p>
      {actionLabel && onAction ? (
        <div className="mt-5">
          <button
            type="button"
            onClick={onAction}
            disabled={disabled}
            className="px-4 py-2 text-sm font-medium text-white bg-teal-700 hover:bg-teal-800 disabled:opacity-50 rounded-md transition-colors whitespace-nowrap"
          >
            {actionLabel}
          </button>
        </div>
      ) : null}
    </div>
  );
}

export function ErrorBanner({
  message,
  onDismiss,
}: {
  message: string | null;
  onDismiss?: () => void;
}) {
  if (!message) return null;
  return (
    <div
      role="alert"
      className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 flex items-center justify-between gap-4 text-sm text-red-800"
    >
      <div className="flex items-center gap-2">
        <AlertCircle className="w-4 h-4 shrink-0 text-red-700" aria-hidden="true" />
        <span>{message}</span>
      </div>
      {onDismiss ? (
        <button
          type="button"
          onClick={onDismiss}
          className="text-xs font-medium text-red-700 hover:text-red-900 whitespace-nowrap"
        >
          Dismiss
        </button>
      ) : null}
    </div>
  );
}

export function SectionCard({
  title,
  subtitle,
  action,
  children,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="bg-white border border-slate-200 rounded-lg">
      <div className="px-6 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
          {subtitle ? <p className="mt-0.5 text-xs text-slate-500">{subtitle}</p> : null}
        </div>
        {action ? <div className="shrink-0">{action}</div> : null}
      </div>
      <div className="p-6">{children}</div>
    </section>
  );
}
