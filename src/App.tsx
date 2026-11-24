import { useCallback, useState } from 'react';
import { AppShell } from './layouts/AppShell';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { ExplainabilityPage } from './pages/ExplainabilityPage';
import { FederatedPage } from './pages/FederatedPage';
import { HospitalsPage } from './pages/HospitalsPage';
import { LedgerPage } from './pages/LedgerPage';
import { OverviewPage } from './pages/OverviewPage';
import { SecurityPage } from './pages/SecurityPage';
import { api } from './services/api';
import type { PageId } from './types/api';

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('overview');
  const [refreshKey, setRefreshKey] = useState(0);

  const triggerRefresh = useCallback(() => {
    setRefreshKey((k) => k + 1);
  }, []);

  const handleQuickInitialize = useCallback(async () => {
    await api.initializeExperiment({ distribution: 'balanced', seed: 42 });
    triggerRefresh();
  }, [triggerRefresh]);

  const topAction =
    activePage === 'overview' ? (
      <button
        type="button"
        onClick={() => setActivePage('federated')}
        className="h-9 px-4 text-xs font-medium text-white bg-teal-700 hover:bg-teal-800 rounded-md transition-colors whitespace-nowrap shrink-0"
      >
        Open Training
      </button>
    ) : null;

  return (
    <AppShell activePage={activePage} onNavigate={setActivePage} topAction={topAction}>
      {activePage === 'overview' ? (
        <OverviewPage
          onNavigate={setActivePage}
          refreshKey={refreshKey}
          onQuickInitialize={handleQuickInitialize}
        />
      ) : null}

      {activePage === 'hospitals' ? (
        <HospitalsPage
          refreshKey={refreshKey}
          onMutated={triggerRefresh}
          onQuickInitialize={handleQuickInitialize}
        />
      ) : null}

      {activePage === 'federated' ? (
        <FederatedPage refreshKey={refreshKey} onMutated={triggerRefresh} />
      ) : null}

      {activePage === 'security' ? (
        <SecurityPage refreshKey={refreshKey} onMutated={triggerRefresh} />
      ) : null}

      {activePage === 'ledger' ? (
        <LedgerPage refreshKey={refreshKey} onMutated={triggerRefresh} />
      ) : null}

      {activePage === 'explainability' ? (
        <ExplainabilityPage refreshKey={refreshKey} onNavigate={setActivePage} />
      ) : null}

      {activePage === 'analytics' ? (
        <AnalyticsPage refreshKey={refreshKey} onNavigate={setActivePage} />
      ) : null}
    </AppShell>
  );
}
