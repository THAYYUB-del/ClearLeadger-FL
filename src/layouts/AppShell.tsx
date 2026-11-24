import { useState, type ReactNode } from 'react';
import {
  Activity,
  BarChart3,
  Building2,
  GitMerge,
  Link2,
  Menu,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react';
import type { PageId } from '../types/api';

interface NavItem {
  id: PageId;
  label: string;
  icon: typeof Activity;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'overview', label: 'Overview', icon: Activity },
  { id: 'hospitals', label: 'Hospital Network', icon: Building2 },
  { id: 'federated', label: 'Federated Learning', icon: GitMerge },
  { id: 'security', label: 'Security Center', icon: ShieldCheck },
  { id: 'ledger', label: 'Audit Ledger', icon: Link2 },
  { id: 'explainability', label: 'Explainable AI', icon: Sparkles },
  { id: 'analytics', label: 'Model Analytics', icon: BarChart3 },
];

const PAGE_TITLES: Record<PageId, string> = {
  overview: 'Overview',
  hospitals: 'Hospital Network',
  federated: 'Federated Learning',
  security: 'Security Center',
  ledger: 'Audit Ledger',
  explainability: 'Explainable AI',
  analytics: 'Model Analytics',
};

export function AppShell({
  activePage,
  onNavigate,
  topAction,
  children,
}: {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
  topAction?: ReactNode;
  children: ReactNode;
}) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleSelect = (page: PageId) => {
    onNavigate(page);
    setMobileMenuOpen(false);
  };

  return (
    <div className="min-h-screen flex bg-[#F8FAFC] text-slate-900">
      {/* Desktop Deep Navy Sidebar */}
      <aside className="hidden lg:flex lg:flex-col lg:w-60 lg:fixed lg:inset-y-0 bg-[#0F172A] text-slate-200 border-r border-slate-800">
        <div className="h-16 px-5 flex items-center border-b border-slate-800/80">
          <span className="text-base font-semibold tracking-tight text-white whitespace-nowrap">
            ClearLedger-FL
          </span>
        </div>

        <nav aria-label="Main navigation" className="flex-1 px-3 py-4 space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activePage === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => handleSelect(item.id)}
                aria-current={active ? 'page' : undefined}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
                  active
                    ? 'bg-teal-800/40 text-white'
                    : 'text-slate-300 hover:bg-slate-800/70 hover:text-white'
                }`}
              >
                <Icon
                  className={`w-4 h-4 shrink-0 ${active ? 'text-teal-300' : 'text-slate-400'}`}
                  aria-hidden="true"
                />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="px-5 py-4 border-t border-slate-800/80 text-xs text-slate-400">
          Educational ML simulation · Not for clinical use
        </div>
      </aside>

      {/* Mobile Drawer */}
      {mobileMenuOpen ? (
        <div className="fixed inset-0 z-40 lg:hidden flex">
          <div
            className="fixed inset-0 bg-slate-900/50"
            onClick={() => setMobileMenuOpen(false)}
            aria-hidden="true"
          />
          <aside className="relative z-50 w-64 max-w-[80vw] bg-[#0F172A] text-slate-200 flex flex-col">
            <div className="h-14 px-5 flex items-center justify-between border-b border-slate-800">
              <span className="text-base font-semibold text-white">ClearLedger-FL</span>
              <button
                type="button"
                onClick={() => setMobileMenuOpen(false)}
                aria-label="Close navigation menu"
                className="p-1.5 text-slate-400 hover:text-white rounded-md"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <nav aria-label="Mobile navigation" className="flex-1 px-3 py-4 space-y-1">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                const active = activePage === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => handleSelect(item.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors ${
                      active
                        ? 'bg-teal-800/40 text-white'
                        : 'text-slate-300 hover:bg-slate-800/70 hover:text-white'
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" aria-hidden="true" />
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </aside>
        </div>
      ) : null}

      {/* Main Content Workspace */}
      <div className="flex-1 lg:pl-60 flex flex-col min-w-0">
        <header className="sticky top-0 z-20 h-14 lg:h-16 bg-white border-b border-slate-200 px-4 lg:px-8 flex items-center justify-between gap-6">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setMobileMenuOpen(true)}
              aria-label="Open navigation menu"
              className="lg:hidden p-2 -ml-2 text-slate-600 hover:text-slate-900 rounded-md"
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-2 text-sm min-w-0">
              <span className="hidden sm:inline text-slate-400 whitespace-nowrap">
                ClearLedger-FL
              </span>
              <span className="hidden sm:inline text-slate-300" aria-hidden="true">
                /
              </span>
              <h1 className="text-base font-semibold text-slate-900 truncate">
                {PAGE_TITLES[activePage]}
              </h1>
            </div>
          </div>

          <div className="shrink-0">{topAction}</div>
        </header>

        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-[1280px] w-full mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
