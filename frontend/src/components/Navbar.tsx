import React from 'react';
import {
  Globe2,
  Cpu,
  Sparkles,
  BarChart3,
  Layers,
  FileSpreadsheet,
  Zap,
} from 'lucide-react';
import type { HealthResponse } from '../types';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  health: HealthResponse | null;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  health,
}) => {
  const navItems = [
    { id: 'landing', label: 'Overview', icon: Globe2 },
    { id: 'analyzer', label: 'Stance Analyzer', icon: Sparkles },
    { id: 'explainability', label: 'Explainability', icon: Zap },
    { id: 'batch', label: 'Batch Processing', icon: FileSpreadsheet },
    { id: 'analytics', label: 'Research Benchmarks', icon: BarChart3 },
    { id: 'architecture', label: 'Model Architecture', icon: Layers },
  ];

  return (
    <header className="sticky top-0 z-50 bg-[#090D10]/85 backdrop-blur-md border-b border-slate-800/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <div
            onClick={() => setActiveTab('landing')}
            className="flex items-center space-x-3 cursor-pointer group"
          >
            <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 group-hover:border-emerald-400 transition-colors shadow-sm shadow-emerald-500/10">
              <Globe2 className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-semibold text-slate-100 text-sm tracking-tight">
                  Indic Climate Stance
                </span>
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-emerald-950/80 border border-emerald-800/60 text-emerald-300 font-medium">
                  SLM
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-mono hidden sm:block">
                4-Layer MuRIL + LoRA (Distilled)
              </p>
            </div>
          </div>

          {/* Nav Items */}
          <nav className="hidden md:flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-slate-800 text-emerald-400 border border-slate-700/80 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-emerald-400' : 'text-slate-500'}`} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Model Status Indicator */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-slate-300 font-mono text-[11px] hidden sm:inline">
                {health ? (
                  health.device.toUpperCase() === 'CUDA' ? (
                    <span className="text-emerald-400 flex items-center gap-1">
                      <Cpu className="w-3 h-3 inline" /> CUDA GPU Active
                    </span>
                  ) : (
                    <span className="text-amber-400 flex items-center gap-1">
                      <Cpu className="w-3 h-3 inline" /> CPU Fallback
                    </span>
                  )
                ) : (
                  'Connecting...'
                )}
              </span>
              <span className="text-slate-500 font-mono text-[10px] hidden lg:inline">
                P50: 7.4ms
              </span>
            </div>
          </div>
        </div>

        {/* Mobile Submenu */}
        <div className="flex md:hidden overflow-x-auto py-2 space-x-1 border-t border-slate-800/60 no-scrollbar">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center space-x-1 px-2.5 py-1 rounded text-xs whitespace-nowrap ${
                  isActive
                    ? 'bg-slate-800 text-emerald-400 font-medium'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Icon className="w-3 h-3" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
