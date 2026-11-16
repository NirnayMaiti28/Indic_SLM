import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { LandingView } from './components/LandingView';
import { AnalyzerView } from './components/AnalyzerView';
import { ExplainabilityView } from './components/ExplainabilityView';
import { BatchView } from './components/BatchView';
import { AnalyticsView } from './components/AnalyticsView';
import { ModelInfoView } from './components/ModelInfoView';
import type { HealthResponse, PredictResponse } from './types';
import { fetchHealth } from './api';
import { Globe2 } from 'lucide-react';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('landing');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [lastAnalysis, setLastAnalysis] = useState<PredictResponse | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((err) => {
        console.warn('Backend not yet reachable on startup:', err);
      });

    // Recheck health periodically
    const interval = setInterval(() => {
      fetchHealth().then(setHealth).catch(() => {});
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  const handleInspectExplainability = (res: PredictResponse) => {
    setLastAnalysis(res);
    setActiveTab('explainability');
  };

  return (
    <div className="min-h-screen bg-[#090D10] text-slate-100 flex flex-col font-sans selection:bg-emerald-500/30 selection:text-emerald-300">
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'landing' && (
          <LandingView
            onLaunchAnalyzer={() => setActiveTab('analyzer')}
            onExploreArchitecture={() => setActiveTab('architecture')}
            onViewBenchmarks={() => setActiveTab('analytics')}
          />
        )}

        {activeTab === 'analyzer' && (
          <AnalyzerView
            onInspectExplainability={handleInspectExplainability}
          />
        )}

        {activeTab === 'explainability' && (
          <ExplainabilityView lastResult={lastAnalysis} />
        )}

        {activeTab === 'batch' && <BatchView />}

        {activeTab === 'analytics' && <AnalyticsView />}

        {activeTab === 'architecture' && <ModelInfoView />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-[#070A0D] py-6 text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2">
            <Globe2 className="w-4 h-4 text-emerald-500" />
            <span className="text-slate-400">
              Indic Climate Stance Detection Research Platform
            </span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-500">v0.1.0</span>
          </div>

          <div className="flex items-center space-x-4">
            <span className="text-slate-400">
              Compact Student SLM (4-Layer MuRIL + LoRA + KD)
            </span>
            <span className="text-slate-600">|</span>
            <span className="text-emerald-400/80">Macro-F1: 0.9574</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
