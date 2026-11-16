import React, { useState } from 'react';
import {
  Zap,
  Info,
  HelpCircle,
  Sparkles,
  ArrowRight,
  TrendingUp,
  TrendingDown,
  Layers,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ReferenceLine,
} from 'recharts';
import type { PredictResponse, TokenAttribution } from '../types';
import { predictStance } from '../api';

interface ExplainabilityViewProps {
  lastResult: PredictResponse | null;
}

export const ExplainabilityView: React.FC<ExplainabilityViewProps> = ({
  lastResult,
}) => {
  const [explainText, setExplainText] = useState(
    lastResult?.text ||
      'ग्लोबल वार्मिंग कोई गंभीर समस्या नहीं है, यह सिर्फ पश्चिमी देशों का प्रोपेगैंडा है।'
  );
  const [loading, setLoading] = useState(false);
  const [activeResult, setActiveResult] = useState<PredictResponse | null>(
    lastResult
  );
  const [error, setError] = useState<string | null>(null);

  const handleRunExplain = async () => {
    if (!explainText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await predictStance(explainText, true);
      setActiveResult(res);
    } catch (err: any) {
      setError(err.message || 'Failed to compute attributions.');
    } finally {
      setLoading(false);
    }
  };

  const attributions: TokenAttribution[] = activeResult?.explanations || [];

  const maxAbsAttr = Math.max(
    ...attributions.map((a) => Math.abs(a.attribution)),
    1e-5
  );

  const chartData = attributions.map((a) => ({
    token: a.token,
    score: a.attribution,
    absScore: Math.abs(a.attribution),
    color: a.attribution >= 0 ? '#10B981' : '#F43F5E',
  }));

  return (
    <div className="space-y-8 py-2">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <Zap className="w-6 h-6 text-amber-400" />
          <span>Model Explainability: Integrated Gradients</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Visualizing axiomatic token-level attribution scores along the embedding path between a baseline embedding and input tokens.
        </p>
      </div>

      {/* Input control */}
      <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-4">
        <label className="text-xs font-mono uppercase text-slate-400 block tracking-wider">
          Input Comment to Explain
        </label>
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            value={explainText}
            onChange={(e) => setExplainText(e.target.value)}
            placeholder="Enter text to compute token attributions..."
            className="flex-1 bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/50 font-sans"
          />
          <button
            onClick={handleRunExplain}
            disabled={loading}
            className="px-6 py-2.5 rounded-xl bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-slate-950 font-semibold text-sm transition-all shadow-md shadow-amber-500/10 flex items-center justify-center gap-2 cursor-pointer font-sans shrink-0"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                <span>Computing Gradients...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                <span>Compute Attributions</span>
              </>
            )}
          </button>
        </div>
        {error && (
          <p className="text-xs text-rose-400">{error}</p>
        )}
      </div>

      {/* Token Attribution Results */}
      {activeResult && activeResult.predicted_label && (
        <div className="space-y-6">
          {/* Highlighted Token Preview Box */}
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 block">
                  Token Heatmap for Predicted Class:{' '}
                  <span className="font-bold text-white">
                    {activeResult.predicted_label}
                  </span>{' '}
                  ({(activeResult.confidence! * 100).toFixed(1)}% Confidence)
                </span>
              </div>
              <div className="flex items-center gap-4 text-xs font-mono">
                <span className="flex items-center gap-1.5 text-emerald-400">
                  <span className="w-3 h-3 rounded bg-emerald-500/30 border border-emerald-500/60 inline-block" />
                  Positive Attribution (Supports Stance)
                </span>
                <span className="flex items-center gap-1.5 text-rose-400">
                  <span className="w-3 h-3 rounded bg-rose-500/30 border border-rose-500/60 inline-block" />
                  Negative Attribution
                </span>
              </div>
            </div>

            {/* Tokenized Heatmap */}
            <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/80 min-h-[80px] flex flex-wrap items-center gap-2 leading-relaxed">
              {attributions.length > 0 ? (
                attributions.map((item, idx) => {
                  const intensity = Math.min(
                    Math.abs(item.attribution) / maxAbsAttr,
                    1
                  );
                  const isPositive = item.attribution >= 0;
                  return (
                    <span
                      key={idx}
                      title={`Token: ${item.token} | Attribution: ${item.attribution.toFixed(4)}`}
                      className="inline-flex items-center px-2.5 py-1 rounded-md text-sm font-mono border transition-transform hover:scale-105 cursor-help"
                      style={{
                        backgroundColor: isPositive
                          ? `rgba(16, 185, 129, ${0.15 + intensity * 0.45})`
                          : `rgba(244, 63, 94, ${0.15 + intensity * 0.45})`,
                        borderColor: isPositive
                          ? `rgba(16, 185, 129, ${0.3 + intensity * 0.5})`
                          : `rgba(244, 63, 94, ${0.3 + intensity * 0.5})`,
                        color: isPositive ? '#6EE7B7' : '#FDA4AF',
                      }}
                    >
                      <span>{item.token}</span>
                      <span className="text-[10px] ml-1.5 opacity-75 font-sans font-bold">
                        {item.attribution > 0 ? '+' : ''}
                        {item.attribution.toFixed(2)}
                      </span>
                    </span>
                  );
                })
              ) : (
                <span className="text-xs text-slate-400">
                  Click 'Compute Attributions' to view token gradient weights.
                </span>
              )}
            </div>
          </div>

          {/* Attribution Magnitude Chart */}
          {chartData.length > 0 && (
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
              <span className="text-xs font-mono uppercase tracking-wider text-slate-400 block">
                Top Decisive Semantic Tokens (Attribution Magnitude)
              </span>

              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={chartData}
                    layout="vertical"
                    margin={{ top: 10, right: 30, left: 60, bottom: 5 }}
                  >
                    <XAxis
                      type="number"
                      stroke="#475569"
                      fontSize={11}
                      fontFamily="monospace"
                    />
                    <YAxis
                      dataKey="token"
                      type="category"
                      stroke="#94A3B8"
                      fontSize={12}
                      fontFamily="monospace"
                      fontWeight={600}
                    />
                    <Tooltip
                      formatter={(val: any) => [val, 'Attribution Score']}
                      contentStyle={{
                        backgroundColor: '#0F172A',
                        border: '1px solid #334155',
                        borderRadius: '8px',
                        fontSize: '12px',
                        fontFamily: 'monospace',
                      }}
                    />
                    <ReferenceLine x={0} stroke="#64748B" />
                    <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={16}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Research Methodological Insight */}
          <div className="p-5 rounded-2xl bg-slate-900/50 border border-slate-800 text-xs text-slate-300 space-y-2 font-sans">
            <span className="font-semibold text-white flex items-center gap-1.5 font-mono uppercase tracking-wider text-[11px]">
              <Info className="w-3.5 h-3.5 text-emerald-400" />
              <span>Axiomatic Integrated Gradients Formulation</span>
            </span>
            <p className="text-slate-400 leading-relaxed">
              Attributions are computed via Riemann-sum integration across $m=16$ steps along the straight-line path from a zero-vector baseline embedding $x'$ to the input representation $x$:
            </p>
            <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800 font-mono text-[11px] text-emerald-300 overflow-x-auto">
              {"IG_i(x) = (x_i - x'_i) × (1/m) · Σ [∂F(x' + (k/m)(x - x')) / ∂x_i]"}
            </div>
            <p className="text-slate-400 leading-relaxed text-[11px]">
              Tokens receiving large positive weights (e.g., <code className="text-emerald-300 bg-slate-950 px-1 py-0.5 rounded">गंभीर</code>, <code className="text-emerald-300 bg-slate-950 px-1 py-0.5 rounded">खतरा</code>, <code className="text-emerald-300 bg-slate-950 px-1 py-0.5 rounded">সংকট</code>) actively steer predictions towards concern (<code className="text-emerald-300">FAVOUR</code>), whereas negation particles (<code className="text-rose-300 bg-slate-950 px-1 py-0.5 rounded">नहीं</code>, <code className="text-rose-300 bg-slate-950 px-1 py-0.5 rounded">নয়</code>) and irony cues flip orientation towards denial (<code className="text-rose-300">AGAINST</code>).
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
