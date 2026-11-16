import React, { useState } from 'react';
import {
  Sparkles,
  Zap,
  Clock,
  Cpu,
  Layers,
  AlertCircle,
  HelpCircle,
  CheckCircle2,
  XCircle,
  MinusCircle,
  Share2,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from 'recharts';
import type { PredictResponse } from '../types';
import { predictStance } from '../api';

const PRESETS = [
  {
    label: '🇮🇳 Hindi — Concern (FAVOUR)',
    text: 'जलवायु परिवर्तन हमारे अस्तित्व के लिए सबसे गंभीर खतरा है।',
  },
  {
    label: '🇮🇳 Hindi — Negation Denial (AGAINST)',
    text: 'ग्लोबल वार्मिंग कोई गंभीर समस्या नहीं है, यह सिर्फ पश्चिमी देशों का प्रोपेगैंडा है।',
  },
  {
    label: '🇮🇳 Hindi — Winter Sarcasm (AGAINST)',
    text: 'वाह भाई! दिसंबर में 5 डिग्री तापमान है और इनको ग्लोबल वार्मिंग नजर आ रही है। 😂',
  },
  {
    label: '🇧🇩 Bengali — Concern (FAVOUR)',
    text: 'জলবায়ু পরিবর্তন একটি বাস্তব ও ভয়াবহ সংকট, এখনই কার্বন নির্গমন কমাতে হবে। 🌍',
  },
  {
    label: '🇧🇩 Bengali — Negation Denial (AGAINST)',
    text: 'গ্লোবাল ওয়ার্মিং কোনো গুরুতর সমস্যা নয়, এটি একটি সাজানো নাটক।',
  },
  {
    label: '💬 Hinglish (FAVOUR)',
    text: 'Climate change sach me ek bahut serious concern hai bhai, glaciers melt ho rahe hain.',
  },
  {
    label: '💬 Banglish (AGAINST)',
    text: 'Global warming kono serious issue na, shob funding pabar natok. ❄️',
  },
  {
    label: '⚪ Neutral Inquiry (NONE)',
    text: 'सौर ऊर्जा पैनल लगाने का सरकारी खर्च कितना आता है, कोई जानकारी दे सकता है?',
  },
];

interface AnalyzerViewProps {
  onInspectExplainability?: (result: PredictResponse) => void;
}

export const AnalyzerView: React.FC<AnalyzerViewProps> = ({
  onInspectExplainability,
}) => {
  const [inputText, setInputText] = useState(
    'जलवायु परिवर्तन हमारे अस्तित्व के लिए सबसे गंभीर खतरा है।'
  );
  const [includeExplain, setIncludeExplain] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!inputText.trim()) {
      setError('Please enter a comment or select a preset statement.');
      return;
    }
    setError(null);
    setLoading(true);

    try {
      const res = await predictStance(inputText, includeExplain);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'An error occurred while running inference.');
    } finally {
      setLoading(false);
    }
  };

  const chartData = result?.probabilities
    ? [
        {
          name: 'FAVOUR',
          probability: Math.round(result.probabilities.FAVOUR * 1000) / 10,
          color: '#10B981', // emerald-500
        },
        {
          name: 'AGAINST',
          probability: Math.round(result.probabilities.AGAINST * 1000) / 10,
          color: '#F43F5E', // rose-500
        },
        {
          name: 'NONE',
          probability: Math.round(result.probabilities.NONE * 1000) / 10,
          color: '#64748B', // slate-500
        },
      ]
    : [];

  return (
    <div className="space-y-8 py-2">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-emerald-400" />
          <span>Interactive Stance Analyzer</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Target Claim: <span className="text-slate-200 italic font-serif">“Climate change and global warming is a serious concern.”</span>
        </p>
      </div>

      {/* Benchmark Presets */}
      <div className="space-y-2">
        <span className="text-xs font-mono uppercase text-slate-400 tracking-wider">
          Benchmark Presets:
        </span>
        <div className="flex flex-wrap gap-1.5">
          {PRESETS.map((p, i) => (
            <button
              key={i}
              onClick={() => {
                setInputText(p.text);
                setResult(null);
                setError(null);
              }}
              className="px-2.5 py-1 rounded bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-800 text-xs transition-colors cursor-pointer"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Input Box */}
      <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800/90 shadow-xl space-y-4">
        <div className="relative">
          <textarea
            rows={4}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type or paste Hindi, Bengali, Hinglish, Banglish, or English social-media text here..."
            className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-4 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/50 transition-all font-sans leading-relaxed"
          />
          <div className="flex justify-between items-center text-[11px] text-slate-400 px-1 pt-1 font-mono">
            <span>{inputText.length} characters</span>
            <span className="text-slate-400">Context window: max 64 subword tokens</span>
          </div>
        </div>

        {/* Options & Action */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pt-2">
          <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-300 select-none">
            <input
              type="checkbox"
              checked={includeExplain}
              onChange={(e) => setIncludeExplain(e.target.checked)}
              className="rounded bg-slate-950 border-slate-700 text-emerald-500 focus:ring-emerald-500/30 w-4 h-4"
            />
            <span className="flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span>Compute Token-Level Attributions (Integrated Gradients)</span>
            </span>
          </label>

          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="w-full sm:w-auto px-6 py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-slate-950 font-semibold text-sm transition-all shadow-md shadow-emerald-500/20 flex items-center justify-center gap-2 cursor-pointer font-sans"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                <span>Running Inference...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Analyze Stance</span>
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="p-3 rounded-lg bg-rose-950/50 border border-rose-800 text-rose-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Prediction Output Section */}
      {result && result.predicted_label && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Primary Classification Result */}
            <div className="lg:col-span-1 p-6 rounded-2xl bg-slate-900/80 border border-slate-800/90 flex flex-col justify-between space-y-6">
              <div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 block">
                  Predicted Stance
                </span>

                <div className="mt-2 flex items-center gap-3">
                  {result.predicted_label === 'FAVOUR' && (
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-950/80 border border-emerald-700/60 text-emerald-300">
                      <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                      <span className="text-xl font-bold font-mono">FAVOUR</span>
                    </div>
                  )}
                  {result.predicted_label === 'AGAINST' && (
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-rose-950/80 border border-rose-700/60 text-rose-300">
                      <XCircle className="w-6 h-6 text-rose-400" />
                      <span className="text-xl font-bold font-mono">AGAINST</span>
                    </div>
                  )}
                  {result.predicted_label === 'NONE' && (
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 border border-slate-700 text-slate-300">
                      <MinusCircle className="w-6 h-6 text-slate-400" />
                      <span className="text-xl font-bold font-mono">NONE</span>
                    </div>
                  )}
                </div>

                <p className="text-xs text-slate-300 mt-3 leading-relaxed">
                  {result.predicted_label === 'FAVOUR' &&
                    'The input supports, affirms concern, or agrees with the reality and urgency of climate change.'}
                  {result.predicted_label === 'AGAINST' &&
                    'The input dismisses, opposes, denies, or rejects climate change concern (e.g. natural cycle, hoax, scam).'}
                  {result.predicted_label === 'NONE' &&
                    'The input is neutral, factual, an unrelated query, or expresses ambiguous stance.'}
                </p>
              </div>

              {/* Confidence Meter */}
              <div className="space-y-1.5 pt-4 border-t border-slate-800">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-slate-400">Model Confidence</span>
                  <span className="font-bold text-white">
                    {result.confidence ? (result.confidence * 100).toFixed(1) : 0}%
                  </span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden border border-slate-800">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      result.predicted_label === 'FAVOUR'
                        ? 'bg-emerald-500'
                        : result.predicted_label === 'AGAINST'
                        ? 'bg-rose-500'
                        : 'bg-slate-500'
                    }`}
                    style={{ width: `${(result.confidence || 0) * 100}%` }}
                  />
                </div>
              </div>

              {/* Latency & Device metadata */}
              <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 pt-2 border-t border-slate-800/80">
                <span className="flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5 text-slate-500" />
                  <span>{result.latency_ms} ms</span>
                </span>
                <span className="flex items-center gap-1">
                  <Cpu className="w-3.5 h-3.5 text-slate-500" />
                  <span>{result.device.toUpperCase()}</span>
                </span>
              </div>
            </div>

            {/* Three-Class Probability Distribution Chart */}
            <div className="lg:col-span-2 p-6 rounded-2xl bg-slate-900/80 border border-slate-800/90 flex flex-col justify-between">
              <div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 block mb-1">
                  {"Probability Distribution (Softmax Posterior)"}
                </span>
                <p className="text-xs text-slate-400 mb-4">
                  Full posterior distribution output by the student model's classification head.
                </p>
              </div>

              <div className="h-44 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={chartData}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                  >
                    <XAxis
                      type="number"
                      domain={[0, 100]}
                      unit="%"
                      stroke="#475569"
                      fontSize={11}
                      fontFamily="monospace"
                    />
                    <YAxis
                      dataKey="name"
                      type="category"
                      stroke="#94A3B8"
                      fontSize={12}
                      fontFamily="monospace"
                      fontWeight={600}
                    />
                    <Tooltip
                      formatter={(val: any) => [`${val}%`, 'Probability']}
                      contentStyle={{
                        backgroundColor: '#0F172A',
                        border: '1px solid #334155',
                        borderRadius: '8px',
                        fontSize: '12px',
                        fontFamily: 'monospace',
                      }}
                    />
                    <Bar dataKey="probability" radius={[0, 4, 4, 0]} barSize={22}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Exact percentages */}
              <div className="grid grid-cols-3 gap-2 pt-4 border-t border-slate-800 text-center font-mono text-xs">
                <div>
                  <span className="text-slate-400 block text-[10px]">FAVOUR</span>
                  <span className="text-emerald-400 font-semibold">
                    {result.probabilities?.FAVOUR ? (result.probabilities.FAVOUR * 100).toFixed(2) : 0}%
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px]">AGAINST</span>
                  <span className="text-rose-400 font-semibold">
                    {result.probabilities?.AGAINST ? (result.probabilities.AGAINST * 100).toFixed(2) : 0}%
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px]">NONE</span>
                  <span className="text-slate-400 font-semibold">
                    {result.probabilities?.NONE ? (result.probabilities.NONE * 100).toFixed(2) : 0}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Linguistic Metadata Strip */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-4 font-mono text-xs">
            <div>
              <span className="text-slate-400 block text-[10px] uppercase">Detected Script</span>
              <span className="text-slate-200 font-medium">{result.script || 'Unknown'}</span>
            </div>
            <div>
              <span className="text-slate-400 block text-[10px] uppercase">Detected Language</span>
              <span className="text-slate-200 font-medium">{result.language || 'Unknown'}</span>
            </div>
            <div>
              <span className="text-slate-400 block text-[10px] uppercase">Code-Mixed Text</span>
              <span className={result.is_code_mixed ? 'text-amber-400 font-medium' : 'text-slate-400'}>
                {result.is_code_mixed ? 'Yes (Mixed Script)' : 'No (Monolingual)'}
              </span>
            </div>
            <div>
              <span className="text-slate-400 block text-[10px] uppercase">Linguistic Negation</span>
              <span className={result.has_negation ? 'text-emerald-400 font-medium' : 'text-slate-400'}>
                {result.has_negation ? 'Detected (Preserved)' : 'Not Present'}
              </span>
            </div>
          </div>

          {/* Quick Token Attributions (Integrated Gradients preview) */}
          {result.explanations && result.explanations.length > 0 && (
            <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5 text-amber-400" />
                  <span>Decisive Semantic Cues (Integrated Gradients Attributions)</span>
                </span>
                {onInspectExplainability && (
                  <button
                    onClick={() => onInspectExplainability(result)}
                    className="text-xs text-emerald-400 hover:text-emerald-300 underline font-mono cursor-pointer"
                  >
                    Deep Dive in Explainability Tab →
                  </button>
                )}
              </div>

              <div className="flex flex-wrap gap-2 pt-1">
                {result.explanations.map((tok, idx) => (
                  <span
                    key={idx}
                    className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-mono border ${
                      tok.attribution > 0
                        ? 'bg-emerald-950/60 border-emerald-800/80 text-emerald-300'
                        : 'bg-rose-950/60 border-rose-800/80 text-rose-300'
                    }`}
                  >
                    <span className="font-semibold">{tok.token}</span>
                    <span className="text-[10px] opacity-75">
                      ({tok.attribution > 0 ? '+' : ''}{tok.attribution.toFixed(3)})
                    </span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
