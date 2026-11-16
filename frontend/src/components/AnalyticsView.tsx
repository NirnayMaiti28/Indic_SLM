import React, { useEffect, useState } from 'react';
import {
  BarChart3,
  Award,
  CheckCircle2,
  Clock,
  Cpu,
  Layers,
  PieChart as PieIcon,
  TrendingUp,
  Trophy,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  PieChart,
  Pie,
} from 'recharts';
import type { BenchmarkResponse } from '../types';
import { fetchBenchmarks } from '../api';

export const AnalyticsView: React.FC = () => {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchBenchmarks()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load benchmarks:', err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="py-20 flex flex-col items-center justify-center space-y-3 text-slate-400 font-mono text-xs">
        <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
        <span>Loading Research Benchmarks & Metrics...</span>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="py-12 text-center text-slate-400 text-xs font-mono">
        Failed to load benchmark data.
      </div>
    );
  }

  const highestScoringModel = data.models.reduce((prev, curr) =>
    curr.macro_f1 > prev.macro_f1 ? curr : prev
  , data.models[0]);
  const deployedModel = data.models.find((m) => m.name.includes('Distilled')) || data.models[0];

  const modelChartData = data.models.map((m) => {
    const isTopOverall = m.name === highestScoringModel.name;
    const isDeployed = m.name.includes('Distilled');
    return {
      name: m.name.replace(' Baseline', '').replace(' Linear', '').replace(' (Ours)', ''),
      macroF1: Math.round(m.macro_f1 * 1000) / 10,
      accuracy: Math.round(m.accuracy * 1000) / 10,
      color: isTopOverall ? '#F59E0B' : isDeployed ? '#10B981' : m.name.includes('Teacher') ? '#06B6D4' : '#64748B',
    };
  });

  const splitPieData = [
    { name: 'Train Split', value: data.datasets.splits.train.total, color: '#10B981' },
    { name: 'Val Split', value: data.datasets.splits.val.total, color: '#0EA5E9' },
    { name: 'Test Split', value: data.datasets.splits.test.total, color: '#8B5CF6' },
  ];

  const langPieData = Object.entries(data.datasets.languages).map(([lang, count], idx) => ({
    name: lang,
    value: count,
    color: idx === 0 ? '#10B981' : '#14B8A6',
  }));

  return (
    <div className="space-y-10 py-2">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-emerald-400" />
          <span>Research Evaluation & Empirical Benchmarks</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Strictly evaluated on the standardized, leakage-controlled test split of the Custom Indic Climate Stance Dataset ($n=235$).
        </p>
      </div>

      {/* Benchmark Summary KPI Banner */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 font-mono text-xs">
        <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-600/40 space-y-1.5">
          <div className="flex items-center justify-between text-amber-400">
            <span className="text-[10px] uppercase font-bold tracking-wider">Highest Benchmark (Overall)</span>
            <Trophy className="w-4 h-4 text-amber-400" />
          </div>
          <span className="text-2xl font-bold text-amber-300 block">{highestScoringModel.macro_f1.toFixed(4)}</span>
          <span className="text-[11px] text-slate-200 font-semibold block truncate">{highestScoringModel.name}</span>
          <span className="text-[10px] text-slate-400 block">Subword character n-gram SVM</span>
        </div>

        <div className="p-4 rounded-xl bg-emerald-950/30 border border-emerald-600/50 space-y-1.5">
          <div className="flex items-center justify-between text-emerald-400">
            <span className="text-[10px] uppercase font-bold tracking-wider">Top Neural SLM (Deployed)</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          </div>
          <span className="text-2xl font-bold text-emerald-300 block">{deployedModel.macro_f1.toFixed(4)}</span>
          <span className="text-[11px] text-slate-200 font-semibold block truncate">{deployedModel.name}</span>
          <span className="text-[10px] text-slate-400 block">4L Distilled contextual Small LM</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 space-y-1.5">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[10px] uppercase font-bold tracking-wider">Distillation Gain</span>
            <TrendingUp className="w-4 h-4 text-emerald-400" />
          </div>
          <span className="text-2xl font-bold text-emerald-400 block">+14.54%</span>
          <span className="text-[11px] text-slate-200 block">vs. Standalone Student</span>
          <span className="text-[10px] text-slate-400 block">Macro-F1 0.8120 → 0.9574</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 space-y-1.5">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-[10px] uppercase font-bold tracking-wider">Inference Speedup</span>
            <Clock className="w-4 h-4 text-cyan-400" />
          </div>
          <span className="text-2xl font-bold text-cyan-300 block">2.36x Faster</span>
          <span className="text-[11px] text-slate-200 block">P50 Latency: 7.40 ms</span>
          <span className="text-[10px] text-slate-400 block">vs. 17.50 ms (12L Teacher)</span>
        </div>
      </div>

      {/* Model Benchmark Comparison Table */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Award className="w-4 h-4 text-emerald-400" />
            <span>Comprehensive Model Performance Matrix</span>
          </span>
          <div className="flex items-center gap-3 text-[11px] font-mono">
            <span className="text-amber-400 flex items-center gap-1">
              <Trophy className="w-3 h-3" />
              Highest Overall: {highestScoringModel.macro_f1.toFixed(4)}
            </span>
            <span className="text-emerald-400 flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3" />
              Deployed SLM: {deployedModel.macro_f1.toFixed(4)}
            </span>
          </div>
        </div>

        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-950 text-slate-400 text-[11px] uppercase border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Model Architecture</th>
                <th className="py-2.5 px-3">Category</th>
                <th className="py-2.5 px-3">Total Params</th>
                <th className="py-2.5 px-3">Trainable</th>
                <th className="py-2.5 px-3 font-bold">Macro F1</th>
                <th className="py-2.5 px-3">Accuracy</th>
                <th className="py-2.5 px-3">Hindi F1</th>
                <th className="py-2.5 px-3">Bengali F1</th>
                <th className="py-2.5 px-3">Latency (P50)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 bg-slate-900/40">
              {data.models.map((m, idx) => {
                const isTopOverall = m.name === highestScoringModel.name;
                const isDeployed = m.name.includes('Distilled');
                return (
                  <tr
                    key={idx}
                    className={`transition-colors ${
                      isTopOverall
                        ? 'bg-amber-950/20 border-l-2 border-amber-500 font-medium'
                        : isDeployed
                        ? 'bg-emerald-950/25 border-l-2 border-emerald-500 font-semibold'
                        : 'hover:bg-slate-800/40'
                    }`}
                  >
                    <td className="py-2.5 px-3 text-slate-200">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        {isTopOverall && (
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-900/60 text-amber-300 border border-amber-600/50 flex items-center gap-1">
                            <Trophy className="w-3 h-3 text-amber-400 shrink-0" />
                            Highest Macro-F1
                          </span>
                        )}
                        {isDeployed && (
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-900/60 text-emerald-300 border border-emerald-600/50 flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
                            Deployed SLM
                          </span>
                        )}
                        <span>{m.name}</span>
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">{m.category}</td>
                    <td className="py-2.5 px-3 text-slate-400">{m.parameters}</td>
                    <td className="py-2.5 px-3 text-slate-400">{m.trainable_parameters}</td>
                    <td className={`py-2.5 px-3 font-bold text-sm ${
                      isTopOverall ? 'text-amber-400' : isDeployed ? 'text-emerald-400' : 'text-slate-300'
                    }`}>
                      {m.macro_f1.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {m.accuracy.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {m.hindi_f1.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {m.bengali_f1.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">
                      {m.latency_ms} ms
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confusion Matrix & Per-Class Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion Matrix Grid */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-300 block">
            Test Confusion Matrix (Rows: Gold Label, Cols: Predicted)
          </span>

          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800">
            <div className="grid grid-cols-4 gap-2 text-center font-mono text-xs">
              {/* Header */}
              <div className="text-slate-400 font-bold text-[10px]">GOLD \ PRED</div>
              <div className="text-emerald-400 font-semibold">FAVOUR</div>
              <div className="text-rose-400 font-semibold">AGAINST</div>
              <div className="text-slate-400 font-semibold">NONE</div>

              {/* Rows */}
              {data.confusion_matrix.labels.map((rowLbl, rIdx) => (
                <React.Fragment key={rIdx}>
                  <div className="text-slate-300 font-semibold py-3 text-left">
                    {rowLbl}
                  </div>
                  {data.confusion_matrix.matrix[rIdx].map((val, cIdx) => {
                    const isDiagonal = rIdx === cIdx;
                    return (
                      <div
                        key={cIdx}
                        className={`py-3 rounded-lg font-bold text-sm flex items-center justify-center transition-all ${
                          isDiagonal
                            ? 'bg-emerald-950/70 border border-emerald-600/50 text-emerald-300 shadow-sm'
                            : val > 0
                            ? 'bg-rose-950/40 border border-rose-800/40 text-rose-300'
                            : 'bg-slate-900/40 text-slate-400'
                        }`}
                      >
                        {val}
                      </div>
                    );
                  })}
                </React.Fragment>
              ))}
            </div>
          </div>

          {/* Per-class precision/recall */}
          <div className="grid grid-cols-3 gap-2 pt-2 text-xs font-mono">
            {Object.entries(data.confusion_matrix.per_class).map(([cls, scores]) => (
              <div key={cls} className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                <span className="text-[10px] text-slate-400 uppercase font-bold">{cls}</span>
                <div className="text-slate-300 text-[11px]">
                  F1: <span className="font-semibold text-emerald-400">{scores.f1.toFixed(3)}</span>
                </div>
                <div className="text-slate-400 text-[10px]">
                  Prec: {scores.precision.toFixed(3)} | Rec: {scores.recall.toFixed(3)}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Macro-F1 Comparison Bar Chart */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-300 block">
            Model Macro-F1 Comparison (%)
          </span>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={modelChartData}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
              >
                <XAxis type="number" domain={[0, 100]} unit="%" stroke="#475569" fontSize={11} fontFamily="monospace" />
                <YAxis dataKey="name" type="category" stroke="#94A3B8" fontSize={11} fontFamily="monospace" />
                <Tooltip
                  formatter={(val: any) => [`${val}%`, 'Macro F1']}
                  contentStyle={{
                    backgroundColor: '#0F172A',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                    fontSize: '12px',
                    fontFamily: 'monospace',
                  }}
                />
                <Bar dataKey="macroF1" radius={[0, 4, 4, 0]} barSize={16}>
                  {modelChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-2">
            <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 text-[11px] text-slate-300 font-sans">
              <span className="text-amber-400 font-semibold font-mono">🏆 Benchmark Analysis:</span> <strong>TF-IDF Char (2-5) + Linear SVM</strong> achieves the highest overall Macro-F1 (<strong className="text-amber-300">0.9916</strong>) on this corpus by exploiting character subword n-gram patterns.
            </div>
            <div className="p-3 rounded-xl bg-emerald-950/30 border border-emerald-900/50 text-[11px] text-slate-300 font-sans">
              <span className="text-emerald-400 font-semibold font-mono">🚀 Deployed Model:</span> <strong>Distilled Student SLM (4L + KD)</strong> is our deployed architecture (<strong className="text-emerald-300">0.9574</strong> Macro-F1). Distillation delivers a <strong className="text-white">+14.5% absolute gain</strong> over the standalone 4L student (0.8120 $\to$ 0.9574) while providing deep semantic contextualization, negation sensitivity, and Integrated Gradients explainability that n-gram models cannot provide.
            </div>
          </div>
        </div>
      </div>

      {/* Dataset & Latency Benchmarks */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Dataset Distribution */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-300 block">
            Indic Climate Dataset Splits & Distribution (n=1,500)
          </span>

          <div className="grid grid-cols-3 gap-2 font-mono text-center text-xs">
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase">Train Set</span>
              <span className="text-lg font-bold text-emerald-400 block my-1">
                {data.datasets.splits.train.total}
              </span>
              <span className="text-[10px] text-slate-400">70% Stratified</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase">Val Set</span>
              <span className="text-lg font-bold text-sky-400 block my-1">
                {data.datasets.splits.val.total}
              </span>
              <span className="text-[10px] text-slate-400">15% Stratified</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
              <span className="text-[10px] text-slate-400 uppercase">Test Set</span>
              <span className="text-lg font-bold text-purple-400 block my-1">
                {data.datasets.splits.test.total}
              </span>
              <span className="text-[10px] text-slate-400">15% Stratified</span>
            </div>
          </div>

          <div className="text-xs text-slate-400 font-mono space-y-1.5 pt-2">
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Hindi Samples (Devanagari + Hinglish)</span>
              <span className="text-slate-200 font-bold">{data.datasets.languages.Hindi}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Bengali Samples (Bangla script + Banglish)</span>
              <span className="text-slate-200 font-bold">{data.datasets.languages.Bengali}</span>
            </div>
            <div className="flex justify-between py-1">
              <span>Data Split Strategy</span>
              <span className="text-emerald-400 font-semibold">Group-Stratified (Zero Duplicate Leakage)</span>
            </div>
          </div>
        </div>

        {/* Latency & Throughput Benchmarks */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-300 block">
            Inference Latency & Throughput Profile
          </span>

          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-950 text-slate-400 text-[10px] uppercase border-b border-slate-800">
                <tr>
                  <th className="py-2 px-3">Batch Size</th>
                  <th className="py-2 px-3">Hardware Mode</th>
                  <th className="py-2 px-3">P50 Latency</th>
                  <th className="py-2 px-3">P95 Latency</th>
                  <th className="py-2 px-3">Throughput</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 bg-slate-900/40">
                {data.latency_benchmarks.map((lb, i) => (
                  <tr key={i} className="hover:bg-slate-800/40">
                    <td className="py-2 px-3 text-slate-300">{lb.batch_size}</td>
                    <td className="py-2 px-3 text-slate-300">{lb.device}</td>
                    <td className="py-2 px-3 text-emerald-400 font-bold">{lb.p50_ms} ms</td>
                    <td className="py-2 px-3 text-slate-400">{lb.p95_ms} ms</td>
                    <td className="py-2 px-3 text-slate-300">{lb.throughput} /s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
            By pruning from 12 layers down to 4 layers, transformer computation is cut by <strong>66.7%</strong>, yielding a <strong>2.21x latency acceleration</strong> over the full MuRIL teacher.
          </p>
        </div>
      </div>
    </div>
  );
};
