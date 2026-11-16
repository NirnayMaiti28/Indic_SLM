import React from 'react';
import {
  Sparkles,
  Zap,
  BarChart3,
  Layers,
  ArrowRight,
  Languages,
  ShieldCheck,
  CheckCircle2,
  Clock,
  Check,
} from 'lucide-react';

interface LandingViewProps {
  onLaunchAnalyzer: () => void;
  onExploreArchitecture: () => void;
  onViewBenchmarks: () => void;
}

export const LandingView: React.FC<LandingViewProps> = ({
  onLaunchAnalyzer,
  onExploreArchitecture,
  onViewBenchmarks,
}) => {
  const highlights = [
    {
      label: 'SLM Test Macro F1',
      value: '0.9574',
      detail: 'Distilled Student SLM (Overall high: 0.9916 by Char-SVM)',
      color: 'text-emerald-400',
    },
    {
      label: 'Model Architecture',
      value: '4 Layers',
      detail: '66.7% depth reduction from 12-layer MuRIL',
      color: 'text-teal-400',
    },
    {
      label: 'Inference Latency',
      value: '7.40 ms',
      detail: 'P50 single-comment latency on CUDA FP16',
      color: 'text-cyan-400',
    },
    {
      label: 'Trainable Params',
      value: '100,611',
      detail: 'LoRA r=8 adapter + Mean Pooling StanceHead',
      color: 'text-emerald-300',
    },
  ];

  const linguisticFeatures = [
    {
      title: 'Native Hindi (Devanagari)',
      sample: 'जलवायु परिवर्तन हमारे अस्तित्व के लिए गंभीर खतरा है।',
      stance: 'FAVOUR',
    },
    {
      title: 'Native Bengali (বাংলা লিপি)',
      sample: 'জলবায়ু পরিবর্তন একটি বাস্তব সংকট, কার্বন নির্গমন কমাতে হবে।',
      stance: 'FAVOUR',
    },
    {
      title: 'Hindi Negation Denial',
      sample: 'ग्लोबल वार्मिंग कोई गंभीर समस्या नहीं है, यह सिर्फ प्रोपेगैंडा है।',
      stance: 'AGAINST',
    },
    {
      title: 'Romanized Hindi (Hinglish)',
      sample: 'Climate change sach me ek bahut serious concern hai bhai.',
      stance: 'FAVOUR',
    },
    {
      title: 'Romanized Bengali (Banglish)',
      sample: 'Global warming kono issue na, shob funding pabar natok.',
      stance: 'AGAINST',
    },
    {
      title: 'Neutral Inquiry / Query',
      sample: 'सौर ऊर्जा पैनल लगाने का सरकारी खर्च कितना आता है?',
      stance: 'NONE',
    },
  ];

  return (
    <div className="space-y-12 py-4">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-b from-slate-900/90 to-slate-950/70 border border-slate-800/90 p-8 sm:p-12 shadow-2xl">
        <div className="absolute top-0 right-0 -mt-12 -mr-12 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 -mb-12 -ml-12 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-4xl space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/60 border border-emerald-800/50 text-emerald-400 text-xs font-mono">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Research-Grade Multilingual Small Language Model</span>
          </div>

          <h1 className="text-3xl sm:text-5xl font-bold tracking-tight text-white leading-tight">
            Indic Climate Stance Detection
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-3xl font-normal">
            A high-efficiency, knowledge-distilled multilingual Small Language Model (SLM) 
            tailored for low-latency three-way stance classification on Hindi, Bengali, Hinglish, 
            and Banglish social media discourse.
          </p>

          {/* Target Claim Callout */}
          <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-xs sm:text-sm font-sans flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <span className="text-slate-400 uppercase tracking-wider font-mono text-[11px] block">
                Target Research Claim
              </span>
              <span className="text-emerald-300 font-medium italic">
                “Climate change and global warming is a serious concern.”
              </span>
            </div>
            <div className="flex items-center gap-2 font-mono text-xs">
              <span className="px-2 py-0.5 rounded bg-emerald-900/40 text-emerald-300 border border-emerald-800/60">
                FAVOUR
              </span>
              <span className="px-2 py-0.5 rounded bg-rose-900/40 text-rose-300 border border-rose-800/60">
                AGAINST
              </span>
              <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                NONE
              </span>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap items-center gap-4 pt-2">
            <button
              onClick={onLaunchAnalyzer}
              className="px-5 py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-sm transition-all shadow-lg shadow-emerald-500/20 flex items-center gap-2 cursor-pointer"
            >
              <span>Launch Stance Analyzer</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={onViewBenchmarks}
              className="px-5 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-sm font-medium transition-all flex items-center gap-2 cursor-pointer"
            >
              <BarChart3 className="w-4 h-4 text-emerald-400" />
              <span>View Benchmarks</span>
            </button>
            <button
              onClick={onExploreArchitecture}
              className="px-4 py-2.5 rounded-lg text-slate-400 hover:text-slate-200 text-sm font-medium transition-colors flex items-center gap-1.5 cursor-pointer"
            >
              <Layers className="w-4 h-4 text-slate-400" />
              <span>Model Architecture</span>
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Banner */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {highlights.map((h, i) => (
          <div
            key={i}
            className="p-5 rounded-xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-sm"
          >
            <span className="text-xs font-mono uppercase text-slate-400 block tracking-wider">
              {h.label}
            </span>
            <div className={`text-2xl sm:text-3xl font-bold font-mono my-1 ${h.color}`}>
              {h.value}
            </div>
            <p className="text-[12px] text-slate-400 leading-snug">
              {h.detail}
            </p>
          </div>
        ))}
      </div>

      {/* Linguistic Coverage Grid */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-white tracking-tight flex items-center gap-2">
              <Languages className="w-5 h-5 text-emerald-400" />
              <span>Supported Linguistic Variations & Scripts</span>
            </h2>
            <p className="text-xs text-slate-400">
              Evaluated on complex social-media syntax including code-mixing, dialectal negation, and sarcasm.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {linguisticFeatures.map((feat, idx) => (
            <div
              key={idx}
              className="p-4 rounded-xl bg-slate-900/50 border border-slate-800/70 hover:border-slate-700 transition-all flex flex-col justify-between space-y-3"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-300 font-mono">
                  {feat.title}
                </span>
                <span
                  className={`text-[10px] font-mono px-2 py-0.5 rounded font-semibold ${
                    feat.stance === 'FAVOUR'
                      ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-800/60'
                      : feat.stance === 'AGAINST'
                      ? 'bg-rose-950/80 text-rose-300 border border-rose-800/60'
                      : 'bg-slate-800 text-slate-300 border border-slate-700'
                  }`}
                >
                  {feat.stance}
                </span>
              </div>
              <p className="text-xs text-slate-300 italic font-sans bg-slate-950/50 p-2.5 rounded-lg border border-slate-800/60">
                “{feat.sample}”
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Research Methodology Card */}
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider font-mono text-emerald-400 flex items-center gap-2">
          <ShieldCheck className="w-4 h-4" />
          <span>Core Research Methodology Highlights</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-slate-300">
          <div className="space-y-2">
            <span className="font-semibold text-white block text-sm">
              1. Vocabulary-Preserving Pruning
            </span>
            <p className="text-slate-400 leading-relaxed">
              Pruning depth from 12 to 4 layers while preserving MuRIL's complete 197,258 token vocabulary
              prevents the out-of-vocabulary (UNK) explosion typical of smaller multilingual models.
            </p>
          </div>
          <div className="space-y-2">
            <span className="font-semibold text-white block text-sm">
              2. Offline Knowledge Distillation
            </span>
            <p className="text-slate-400 leading-relaxed">
              Temperature-scaled KL divergence (T=2.0, α=0.5, β=0.5) transfers dark knowledge from the 12-layer
              MuRIL teacher, delivering a +14.5% Macro-F1 improvement over standalone student training.
            </p>
          </div>
          <div className="space-y-2">
            <span className="font-semibold text-white block text-sm">
              3. Hard Negation Preservation
            </span>
            <p className="text-slate-400 leading-relaxed">
              Custom multilingual preprocessing strictly safeguards negation particles (नहीं, নয়, nahi, nai)
              and social media emojis, preventing semantic polarity inversions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

