import React, { useEffect, useState } from 'react';
import {
  Layers,
  Cpu,
  Zap,
  ShieldCheck,
  CheckCircle2,
  GitBranch,
  Terminal,
  Code2,
} from 'lucide-react';
import type { ModelInfoResponse } from '../types';
import { fetchModelInfo } from '../api';

export const ModelInfoView: React.FC = () => {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);

  useEffect(() => {
    fetchModelInfo()
      .then(setInfo)
      .catch((err) => console.error('Failed to load model info:', err));
  }, []);

  return (
    <div className="space-y-10 py-2">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <Layers className="w-6 h-6 text-emerald-400" />
          <span>Model Architecture & Distillation Methodology</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Detailed technical specifications of the 4-layer Compact Student Small Language Model (SLM).
        </p>
      </div>

      {/* Architecture Flow Diagram */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
        <span className="text-xs font-mono uppercase tracking-wider text-slate-300 block">
          End-to-End Inference Pipeline
        </span>

        <div className="grid grid-cols-1 md:grid-cols-6 gap-3 text-xs font-mono">
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-center flex flex-col justify-between space-y-2">
            <span className="text-[10px] text-slate-400 font-bold uppercase">1. Raw Input</span>
            <span className="text-slate-200">Hindi / Bengali / Romanized text</span>
            <span className="text-[10px] text-slate-400">Social Media</span>
          </div>

          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-center flex flex-col justify-between space-y-2">
            <span className="text-[10px] text-slate-400 font-bold uppercase">2. Preprocessing</span>
            <span className="text-emerald-400 font-semibold">Unicode NFC</span>
            <span className="text-[10px] text-slate-400">URL/Mention Mask</span>
          </div>

          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-center flex flex-col justify-between space-y-2">
            <span className="text-[10px] text-slate-400 font-bold uppercase">3. Tokenizer</span>
            <span className="text-teal-400 font-semibold">MuRIL WordPiece</span>
            <span className="text-[10px] text-slate-400">197,258 vocab (max 64)</span>
          </div>

          <div className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-600/50 text-center flex flex-col justify-between space-y-2">
            <span className="text-[10px] text-emerald-400 font-bold uppercase">4. Encoder + LoRA</span>
            <span className="text-white font-bold">4 Pruned Layers</span>
            <span className="text-[10px] text-emerald-300">r=8, α=16 (q, v)</span>
          </div>

          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-center flex flex-col justify-between space-y-2">
            <span className="text-[10px] text-slate-400 font-bold uppercase">5. StanceHead</span>
            <span className="text-cyan-400 font-semibold">Mean Pooling</span>
            <span className="text-[10px] text-slate-400">Dropout(0.1) → Linear</span>
          </div>

          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-center flex flex-col justify-between space-y-2">
            <span className="text-[10px] text-slate-400 font-bold uppercase">6. Output</span>
            <span className="text-emerald-400 font-bold">3 Logits</span>
            <span className="text-[10px] text-slate-400">Softmax Stance Prob</span>
          </div>
        </div>
      </div>

      {/* Technical Parameter Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 font-mono text-xs">
        <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 block text-[10px] uppercase">Base Foundation Encoder</span>
          <span className="text-sm font-bold text-white block my-1">
            google/muril-base-cased
          </span>
          <span className="text-slate-400 text-[11px]">
            {"BERT Architecture (d_model = 768)"}
          </span>
        </div>

        <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 block text-[10px] uppercase">Layer Depth Pruning</span>
          <span className="text-sm font-bold text-emerald-400 block my-1">
            4 Layers (from 12)
          </span>
          <span className="text-slate-400 text-[11px]">
            66.7% reduction in FLOPs
          </span>
        </div>

        <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 block text-[10px] uppercase">LoRA Rank & Projections</span>
          <span className="text-sm font-bold text-teal-400 block my-1">
            r = 8, α = 16
          </span>
          <span className="text-slate-400 text-[11px]">
            Target: query & value
          </span>
        </div>

        <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 block text-[10px] uppercase">Trainable Weight Ratio</span>
          <span className="text-sm font-bold text-cyan-400 block my-1">
            0.056% (100,611 params)
          </span>
          <span className="text-slate-400 text-[11px]">
            Frozen: 180,853,248 params
          </span>
        </div>
      </div>

      {/* Detailed Research Modules */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Distillation Protocol */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Zap className="w-4 h-4 text-emerald-400" />
            <span>Knowledge Distillation Objective</span>
          </span>

          <p className="text-xs text-slate-300 leading-relaxed font-sans">
            The student model is trained on a composite loss balancing task ground truth and soft teacher posteriors:
          </p>

          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs text-emerald-300">
            {"L_total = α · L_CE(z_s, y) + β · T² · L_KL(σ(z_s/T) || σ(z_t/T))"}
          </div>

          <div className="space-y-2 text-xs text-slate-400 font-mono">
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Distillation Temperature ($T$)</span>
              <span className="text-slate-200">2.0</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Hard CE Weight ($\alpha$)</span>
              <span className="text-slate-200">0.5</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800">
              <span>Soft KL Weight ($\beta$)</span>
              <span className="text-slate-200">0.5</span>
            </div>
            <div className="flex justify-between py-1">
              <span>Teacher Logit Detachment</span>
              <span className="text-emerald-400">Enabled (Fixed Target)</span>
            </div>
          </div>
        </div>

        {/* Vocabulary & Tokenizer Screening */}
        <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Tokenizer Alignment & Vocabulary Integrity</span>
          </span>

          <p className="text-xs text-slate-300 leading-relaxed font-sans">
            Phase 8 empirical screening revealed why smaller multilingual student tokenizers fail on Indic text:
          </p>

          <div className="space-y-2.5 text-xs font-mono">
            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center">
              <div>
                <span className="text-emerald-400 font-bold block">MuRIL-base (Our Student SLM)</span>
                <span className="text-slate-400 text-[11px]">197,258 vocab size · WordPiece</span>
              </div>
              <div className="text-right">
                <span className="text-emerald-300 font-bold block">0.00% UNK</span>
                <span className="text-[10px] text-slate-400">Fertility: 1.50 tok/word</span>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center opacity-70">
              <div>
                <span className="text-slate-300 font-bold block">DistilmBERT (Rejected Candidate)</span>
                <span className="text-slate-400 text-[11px]">119,547 vocab size</span>
              </div>
              <div className="text-right">
                <span className="text-rose-400 font-bold block">1.82% UNK</span>
                <span className="text-[10px] text-rose-300">31.2% Truncation at 64</span>
              </div>
            </div>
          </div>

          <p className="text-[11px] text-slate-400 font-sans">
            By inheriting the exact tokenizer and embedding matrix of the teacher, the student SLM maintains zero UNK fragmentation across Devanagari and Bengali characters.
          </p>
        </div>
      </div>
    </div>
  );
};
