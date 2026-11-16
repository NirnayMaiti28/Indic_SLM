import React, { useState } from 'react';
import {
  FileSpreadsheet,
  Upload,
  Download,
  CheckCircle2,
  AlertCircle,
  Clock,
  Zap,
  Filter,
} from 'lucide-react';
import type { FilePredictionResponse } from '../types';
import { predictFile } from '../api';

export const BatchView: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [textColumn, setTextColumn] = useState('text');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FilePredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filterStance, setFilterStance] = useState<string>('ALL');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUploadAndRun = async () => {
    if (!selectedFile) {
      setError('Please select a CSV, TSV, or JSONL file to analyze.');
      return;
    }
    setLoading(true);
    setError(null);

    try {
      const res = await predictFile(selectedFile, textColumn);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Failed to process file.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadCSV = () => {
    if (!result || !result.records.length) return;

    const headers = Object.keys(result.records[0]);
    const csvRows = [
      headers.join(','),
      ...result.records.map((r) =>
        headers
          .map((h) => {
            const val = r[h] === undefined || r[h] === null ? '' : String(r[h]);
            return `"${val.replace(/"/g, '""')}"`;
          })
          .join(',')
      ),
    ];

    const blob = new Blob([csvRows.join('\n')], {
      type: 'text/csv;charset=utf-8;',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute(
      'download',
      `classified_${result.filename.replace(/\.[^/.]+$/, '')}.csv`
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredRecords = result
    ? result.records.filter((r) =>
        filterStance === 'ALL' ? true : r.predicted_stance === filterStance
      )
    : [];

  return (
    <div className="space-y-8 py-2">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <FileSpreadsheet className="w-6 h-6 text-emerald-400" />
          <span>Batch Stance Processing</span>
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Upload social media datasets in CSV, TSV, or JSONL format for high-throughput batch classification.
        </p>
      </div>

      {/* Upload Box */}
      <div className="p-6 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-4">
        <div className="border-2 border-dashed border-slate-700 hover:border-emerald-500/60 rounded-xl p-8 text-center transition-all bg-slate-950/40 cursor-pointer relative">
          <input
            type="file"
            accept=".csv,.tsv,.jsonl,.json"
            onChange={handleFileChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          <div className="flex flex-col items-center space-y-2 pointer-events-none">
            <Upload className="w-8 h-8 text-slate-400" />
            <span className="text-sm font-medium text-slate-200">
              {selectedFile ? selectedFile.name : 'Choose or drag a CSV / JSONL dataset here'}
            </span>
            <span className="text-xs text-slate-400 font-mono">
              Supported: .csv, .tsv, .jsonl (Max 500 rows per batch)
            </span>
          </div>
        </div>

        {/* Configuration Row */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
          <div className="flex items-center gap-2 text-xs font-mono w-full sm:w-auto">
            <span className="text-slate-400">Target Text Column:</span>
            <input
              type="text"
              value={textColumn}
              onChange={(e) => setTextColumn(e.target.value)}
              placeholder="text"
              className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-slate-200 text-xs w-28 focus:outline-none focus:border-emerald-500 font-mono"
            />
          </div>

          <button
            onClick={handleUploadAndRun}
            disabled={loading || !selectedFile}
            className="w-full sm:w-auto px-6 py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-slate-950 font-semibold text-sm transition-all shadow-md shadow-emerald-500/20 flex items-center justify-center gap-2 cursor-pointer font-sans"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                <span>Processing Batch...</span>
              </>
            ) : (
              <>
                <FileSpreadsheet className="w-4 h-4" />
                <span>Run Batch Classification</span>
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

      {/* Batch Results */}
      {result && (
        <div className="space-y-6">
          {/* Summary Metric Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 font-mono text-xs">
              <span className="text-slate-400 block text-[10px] uppercase">Records Processed</span>
              <span className="text-2xl font-bold text-white my-1 block">
                {result.total_records}
              </span>
              <span className="text-slate-400 text-[11px]">
                File: {result.filename}
              </span>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 font-mono text-xs">
              <span className="text-slate-400 block text-[10px] uppercase">Batch Latency</span>
              <span className="text-2xl font-bold text-emerald-400 my-1 block">
                {result.latency_ms} ms
              </span>
              <span className="text-slate-400 text-[11px]">
                {result.throughput_items_per_sec} items/sec
              </span>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 font-mono text-xs">
              <span className="text-slate-400 block text-[10px] uppercase">Class Breakdown</span>
              <div className="flex items-center gap-2 mt-2">
                <span className="text-emerald-400 font-semibold">
                  F: {result.class_distribution?.FAVOUR || 0}
                </span>
                <span className="text-rose-400 font-semibold">
                  A: {result.class_distribution?.AGAINST || 0}
                </span>
                <span className="text-slate-400 font-semibold">
                  N: {result.class_distribution?.NONE || 0}
                </span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 font-mono text-xs flex items-center justify-between">
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Export</span>
                <span className="text-xs text-slate-300">Download with predictions</span>
              </div>
              <button
                onClick={handleDownloadCSV}
                className="px-3 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-semibold text-xs flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                <span>CSV</span>
              </button>
            </div>
          </div>

          {/* Table Header and Filter */}
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
              <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                Annotated Results Preview ({filteredRecords.length} of {result.total_records} rows)
              </span>

              {/* Filter Pills */}
              <div className="flex items-center gap-1.5 text-xs font-mono">
                <Filter className="w-3.5 h-3.5 text-slate-500" />
                {['ALL', 'FAVOUR', 'AGAINST', 'NONE'].map((stance) => (
                  <button
                    key={stance}
                    onClick={() => setFilterStance(stance)}
                    className={`px-2 py-0.5 rounded text-[11px] cursor-pointer ${
                      filterStance === stance
                        ? 'bg-slate-700 text-white font-semibold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {stance}
                  </button>
                ))}
              </div>
            </div>

            {/* Scrollable Table */}
            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-slate-950 text-slate-400 text-[11px] uppercase border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-3">#</th>
                    <th className="py-2.5 px-3 w-1/2 font-sans">Text Content</th>
                    <th className="py-2.5 px-3">Predicted Stance</th>
                    <th className="py-2.5 px-3">Confidence</th>
                    <th className="py-2.5 px-3">Language</th>
                    <th className="py-2.5 px-3">Negation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 bg-slate-900/40">
                  {filteredRecords.slice(0, 50).map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-2 px-3 text-slate-400">{idx + 1}</td>
                      <td className="py-2 px-3 text-slate-200 font-sans text-xs max-w-xs truncate">
                        {row[textColumn] || row.text || '—'}
                      </td>
                      <td className="py-2 px-3">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                            row.predicted_stance === 'FAVOUR'
                              ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-800/60'
                              : row.predicted_stance === 'AGAINST'
                              ? 'bg-rose-950/80 text-rose-300 border border-rose-800/60'
                              : 'bg-slate-800 text-slate-300 border border-slate-700'
                          }`}
                        >
                          {row.predicted_stance}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-slate-300">
                        {row.confidence ? (row.confidence * 100).toFixed(1) + '%' : '—'}
                      </td>
                      <td className="py-2 px-3 text-slate-400">
                        {row.detected_language || '—'}
                      </td>
                      <td className="py-2 px-3">
                        {row.negation_detected ? (
                          <span className="text-emerald-400 font-semibold">Yes</span>
                        ) : (
                          <span className="text-slate-400">No</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {filteredRecords.length > 50 && (
              <p className="text-center text-[11px] font-mono text-slate-400 pt-1">
                Showing top 50 records in preview. Download CSV for the complete {result.total_records} records.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
