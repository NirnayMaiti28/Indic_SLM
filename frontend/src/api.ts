import type {
  HealthResponse,
  ModelInfoResponse,
  PredictResponse,
  BatchPredictResponse,
  BenchmarkResponse,
  FilePredictionResponse,
} from './types';

const API_BASE = '/api/v1';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE}/model/info`);
  if (!res.ok) {
    throw new Error(`Failed to load model info: ${res.statusText}`);
  }
  return res.json();
}

export async function predictStance(
  text: string,
  explain: boolean = false
): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, explain }),
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || `Prediction failed: ${res.statusText}`);
  }
  return res.json();
}

export async function predictBatch(
  texts: string[],
  explain: boolean = false
): Promise<BatchPredictResponse> {
  const res = await fetch(`${API_BASE}/predict/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts, explain }),
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || `Batch prediction failed: ${res.statusText}`);
  }
  return res.json();
}

export async function predictFile(
  file: File,
  textColumn: string = 'text'
): Promise<FilePredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('text_column', textColumn);

  const res = await fetch(`${API_BASE}/predict/file`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || `File processing failed: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchBenchmarks(): Promise<BenchmarkResponse> {
  const res = await fetch(`${API_BASE}/benchmarks`);
  if (!res.ok) {
    throw new Error(`Failed to load benchmarks: ${res.statusText}`);
  }
  return res.json();
}
