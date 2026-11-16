export interface TokenAttribution {
  token: string;
  attribution: number;
}

export interface PredictResponse {
  text: string;
  clean_text?: string;
  predicted_label: 'FAVOUR' | 'AGAINST' | 'NONE' | null;
  confidence: number | null;
  probabilities: {
    FAVOUR: number;
    AGAINST: number;
    NONE: number;
  } | null;
  language: string | null;
  script: string | null;
  is_code_mixed: boolean;
  has_negation: boolean;
  truncated: boolean;
  latency_ms: number;
  device: string;
  explanations: TokenAttribution[] | null;
  error?: string | null;
}

export interface BatchPredictResponse {
  results: PredictResponse[];
  total: number;
  latency_ms: number;
  throughput_items_per_sec: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_name: string;
  checkpoint: string;
  device: string;
  version: string;
}

export interface ModelInfoResponse {
  name: string;
  architecture: string;
  base_encoder: string;
  layers: number;
  pooling: string;
  max_sequence_length: number;
  lora_rank: number;
  lora_alpha: number;
  total_parameters: number;
  trainable_parameters: number;
  classes: string[];
  target_claim: string;
  supported_languages: string[];
  supported_scripts: string[];
  distillation_teacher: string;
  distillation_hyperparameters: {
    temperature: number;
    alpha_ce: number;
    beta_kl: number;
    t_squared_scaling: boolean;
  };
}

export interface BenchmarkModel {
  name: string;
  category: string;
  parameters: string;
  trainable_parameters: string;
  macro_f1: number;
  accuracy: number;
  hindi_f1: number;
  bengali_f1: number;
  latency_ms: number;
  notes: string;
}

export interface DatasetMetrics {
  total_samples: number;
  claim: string;
  splits: {
    train: { total: number; FAVOUR: number; AGAINST: number; NONE: number };
    val: { total: number; FAVOUR: number; AGAINST: number; NONE: number };
    test: { total: number; FAVOUR: number; AGAINST: number; NONE: number };
  };
  languages: Record<string, number>;
  scripts: Record<string, number>;
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
  per_class: Record<
    string,
    { precision: number; recall: number; f1: number; support: number }
  >;
}

export interface LatencyBenchmark {
  batch_size: number;
  device: string;
  p50_ms: number;
  p95_ms: number;
  throughput: number;
}

export interface BenchmarkResponse {
  models: BenchmarkModel[];
  datasets: DatasetMetrics;
  confusion_matrix: ConfusionMatrixData;
  latency_benchmarks: LatencyBenchmark[];
}

export interface FilePredictionResponse {
  filename: string;
  total_records: number;
  latency_ms: number;
  throughput_items_per_sec: number;
  class_distribution: Record<string, number>;
  language_distribution: Record<string, number>;
  records: Array<Record<string, any>>;
}
