
export interface CustomerData {
  Age: number;
  EstimatedSalary: number;
  Purchased: number;
}

export interface PredictionResult {
  algorithm: string;
  purchased: boolean;
  probability: number;
  accuracy: number;
}

export interface KaggleDataset {
  title: string;
  url: string;
  description: string;
  relevance: string;
}

export interface ScaleParams {
  mean: [number, number];
  std: [number, number];
}
