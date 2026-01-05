
import { CustomerData, ScaleParams } from '../types';

export class MLEngine {
  private trainingData: CustomerData[] = [];
  private scaleParams: ScaleParams = { mean: [0, 0], std: [1, 1] };

  // Logistic Regression Weights
  private lrWeights: number[] = [0, 0];
  private lrBias: number = 0;

  constructor(data: CustomerData[]) {
    this.trainingData = data;
    this.calculateScaling();
    this.trainLogisticRegression();
  }

  private calculateScaling() {
    const ages = this.trainingData.map(d => d.Age);
    const salaries = this.trainingData.map(d => d.EstimatedSalary);
    
    const meanAge = ages.reduce((a, b) => a + b, 0) / ages.length;
    const meanSalary = salaries.reduce((a, b) => a + b, 0) / salaries.length;
    
    const stdAge = Math.sqrt(ages.map(x => Math.pow(x - meanAge, 2)).reduce((a, b) => a + b, 0) / ages.length);
    const stdSalary = Math.sqrt(salaries.map(x => Math.pow(x - meanSalary, 2)).reduce((a, b) => a + b, 0) / salaries.length);
    
    this.scaleParams = {
      mean: [meanAge, meanSalary],
      std: [stdAge || 1, stdSalary || 1]
    };
  }

  private scale(age: number, salary: number): [number, number] {
    return [
      (age - this.scaleParams.mean[0]) / this.scaleParams.std[0],
      (salary - this.scaleParams.mean[1]) / this.scaleParams.std[1]
    ];
  }

  private trainLogisticRegression() {
    // Simple Gradient Descent for Logistic Regression
    const learningRate = 0.1;
    const iterations = 1000;
    const m = this.trainingData.length;
    
    let w1 = 0, w2 = 0, b = 0;
    
    const scaledData = this.trainingData.map(d => ({
      ...d,
      scaled: this.scale(d.Age, d.EstimatedSalary)
    }));

    for (let i = 0; i < iterations; i++) {
      let dw1 = 0, dw2 = 0, db = 0;
      
      for (const d of scaledData) {
        const z = w1 * d.scaled[0] + w2 * d.scaled[1] + b;
        const p = 1 / (1 + Math.exp(-z));
        const diff = p - d.Purchased;
        
        dw1 += diff * d.scaled[0];
        dw2 += diff * d.scaled[1];
        db += diff;
      }
      
      w1 -= (learningRate / m) * dw1;
      w2 -= (learningRate / m) * dw2;
      b -= (learningRate / m) * db;
    }
    
    this.lrWeights = [w1, w2];
    this.lrBias = b;
  }

  public predictLogisticRegression(age: number, salary: number): { purchased: boolean, probability: number } {
    const [sAge, sSalary] = this.scale(age, salary);
    const z = this.lrWeights[0] * sAge + this.lrWeights[1] * sSalary + this.lrBias;
    const prob = 1 / (1 + Math.exp(-z));
    return {
      purchased: prob >= 0.5,
      probability: prob
    };
  }

  public predictKNN(age: number, salary: number, k: number = 5): { purchased: boolean, probability: number } {
    const [sAge, sSalary] = this.scale(age, salary);
    
    const distances = this.trainingData.map(d => {
      const [dsAge, dsSalary] = this.scale(d.Age, d.EstimatedSalary);
      const dist = Math.sqrt(Math.pow(sAge - dsAge, 2) + Math.pow(sSalary - dsSalary, 2));
      return { dist, purchased: d.Purchased };
    });

    distances.sort((a, b) => a.dist - b.dist);
    const neighbors = distances.slice(0, k);
    const purchasedCount = neighbors.filter(n => n.purchased === 1).length;
    
    return {
      purchased: purchasedCount > k / 2,
      probability: purchasedCount / k
    };
  }

  public getAccuracy(algorithm: 'LR' | 'KNN'): number {
    let correct = 0;
    for (const d of this.trainingData) {
      const pred = algorithm === 'LR' 
        ? this.predictLogisticRegression(d.Age, d.EstimatedSalary).purchased
        : this.predictKNN(d.Age, d.EstimatedSalary).purchased;
      
      if ((pred ? 1 : 0) === d.Purchased) correct++;
    }
    return correct / this.trainingData.length;
  }
}
