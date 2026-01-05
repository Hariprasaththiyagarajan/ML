
import { GoogleGenAI, Type } from "@google/genai";
import { KaggleDataset } from "../types";

export class AIService {
  static async discoverDatasets(query: string): Promise<KaggleDataset[]> {
    // Initialize inside the method to ensure fresh API key context
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    
    try {
      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: `Find relevant Kaggle or public datasets for: ${query}. 
        Focus on customer behavior, purchase prediction, and demographic data. 
        Return exactly a JSON array of objects with title, url, description, and relevance score (out of 10).`,
        config: {
          tools: [{ googleSearch: {} }],
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                title: { type: Type.STRING },
                url: { type: Type.STRING },
                description: { type: Type.STRING },
                relevance: { type: Type.STRING }
              },
              required: ["title", "url", "description", "relevance"]
            }
          }
        }
      });

      const text = response.text;
      if (!text) return [];
      
      // Extract grounding chunks for extra verification if needed
      const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
      console.log("Grounding Chunks:", chunks);

      return JSON.parse(text);
    } catch (e) {
      console.error("Discovery Engine Error:", e);
      // Fallback or rethrow for UI handling
      return [];
    }
  }

  static async getPredictionInsight(age: number, salary: number, result: boolean, probability: number): Promise<string> {
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    try {
      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: `Explain in 2 short sentences why a customer who is ${age} years old and earns £${salary} was predicted to ${result ? 'PURCHASE' : 'NOT PURCHASE'} a product with ${Math.round(probability * 100)}% confidence, based on common market demographics.`,
      });
      return response.text || "Insight generation paused.";
    } catch (e) {
      return "Unable to generate AI narrative at this time.";
    }
  }
}
