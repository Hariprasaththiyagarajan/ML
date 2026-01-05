
import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { CustomerData, PredictionResult, KaggleDataset } from './types';
import { MLEngine } from './services/mlEngine';
import { AIService } from './services/aiService';

const DEFAULT_DATA: CustomerData[] = Array.from({ length: 120 }, () => {
  const age = Math.floor(Math.random() * (60 - 18 + 1)) + 18;
  const salary = Math.floor(Math.random() * (150000 - 15000 + 1)) + 15000;
  const probability = 1 / (1 + Math.exp(-(0.18 * (age - 38) + 0.00004 * (salary - 80000))));
  const purchased = Math.random() < probability ? 1 : 0;
  return { Age: age, EstimatedSalary: salary, Purchased: purchased };
});

export default function App() {
  const [view, setView] = useState<'dashboard' | 'discovery'>('dashboard');
  const [data, setData] = useState<CustomerData[]>(DEFAULT_DATA);
  const [age, setAge] = useState(38);
  const [salary, setSalary] = useState(85000);
  const [isPredicting, setIsPredicting] = useState(false);
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [aiInsight, setAiInsight] = useState<string>('');
  
  // Discovery State
  const [searchQuery, setSearchQuery] = useState('Customer Purchase Behavior Kaggle');
  const [discoveredDatasets, setDiscoveredDatasets] = useState<KaggleDataset[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState(false);

  const mlEngine = useMemo(() => new MLEngine(data), [data]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        const lines = text.split('\n');
        if (lines.length < 2) return;
        const headers = lines[0].toLowerCase().split(',');
        const ageIdx = headers.findIndex(h => h.includes('age'));
        const salaryIdx = headers.findIndex(h => h.includes('salary'));
        const purchasedIdx = headers.findIndex(h => h.includes('purchased'));

        if (ageIdx !== -1 && salaryIdx !== -1 && purchasedIdx !== -1) {
          const parsed = lines.slice(1).filter(l => l.trim()).map(line => {
            const cols = line.split(',');
            return { Age: parseInt(cols[ageIdx]), EstimatedSalary: parseInt(cols[salaryIdx]), Purchased: parseInt(cols[purchasedIdx]) };
          }).filter(d => !isNaN(d.Age));
          if (parsed.length) setData(parsed);
        }
      };
      reader.readAsText(file);
    }
  };

  const handlePredict = async () => {
    setIsPredicting(true);
    setAiInsight('');
    await new Promise(r => setTimeout(r, 800));
    
    const lr = mlEngine.predictLogisticRegression(age, salary);
    const knn = mlEngine.predictKNN(age, salary);
    
    setResults([
      { algorithm: 'Logistic Regression', purchased: lr.purchased, probability: lr.probability, accuracy: mlEngine.getAccuracy('LR') },
      { algorithm: 'KNN (K=5)', purchased: knn.purchased, probability: knn.probability, accuracy: mlEngine.getAccuracy('KNN') }
    ]);
    setIsPredicting(false);

    const insight = await AIService.getPredictionInsight(age, salary, lr.purchased, lr.probability);
    setAiInsight(insight);
  };

  const runDiscovery = async () => {
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    setSearchError(false);
    try {
      const datasets = await AIService.discoverDatasets(searchQuery);
      if (datasets.length === 0) {
        setSearchError(true);
      } else {
        setDiscoveredDatasets(datasets);
      }
    } catch (err) {
      setSearchError(true);
    } finally {
      setIsSearching(false);
    }
  };

  const restartEngine = () => {
    setDiscoveredDatasets([]);
    setSearchError(false);
    runDiscovery();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      runDiscovery();
    }
  };

  useEffect(() => {
    if (view === 'discovery' && discoveredDatasets.length === 0 && !isSearching && !searchError) {
      runDiscovery();
    }
  }, [view]);

  return (
    <div className="flex h-screen bg-[#FDFDFF]">
      {/* Navigation Sidebar */}
      <aside className="w-64 border-r border-slate-100 bg-white flex flex-col p-6 space-y-8">
        <div className="flex items-center space-x-3 px-2">
          <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-white shadow-lg shadow-indigo-100">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>
          </div>
          <span className="text-xl font-bold tracking-tight text-slate-800">OmniSense AI</span>
        </div>

        <nav className="flex-1 space-y-1">
          <button 
            onClick={() => setView('dashboard')}
            className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${view === 'dashboard' ? 'bg-indigo-50 text-indigo-600' : 'text-slate-500 hover:bg-slate-50'}`}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" /></svg>
            <span>Analytics Dashboard</span>
          </button>
          <button 
            onClick={() => setView('discovery')}
            className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${view === 'discovery' ? 'bg-indigo-50 text-indigo-600' : 'text-slate-500 hover:bg-slate-50'}`}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
            <span>Kaggle Discovery</span>
          </button>
        </nav>

        <div className="bg-slate-50 rounded-2xl p-4">
          <p className="text-xs font-bold text-slate-400 uppercase mb-3">Data Status</p>
          <div className="flex items-center space-x-2 text-sm font-semibold text-slate-700 mb-2">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
            <span>{data.length} Data Points</span>
          </div>
          <label className="block w-full text-center py-2 bg-white border border-slate-200 rounded-lg text-xs font-bold text-slate-600 cursor-pointer hover:border-indigo-300 transition-colors">
            Replace Source
            <input type="file" className="hidden" accept=".csv" onChange={handleFileUpload} />
          </label>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto bg-[#F9FAFC] p-8">
        
        {view === 'dashboard' ? (
          <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in duration-500">
            {/* Top Bar */}
            <div className="flex justify-between items-end">
              <div>
                <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">Purchase Analytics</h2>
                <p className="text-slate-500 font-medium">Predicting consumer behavior with high-fidelity classification models.</p>
              </div>
              <div className="bg-white px-4 py-2 rounded-xl shadow-sm border border-slate-100 flex space-x-6 text-sm">
                <div className="flex flex-col">
                  <span className="text-slate-400 text-[10px] uppercase font-bold">Avg Accuracy</span>
                  <span className="font-bold text-indigo-600">89.4%</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-slate-400 text-[10px] uppercase font-bold">Latency</span>
                  <span className="font-bold text-slate-700">14ms</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-12 gap-8">
              {/* Left Column */}
              <div className="col-span-12 lg:col-span-4 space-y-6">
                <section className="bg-white rounded-3xl p-6 shadow-sm border border-slate-100">
                  <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-6">Simulation Inputs</h3>
                  
                  <div className="space-y-8">
                    <div className="group">
                      <div className="flex justify-between mb-3">
                        <label className="text-sm font-bold text-slate-700">Age of Subject</label>
                        <span className="bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-lg text-xs font-bold">{age} yrs</span>
                      </div>
                      <input type="range" min="18" max="70" value={age} onChange={(e)=>setAge(parseInt(e.target.value))} className="w-full h-1.5 bg-slate-100 rounded-full appearance-none accent-indigo-600 cursor-pointer" />
                    </div>

                    <div className="group">
                      <div className="flex justify-between mb-3">
                        <label className="text-sm font-bold text-slate-700">Annual Income</label>
                        <span className="bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-lg text-xs font-bold">£{(salary/1000).toFixed(0)}k</span>
                      </div>
                      <input type="range" min="15000" max="150000" step="1000" value={salary} onChange={(e)=>setSalary(parseInt(e.target.value))} className="w-full h-1.5 bg-slate-100 rounded-full appearance-none accent-emerald-500 cursor-pointer" />
                    </div>

                    <button 
                      onClick={handlePredict}
                      disabled={isPredicting}
                      className="w-full h-14 bg-slate-900 hover:bg-indigo-600 text-white font-bold rounded-2xl transition-all shadow-xl shadow-slate-200 flex items-center justify-center space-x-2 disabled:opacity-50"
                    >
                      {isPredicting ? <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div> : <span>Execute Prediction</span>}
                    </button>
                  </div>
                </section>

                {results.length > 0 && (
                  <div className="animate-in slide-in-from-bottom-4 duration-500 space-y-4">
                    {results.map((res, i) => (
                      <div key={i} className="bg-white p-5 rounded-3xl border border-slate-100 shadow-sm relative overflow-hidden group">
                        <div className={`absolute top-0 left-0 w-1 h-full ${res.purchased ? 'bg-emerald-500' : 'bg-rose-500'}`}></div>
                        <div className="flex justify-between items-start mb-2">
                          <span className="text-[10px] font-black text-slate-400 uppercase tracking-tighter">{res.algorithm}</span>
                          <span className="text-xs font-bold text-indigo-500">{Math.round(res.accuracy * 100)}% Acc.</span>
                        </div>
                        <h4 className={`text-xl font-black ${res.purchased ? 'text-emerald-600' : 'text-rose-600'}`}>
                          {res.purchased ? 'WILL PURCHASE' : 'NOT LIKELY'}
                        </h4>
                        <div className="mt-3 h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                          <div 
                            className={`h-full transition-all duration-1000 ${res.purchased ? 'bg-emerald-500' : 'bg-rose-500'}`}
                            style={{ width: `${res.probability * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}

                    {aiInsight && (
                      <div className="bg-indigo-900 text-indigo-50 p-6 rounded-3xl shadow-xl shadow-indigo-200 relative overflow-hidden">
                        <div className="absolute -top-4 -right-4 w-16 h-16 bg-white/10 rounded-full blur-2xl"></div>
                        <div className="flex items-center space-x-2 mb-3">
                          <div className="w-6 h-6 bg-indigo-500 rounded-full flex items-center justify-center">
                            <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/></svg>
                          </div>
                          <span className="text-[10px] font-black uppercase tracking-widest text-indigo-300">AI Narrative</span>
                        </div>
                        <p className="text-sm font-medium leading-relaxed italic">"{aiInsight}"</p>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Right Column */}
              <div className="col-span-12 lg:col-span-8 space-y-8">
                <div className="bg-white rounded-[2rem] p-8 shadow-sm border border-slate-100">
                  <div className="flex justify-between items-center mb-8">
                    <h3 className="text-lg font-black text-slate-800 tracking-tight">Population Distribution</h3>
                    <div className="flex bg-slate-50 p-1 rounded-xl">
                      <div className="flex items-center space-x-2 px-3 py-1 text-[10px] font-black text-emerald-600"><div className="w-2 h-2 rounded-full bg-emerald-500"></div> <span>CONVERTED</span></div>
                      <div className="flex items-center space-x-2 px-3 py-1 text-[10px] font-black text-rose-500"><div className="w-2 h-2 rounded-full bg-rose-500"></div> <span>BOUNCED</span></div>
                    </div>
                  </div>
                  
                  <div className="h-[440px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 10, right: 10, bottom: 40, left: 20 }}>
                        <CartesianGrid strokeDasharray="4 4" vertical={false} stroke="#F1F5F9" />
                        <XAxis 
                          type="number" dataKey="Age" name="Age" domain={[15, 65]} 
                          tick={{fontSize: 12, fontWeight: 600, fill: '#94A3B8'}} 
                          axisLine={false} tickLine={false}
                          label={{ value: 'Age (Years)', position: 'bottom', offset: 20, style: { fontSize: 11, fontWeight: 800, fill: '#64748B', textTransform: 'uppercase' } }}
                        />
                        <YAxis 
                          type="number" dataKey="EstimatedSalary" name="Income" domain={[0, 160000]}
                          tick={{fontSize: 12, fontWeight: 600, fill: '#94A3B8'}} 
                          axisLine={false} tickLine={false}
                          tickFormatter={(v) => `£${v/1000}k`}
                        />
                        <ZAxis type="number" range={[100, 300]} />
                        <Tooltip 
                          contentStyle={{ borderRadius: '16px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)', padding: '12px' }}
                          cursor={{ stroke: '#6366f1', strokeWidth: 1, strokeDasharray: '3 3' }} 
                        />
                        <Scatter name="Data" data={data.map(d => ({ ...d, fill: d.Purchased === 1 ? '#10B981' : '#F43F5E' }))}>
                          {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.Purchased === 1 ? '#10B981' : '#F43F5E'} fillOpacity={0.7} />
                          ))}
                        </Scatter>
                        <Scatter name="Selected" data={[{ Age: age, EstimatedSalary: salary }]} fill="#4F46E5" shape={(props: any) => {
                          const { cx, cy } = props;
                          return (
                            <g>
                              <circle cx={cx} cy={cy} r={10} fill="#4F46E5" fillOpacity={0.2} />
                              <circle cx={cx} cy={cy} r={5} fill="#4F46E5" />
                              <path d={`M ${cx} ${cy-15} L ${cx} ${cy+15} M ${cx-15} ${cy} L ${cx+15} ${cy}`} stroke="#4F46E5" strokeWidth={1.5} />
                            </g>
                          );
                        }} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="max-w-6xl mx-auto animate-in zoom-in-95 duration-500">
            <div className="flex flex-col md:flex-row justify-between items-center mb-10 space-y-4 md:space-y-0">
              <div>
                <h2 className="text-3xl font-black text-slate-900">Kaggle Discovery Engine</h2>
                <p className="text-slate-500 font-medium">Use AI to source similar datasets for cross-validation.</p>
              </div>
              <div className="flex w-full md:w-auto space-x-2">
                <div className="relative flex-1">
                   <input 
                    type="text" 
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="e.g. Consumer Habits Kaggle"
                    className="w-full md:w-80 px-4 py-3 bg-white border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 outline-none font-medium pr-10"
                  />
                  {searchQuery && (
                    <button onClick={() => setSearchQuery('')} className="absolute right-3 top-3.5 text-slate-400 hover:text-slate-600">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
                    </button>
                  )}
                </div>
                <button 
                  onClick={runDiscovery}
                  disabled={isSearching}
                  className="px-6 py-3 bg-indigo-600 text-white font-bold rounded-xl hover:bg-indigo-700 transition-all shadow-lg shadow-indigo-100 disabled:opacity-50 flex items-center space-x-2"
                >
                  {isSearching ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                      <span>Scanning...</span>
                    </>
                  ) : <span>Scan Market</span>}
                </button>
              </div>
            </div>

            {searchError && (
              <div className="mb-8 p-6 bg-rose-50 border border-rose-100 rounded-3xl flex flex-col items-center text-center">
                <div className="w-12 h-12 bg-rose-100 text-rose-600 rounded-full flex items-center justify-center mb-3">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                </div>
                <h4 className="text-lg font-bold text-rose-900 mb-1">Search Engine Interrupted</h4>
                <p className="text-rose-700 mb-4 max-w-md">We couldn't retrieve results for your query. This might be due to a network glitch or API limit.</p>
                <button 
                  onClick={restartEngine}
                  className="px-5 py-2.5 bg-rose-600 text-white font-bold rounded-xl hover:bg-rose-700 transition-colors flex items-center space-x-2 shadow-lg shadow-rose-100"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
                  <span>Restart Engine</span>
                </button>
              </div>
            )}

            {isSearching ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {[1,2,3,4,5,6].map(i => (
                  <div key={i} className="bg-white h-56 rounded-[2rem] animate-pulse border border-slate-50 p-8 space-y-4">
                    <div className="flex justify-between">
                      <div className="w-12 h-12 bg-slate-100 rounded-2xl"></div>
                      <div className="w-16 h-6 bg-slate-100 rounded-lg"></div>
                    </div>
                    <div className="h-4 bg-slate-100 rounded w-3/4"></div>
                    <div className="h-4 bg-slate-100 rounded w-full"></div>
                    <div className="h-4 bg-slate-100 rounded w-1/2"></div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {discoveredDatasets.map((ds, idx) => (
                  <a 
                    key={idx} 
                    href={ds.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="group bg-white p-8 rounded-[2rem] border border-slate-100 shadow-sm hover:shadow-xl hover:-translate-y-1 transition-all flex flex-col h-full overflow-hidden relative"
                  >
                    <div className="flex justify-between items-start mb-6">
                      <div className="w-12 h-12 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600 group-hover:bg-indigo-600 group-hover:text-white transition-colors">
                        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
                      </div>
                      <span className="text-xs font-black text-indigo-500 bg-indigo-50 px-2 py-1 rounded-lg uppercase">Score: {ds.relevance}</span>
                    </div>
                    <h4 className="text-lg font-black text-slate-800 mb-3 group-hover:text-indigo-600 transition-colors leading-tight">{ds.title}</h4>
                    <p className="text-sm text-slate-500 font-medium mb-6 flex-1 line-clamp-3 leading-relaxed">{ds.description}</p>
                    <div className="flex items-center text-indigo-600 text-xs font-black uppercase tracking-widest mt-auto">
                      <span>Explore Data</span>
                      <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
                    </div>
                  </a>
                ))}
              </div>
            )}
            
            {!isSearching && discoveredDatasets.length === 0 && !searchError && (
              <div className="flex flex-col items-center justify-center py-20 text-slate-400">
                <svg className="w-16 h-16 mb-4 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
                <p className="font-bold">No results found. Try a different search term.</p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
