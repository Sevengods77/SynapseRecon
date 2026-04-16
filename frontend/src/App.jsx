import React, { useState, useEffect } from 'react';
import { Network, Cpu, Layers, Image as ImageIcon, CheckCircle2, Activity, Play } from 'lucide-react';

const MOCK_RESULTS = [
  { piece_id: 1, target_x: 120, target_y: 340, rotation_deg: 45 },
  { piece_id: 2, target_x: 150, target_y: 360, rotation_deg: 12 },
  { piece_id: 3, target_x: 180, target_y: 290, rotation_deg: 88 },
  { piece_id: 4, target_x: 220, target_y: 350, rotation_deg: 104 },
  { piece_id: 5, target_x: 300, target_y: 410, rotation_deg: -15 },
];

const PIPELINE_STEPS = [
  { id: 1, name: "Initializing DINOv2 Extractors" },
  { id: 2, name: "Segmenting Nodes (Mask R-CNN / SAM)" },
  { id: 3, name: "Generating DINOv2 Feature Vectors" },
  { id: 4, name: "Constructing Spatial k-NN Graph" },
  { id: 5, name: "Calculating DiffAssemble Markov Chain" },
  { id: 6, name: "Finalizing Prediction Matrix" }
];

export default function App() {
  const [referenceFile, setReferenceFile] = useState(null);
  const [referencePreview, setReferencePreview] = useState(null);

  const [scatteredFile, setScatteredFile] = useState(null);
  const [scatteredPreview, setScatteredPreview] = useState(null);

  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [results, setResults] = useState(null);

  const handleFile = (file, type) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    if (type === 'ref') {
      setReferenceFile(file);
      setReferencePreview(url);
    } else {
      setScatteredFile(file);
      setScatteredPreview(url);
    }
  };

  useEffect(() => {
    return () => {
      if (referencePreview) URL.revokeObjectURL(referencePreview);
      if (scatteredPreview) URL.revokeObjectURL(scatteredPreview);
    };
  }, [referencePreview, scatteredPreview]);

  const handleReassemble = async () => {
    if (!referenceFile || !scatteredFile) {
      alert("Please upload both Intact Reference and Scattered Pieces images.");
      return;
    }

    setLoading(true);
    setResults(null);
    setCurrentStep(0);

    for (let i = 0; i < PIPELINE_STEPS.length; i++) {
      setCurrentStep(i);
      await new Promise(r => setTimeout(r, Math.random() * 600 + 800));
    }

    setLoading(false);
    setResults(MOCK_RESULTS);
  };

  return (
    <div className="min-h-screen relative selection:bg-indigo-500/30">
      
      {/* Subtle background glow */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[20%] w-[50%] h-[50%] bg-blue-500/10 blur-[120px] rounded-full mix-blend-screen"></div>
        <div className="absolute bottom-[-10%] right-[10%] w-[40%] h-[40%] bg-violet-600/10 blur-[150px] rounded-full mix-blend-screen"></div>
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-16 flex flex-col gap-12 font-sans text-slate-200">
        
        {/* Header Header */}
        <header className="flex flex-col items-center justify-center text-center space-y-5">
          <div className="inline-flex items-center justify-center p-4 rounded-3xl bg-white/5 border border-white/10 shadow-xl mb-2 backdrop-blur-md">
            <Network className="w-10 h-10 text-indigo-400" strokeWidth={1.5}/>
          </div>
          <h1 className="text-5xl md:text-6xl font-semibold tracking-tight text-white">
            Synapse<span className="text-gradient">Recon</span>
          </h1>
          <p className="text-slate-400 text-lg max-w-2xl font-light">
            Graph Diffusion Engine for Spatial Object Reassembly
          </p>
          
          <div className="flex items-center gap-4 mt-4">
            <div className="flex items-center gap-2 px-4 py-2 rounded-full glass-panel text-xs font-medium text-blue-300">
              <Activity className="w-4 h-4 animate-pulse" /> Live Inference
            </div>
            <div className="flex items-center gap-2 px-4 py-2 rounded-full glass-panel text-xs font-medium text-violet-300">
              <Cpu className="w-4 h-4" /> DINOv2 Pipeline
            </div>
          </div>
        </header>

        {/* File Upload Grid */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-4">
          
          {/* Reference */}
          <div className="glass-panel glass-panel-hover p-6 flex flex-col relative group">
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-lg font-medium text-slate-100 flex items-center gap-3">
                <span className="flex items-center justify-center w-7 h-7 rounded-full bg-slate-800 text-slate-300 text-xs border border-white/10">1</span>
                Intact Reference Target
              </h2>
              {referenceFile && <CheckCircle2 className="text-emerald-400 w-5 h-5" />}
            </div>

            <label className="flex-1 flex flex-col items-center justify-center border border-dashed border-white/15 rounded-2xl bg-black/20 hover:bg-black/40 cursor-pointer overflow-hidden transition-all min-h-[260px]">
              {referencePreview ? (
                <div className="relative w-full h-full min-h-[260px]">
                  <img src={referencePreview} alt="Reference" className="absolute inset-0 w-full h-full object-cover opacity-70" />
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-900/90 via-transparent flex flex-col justify-end p-5">
                    <p className="text-sm font-medium text-white truncate">{referenceFile.name}</p>
                  </div>
                </div>
              ) : (
                <div className="p-8 flex flex-col items-center text-center gap-4 text-slate-400">
                  <div className="p-4 rounded-full bg-white/5 border border-white/5">
                    <ImageIcon className="w-8 h-8 opacity-80" strokeWidth={1.5} />
                  </div>
                  <div>
                    <p className="text-sm text-slate-200 font-medium">Upload Ground Truth</p>
                    <p className="text-xs text-slate-500 mt-1">Extracts target shape tensor.</p>
                  </div>
                </div>
              )}
              <input type="file" className="hidden" accept="image/*" onChange={(e) => handleFile(e.target.files[0], 'ref')} />
            </label>
          </div>

          {/* Scattered */}
          <div className="glass-panel glass-panel-hover p-6 flex flex-col relative group">
            <div className="flex items-center justify-between mb-5">
              <h2 className="text-lg font-medium text-slate-100 flex items-center gap-3">
                <span className="flex items-center justify-center w-7 h-7 rounded-full bg-slate-800 text-slate-300 text-xs border border-white/10">2</span>
                Scattered Node State
              </h2>
              {scatteredFile && <CheckCircle2 className="text-emerald-400 w-5 h-5" />}
            </div>

            <label className="flex-1 flex flex-col items-center justify-center border border-dashed border-white/15 rounded-2xl bg-black/20 hover:bg-black/40 cursor-pointer overflow-hidden transition-all min-h-[260px]">
              {scatteredPreview ? (
                <div className="relative w-full h-full min-h-[260px]">
                  <img src={scatteredPreview} alt="Scattered" className="absolute inset-0 w-full h-full object-cover opacity-70" />
                  <div className="absolute inset-0 bg-gradient-to-t from-slate-900/90 via-transparent flex flex-col justify-end p-5">
                    <p className="text-sm font-medium text-white truncate">{scatteredFile.name}</p>
                  </div>
                </div>
              ) : (
                <div className="p-8 flex flex-col items-center text-center gap-4 text-slate-400">
                  <div className="p-4 rounded-full bg-white/5 border border-white/5">
                    <Layers className="w-8 h-8 opacity-80" strokeWidth={1.5} />
                  </div>
                  <div>
                    <p className="text-sm text-slate-200 font-medium">Upload Fragmented Pieces</p>
                    <p className="text-xs text-slate-500 mt-1">Identifies instances for graph nodes.</p>
                  </div>
                </div>
              )}
              <input type="file" className="hidden" accept="image/*" onChange={(e) => handleFile(e.target.files[0], 'scat')} />
            </label>
          </div>
        </section>

        {/* Global Action Button */}
        <section className="flex justify-center mt-6">
          <button 
            onClick={handleReassemble}
            disabled={loading || (!referenceFile && !scatteredFile)}
            className={`group relative inline-flex items-center gap-3 px-10 py-5 rounded-full text-[15px] font-semibold tracking-wide transition-all duration-300 ${
              loading 
                ? "bg-slate-800/80 text-slate-500 cursor-not-allowed border border-white/5" 
                : (!referenceFile && !scatteredFile)
                  ? "bg-slate-900/50 text-slate-600 cursor-not-allowed border border-white/5"
                  : "bg-slate-800 border border-white/10 text-white hover:bg-indigo-600 hover:border-indigo-500 hover:shadow-[0_0_20px_rgba(79,70,229,0.3)] animate-pulse-glow"
            }`}
          >
            {loading ? (
              <>
                <Activity className="w-5 h-5 animate-spin" />
                Executing Pipeline...
              </>
            ) : (
              <>
                <Play className={`w-5 h-5 ${(!referenceFile && !scatteredFile) ? 'text-slate-600' : 'text-indigo-400 group-hover:text-white transition-colors'}`} fill={(!referenceFile && !scatteredFile) ? "none" : "currentColor"}/>
                Initialize AI Reassembly
              </>
            )}
          </button>
        </section>

        {/* Animated Pipeline Simulation */}
        {loading && (
          <section className="glass-panel p-8 max-w-3xl mx-auto w-full animate-in fade-in slide-in-from-bottom-4 duration-500 relative overflow-hidden mt-6">
             {/* Progress bar background slider */}
             <div className="absolute top-0 left-0 w-full h-[3px] bg-slate-800/50">
               <div 
                 className="h-full bg-gradient-to-r from-blue-500 to-violet-500 transition-all duration-[800ms] ease-out"
                 style={{ width: `${((currentStep + 1) / PIPELINE_STEPS.length) * 100}%` }}
               />
             </div>

             <div className="flex items-start">
                <div className="flex flex-col gap-3 relative w-full pt-4">
                   {PIPELINE_STEPS.map((step, idx) => {
                     const isPast = idx < currentStep;
                     const isCurrent = idx === currentStep;
                     
                     return (
                        <div key={step.id} className={`flex items-center gap-4 transition-all duration-500 ${isCurrent ? 'opacity-100 translate-x-2' : isPast ? 'opacity-50' : 'opacity-20 translate-x-0'}`}>
                           <div className={`z-10 flex-shrink-0 w-[14px] h-[14px] rounded-full border-2 ${isCurrent ? 'border-indigo-400 bg-indigo-500 shadow-[0_0_12px_rgba(129,140,248,0.8)] animate-pulse' : isPast ? 'border-indigo-900 bg-slate-800' : 'border-slate-700 bg-transparent'}`} />
                           <span className={`text-[14px] font-medium ${isCurrent ? 'text-indigo-200' : 'text-slate-400'}`}>
                             {step.name}
                           </span>
                        </div>
                     )
                   })}
                   {/* Vertical Connector Line */}
                   <div className="absolute top-[28px] bottom-[16px] left-[6px] w-[2px] bg-slate-800/80 -z-0 rounded-full" />
                </div>
             </div>
          </section>
        )}

        {/* Results Data Table */}
        {results && !loading && (
          <section className="glass-panel p-8 border-t-[3px] border-t-emerald-500/80 shadow-[0_-15px_40px_rgba(16,185,129,0.05)] animate-in zoom-in-95 fade-in duration-500 mt-6">
            <div className="flex items-center gap-5 mb-8">
              <div className="p-3 bg-emerald-500/10 rounded-full text-emerald-400 border border-emerald-500/20">
                <CheckCircle2 className="w-8 h-8" strokeWidth={1.5}/>
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white tracking-tight">
                  Graph Successfully Solved
                </h2>
                <p className="text-slate-400 text-sm mt-1">Ready for downstream projection rendering.</p>
              </div>
            </div>

            <div className="overflow-hidden rounded-xl border border-white/5 bg-black/20">
              <table className="w-full text-left text-sm font-mono">
                <thead className="text-xs text-slate-500 uppercase bg-slate-900/50 border-b border-white/5">
                  <tr>
                    <th className="px-6 py-4 tracking-wider font-sans font-medium">Node ID</th>
                    <th className="px-6 py-4 tracking-wider text-blue-400 font-sans font-medium">Δ Target X</th>
                    <th className="px-6 py-4 tracking-wider text-blue-400 font-sans font-medium">Δ Target Y</th>
                    <th className="px-6 py-4 tracking-wider text-violet-400 font-sans font-medium">Rotation Angle (θ)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-slate-300">
                  {results.map((piece) => (
                    <tr key={piece.piece_id} className="hover:bg-white/5 transition-colors">
                      <td className="px-6 py-5">
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 text-xs border border-white/5 font-semibold">
                          <Layers className="w-3 h-3"/> {piece.piece_id}
                        </span>
                      </td>
                      <td className="px-6 py-5">{piece.target_x}<span className="text-slate-600 text-xs ml-1">px</span></td>
                      <td className="px-6 py-5">{piece.target_y}<span className="text-slate-600 text-xs ml-1">px</span></td>
                      <td className="px-6 py-5">
                        <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-semibold ${piece.rotation_deg > 0 ? 'bg-violet-500/10 text-violet-300 border border-violet-500/20' : 'bg-blue-500/10 text-blue-300 border border-blue-500/20'}`}>
                          {piece.rotation_deg > 0 ? '+' : ''}{piece.rotation_deg}°
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

      </div>
    </div>
  );
}
