import React, { useState } from 'react';
import { RefreshCw } from 'lucide-react';
import logo from './assets/logo.png';
import uploadIcon from './assets/upload-icon.png';
import footerBar from './assets/footer-bar.png';
import aura from './assets/aura.png';

export default function App() {
  const [status, setStatus] = useState('idle'); 
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState({ 
    probability: 0, label: '', is_fake: false, confidence: '',
    p_real: 0, p_fake: 0, explanation: '', manipulation_type: '', 
    manipulation_scores: [], spectral_flags: [], spectral_summary: '' 
  });

  const onFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setStatus('ready');
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;
    setStatus('scanning');
    setProgress(20);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('https://jennigwen-deepfake-detector-api.hf.space/analyze', {
      // const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
      setProgress(100);
      setTimeout(() => {
        setResult({
          probability:         Number(data.probability),
          label:               data.label,
          is_fake:             data.is_fake,
          confidence:          data.confidence || '',
          p_real:              Number(data.p_real || 0),
          p_fake:              Number(data.p_fake || 0),
          explanation:         data.explanation || '',
          manipulation_type:   data.manipulation_type || '',
          manipulation_scores: data.manipulation_scores || [],
          spectral_flags:      data.spectral_flags || [],
          spectral_summary:    data.spectral_summary || '',
        });
        setStatus('result');
      }, 600);
    } catch (error) {
      alert("Backend Error! Is app.py running?");
      setStatus('ready');
    }
  };

  const confidenceStyle = (level) => {
    const map = {
      high:      { bg: 'bg-emerald-500/20', text: 'text-emerald-300', border: 'border-emerald-500/40', label: 'High confidence' },
      medium:    { bg: 'bg-cyan-500/20',    text: 'text-cyan-300',    border: 'border-cyan-500/40',    label: 'Medium confidence' },
      low:       { bg: 'bg-amber-500/20',   text: 'text-amber-300',   border: 'border-amber-500/40',   label: 'Low confidence' },
      uncertain: { bg: 'bg-red-500/20',     text: 'text-red-300',     border: 'border-red-500/40',     label: 'Uncertain' },
    };
    return map[level] || null;
  };

  return (
    <div className="bg-gradient-to-b from-[#3DACDF]/30 to-[#040505]/45 min-h-screen text-white flex flex-col font-sans overflow-hidden">
      
      {/* HEADER: -15px margin adjustment */}
      <header className="p-8 mt-[-15px] flex items-center justify-between z-50">
        <img src={logo} alt="Logo" className="h-20 w-auto" />
      </header>

      <main className="flex-1 flex flex-col items-center justify-center p-4 relative">
        
        {/* Aura Background */}
        <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[900px] pointer-events-none z-0 transition-opacity duration-1000 ${status === 'scanning' ? 'opacity-50 scale-110' : 'opacity-0'}`}>
          <img src={aura} alt="aura" className="w-full h-full object-contain animate-pulse" />
        </div>

        {/* TITLES */}
        <div className="text-center mb-12 relative z-10 w-full px-4">
          <h2 className="text-4xl md:text-[54px] font-bold tracking-tight whitespace-nowrap py-2">
            {status === 'idle' && (
              <>Detect <span className="text-cyan-400">Deepfakes</span> Instantly</>
            )}
            {(status === 'ready' || status === 'scanning') && (
              <>Ready to <span className="text-cyan-400">Analyze</span></>
            )}
            {status === 'result' && "Analysis Complete"}
          </h2>
          
          {status === 'idle' && (
            <p className="text-gray-400 text-lg font-light mt-4 mx-auto max-w-3xl">
              Upload any image and our AI will analyze it for signs of manipulation or AI generation.
            </p>
          )}
        </div>

        {/* MAIN BOX */}
        <div className="w-full max-w-5xl relative z-10 flex justify-center">
          <div className={`w-full p-16 rounded-[4rem] border-[1px] border-dashed transition-all duration-700 bg-white/[0.02] backdrop-blur-3xl flex flex-col items-center justify-center ${status === 'result' && result.is_fake ? 'border-red-500/50 shadow-red-900/10' : 'border-cyan-500/30'}`}>
            
            {status === 'idle' && (
              <label className="flex flex-col items-center cursor-pointer py-12">
                <input type="file" className="hidden" onChange={onFileChange} accept="image/*" />
                <img src={uploadIcon} alt="Upload" className="w-24 h-24 mb-10" />
                <h3 className="text-2xl font-bold mb-3 tracking-tight">Drop your image here or click to browse</h3>
                <p className="text-gray-500 text-sm tracking-widest uppercase">Supported formats: .jpg / .png / .jpeg</p>
              </label>
            )}

            {(status === 'ready' || status === 'scanning') && (
              <div className="flex flex-col md:flex-row items-center gap-16 py-6 w-full justify-center">
                <div className="w-64 h-80 rounded-[2rem] overflow-hidden border border-white/10 shadow-2xl bg-gray-900">
                  <img src={previewUrl} className="w-full h-full object-cover" alt="preview" />
                </div>
                <div className="flex-1 max-w-sm">
                  <h4 className="text-2xl font-bold mb-4 truncate">{selectedFile?.name}</h4>
                  <div className="h-3 w-full bg-white/10 rounded-full overflow-hidden">
                    <div className="h-full bg-cyan-400 shadow-[0_0_20px_#22d3ee] transition-all" style={{ width: `${progress}%` }}></div>
                  </div>
                </div>
              </div>
            )}

            {status === 'result' && (
              <div className="flex flex-col lg:flex-row w-full gap-12 lg:gap-16 animate-in zoom-in duration-700">

                {/* LEFT COLUMN: Large Image Portrait */}
                <div className="w-full lg:w-5/12 flex flex-col items-center">
                  <div className={`relative w-full max-w-[400px] aspect-[4/5] rounded-[2rem] overflow-hidden border-[6px] shadow-2xl transition-colors duration-500
                    ${result.is_fake ? 'border-red-500/80 shadow-red-500/30' : 'border-green-500/80 shadow-green-500/30'}`}>
                    <img src={previewUrl} className="w-full h-full object-cover" alt="result" />
                    
                    {/* Floating Label */}
                    <div className={`absolute bottom-6 left-1/2 -translate-x-1/2 whitespace-nowrap text-xl font-bold px-8 py-3 rounded-full border shadow-2xl backdrop-blur-md
                      ${result.is_fake ? 'bg-red-500/90 text-white border-red-300' : 'bg-green-500/90 text-white border-green-300'}`}>
                      {result.label}
                    </div>
                  </div>
                </div>

                {/* RIGHT COLUMN: Metrics & Explanations */}
                <div className="w-full lg:w-7/12 flex flex-col justify-center gap-8">
                  
                  {/* Gauge & Probability Bars Block */}
                  <div className="bg-white/5 border border-white/10 p-8 rounded-[2rem] flex flex-col sm:flex-row items-center gap-10 shadow-xl">
                    
                    {/* Enlarged Gauge */}
                    <div className="relative w-64 h-36 flex-shrink-0">
                      <svg className="w-full h-full drop-shadow-lg" viewBox="0 0 160 90">
                        <path d="M 20 80 A 60 60 0 0 1 140 80" fill="none" stroke="#1a2e3e" strokeWidth="16" strokeLinecap="round"/>
                        <path
                          d="M 20 80 A 60 60 0 0 1 140 80"
                          fill="none"
                          stroke={result.is_fake ? '#ef4444' : '#22c55e'}
                          strokeWidth="16"
                          strokeLinecap="round"
                          strokeDasharray={Math.PI * 60}
                          style={{ strokeDashoffset: Math.PI * 60 - (Math.PI * 60 * result.probability / 100), transition: 'stroke-dashoffset 1.5s ease-out' }}
                        />
                      </svg>
                      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-center">
                        <div className={`text-6xl font-black tracking-tighter drop-shadow-md ${result.is_fake ? 'text-red-500' : 'text-green-500'}`}>
                          {Math.round(result.probability)}%
                        </div>
                        <div className="text-xs text-gray-400 uppercase tracking-widest font-bold mt-2">Fake Score</div>
                      </div>
                    </div>

                    {/* Bars & Confidence Badge */}
                    <div className="flex-1 w-full flex flex-col gap-5 justify-center mt-4 sm:mt-0">
                      {[
                        { label: 'Fake Probability', val: result.p_fake, color: 'bg-red-500', text: 'text-red-400' },
                        { label: 'Real Probability', val: result.p_real, color: 'bg-green-500', text: 'text-green-400' },
                      ].map(({ label, val, color, text }) => (
                        <div key={label}>
                          <div className="flex justify-between text-sm mb-2">
                            <span className="text-gray-400 font-medium">{label}</span>
                            <span className={`font-bold ${text}`}>{Number(val).toFixed(1)}%</span>
                          </div>
                          <div className="h-3 w-full bg-black/40 rounded-full overflow-hidden shadow-inner">
                            <div className={`h-full ${color} rounded-full transition-all duration-1000`} style={{ width: `${val}%` }} />
                          </div>
                        </div>
                      ))}

                      {/* Confidence Badge */}
                      {result.confidence && (() => {
                        const s = confidenceStyle(result.confidence);
                        return s ? (
                          <div className="mt-2">
                            <span className={`inline-block px-4 py-1.5 rounded-full border text-xs font-bold uppercase tracking-wider ${s.bg} ${s.text} ${s.border}`}>
                              {s.label}
                            </span>
                          </div>
                        ) : null;
                      })()}
                    </div>
                  </div>

                  {/* SINGLE Explanation Block (FIXED) */}
                  {result.explanation && (
                    <div className="bg-white/5 border border-white/10 rounded-[2rem] p-8 shadow-xl">
                      <h4 className="text-sm text-cyan-400 font-bold uppercase tracking-widest mb-4">Analysis Report</h4>
                      <p className="text-gray-300 text-base leading-relaxed">{result.explanation}</p>

                      {/* Spectral Flags UI (Just the badges!) */}
                      {result.spectral_flags && result.spectral_flags.length > 0 && (
                        <div className="mt-6 pt-5 border-t border-white/10">
                          <p className="text-xs text-gray-400 uppercase tracking-widest font-bold mb-3">Detected Frequency Anomalies</p>
                          <div className="flex flex-wrap gap-2">
                            {result.spectral_flags.map(flag => (
                              <span key={flag} className="px-3 py-1.5 bg-cyan-950/50 rounded-lg text-[11px] font-bold uppercase tracking-wider text-cyan-200 border border-cyan-500/30 shadow-sm">
                                {flag.replace(/_/g, ' ')}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Manipulation Methods */}
                  {result.is_fake && (
                    <div>
                      {/* Explicit Manipulation Type Badge */}
                      {result.manipulation_type && (
                        <div className="mb-6">
                          <h4 className="text-sm text-red-400 font-bold uppercase tracking-widest mb-3">Primary Manipulation Type</h4>
                          <span className="inline-block px-5 py-2 bg-red-500/10 border border-red-500/30 text-red-400 rounded-full text-sm font-bold tracking-wide shadow-lg">
                            {result.manipulation_type}
                          </span>
                        </div>
                      )}

                      {/* Detected Anomalies Score Cards */}
                      {result.manipulation_scores && result.manipulation_scores.length > 0 && (
                        <div>
                          <h4 className="text-sm text-red-400 font-bold uppercase tracking-widest mb-4">Detected Models</h4>
                          <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                            {result.manipulation_scores.map((score, idx) => {
                              const pct = Math.round(Number(score.confidence));
                              const color = pct >= 75 ? 'bg-red-500' : pct >= 50 ? 'bg-amber-400' : 'bg-cyan-500';
                              return (
                                <div key={idx} className="bg-white/5 border border-white/10 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
                                  <p className="text-xs text-gray-400 mb-3 truncate font-medium uppercase tracking-wider" title={score.label}>{score.label}</p>
                                  <div>
                                    <p className="text-3xl font-black text-white">{pct}<span className="text-lg font-normal text-gray-500">%</span></p>
                                    <div className="h-1.5 w-full bg-black/40 rounded-full mt-3 overflow-hidden">
                                      <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
                                    </div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="mt-16 w-full max-w-sm relative z-10">
          {status === 'ready' && <button onClick={runAnalysis} className="w-full py-5 bg-cyan-500 rounded-full font-bold text-2xl hover:scale-105 transition-all shadow-[0_0_30px_rgba(6,182,212,0.4)]">Run Analysis</button>}
          {status === 'scanning' && <button disabled className="w-full py-5 bg-cyan-900/50 border border-cyan-500/30 rounded-full font-bold text-2xl flex items-center justify-center gap-3"><RefreshCw className="animate-spin" /> Analyzing...</button>}
          {status === 'result' && <button onClick={() => setStatus('idle')} className="w-full py-5 border-2 border-cyan-500 text-cyan-400 rounded-full font-bold text-2xl hover:bg-cyan-500/10 transition-all">Try Another</button>}
        </div>
      </main>

      {/* FOOTER BAR: Height changed from h-12 to h-8 for a smaller look */}
      <footer className="w-full mt-auto relative z-20">
        <img src={footerBar} alt="glow bar" className="w-full h-8 object-cover block opacity-65" />
      </footer>
    </div>
  );
}