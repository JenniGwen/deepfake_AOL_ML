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
  const [result, setResult] = useState({ probability: 0, label: '', is_fake: false, confidence: '' });

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
      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
      setProgress(100);
      setTimeout(() => {
        setResult({
          probability: Number(data.probability),
          label: data.label,
          is_fake: data.is_fake,
          confidence: data.confidence || ''
        });
        setStatus('result');
      }, 600);
    } catch (error) {
      alert("Backend Error! Is app.py running?");
      setStatus('ready');
    }
  };

  const radius = 80;
  const circumference = Math.PI * radius;
  const strokeDashoffset = circumference - (circumference * (result.probability / 100));

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
    <div className="min-h-screen text-white flex flex-col font-sans overflow-hidden">
      
      {/* HEADER: -15px margin adjustment */}
      <header className="p-8 mt-[-15px] flex items-center justify-between z-50">
        <img src={logo} alt="Logo" className="h-20 w-auto" />
      </header>

      <main className="flex-1 flex flex-col items-center justify-center p-4 relative">
        
        {/* Aura Background */}
        <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[900px] pointer-events-none z-0 transition-opacity duration-1000 ${status === 'scanning' ? 'opacity-50 scale-110' : 'opacity-0'}`}>
          <img src={aura} alt="aura" className="w-full h-full object-contain animate-pulse" />
        </div>

        {/* TITLES: One line, no italics, sentence-case, no clipping */}
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
              <div className="flex flex-col md:flex-row items-center justify-center gap-20 w-full animate-in zoom-in duration-700">
                 <div className={`w-64 h-80 rounded-[2.5rem] overflow-hidden border-4 ${result.is_fake ? 'border-red-500' : 'border-green-500'}`}>
                    <img src={previewUrl} className="w-full h-full object-cover" alt="result" />
                 </div>

                 <div className="flex flex-col items-center">
                    <div className="relative mb-6">
                        <svg width="240" height="130" viewBox="0 0 200 110">
                            <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#1a2e3e" strokeWidth="14" strokeLinecap="round" />
                            <path 
                                d="M 20 100 A 80 80 0 0 1 180 100" 
                                fill="none" 
                                stroke={result.is_fake ? "#ef4444" : "#22c55e"} 
                                strokeWidth="14" 
                                strokeLinecap="round" 
                                strokeDasharray={circumference}
                                style={{ strokeDashoffset: strokeDashoffset, transition: 'stroke-dashoffset 2s ease-out' }}
                            />
                        </svg>
                        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-center w-full">
                            <div className={`text-6xl font-black tracking-tighter ${result.is_fake ? 'text-red-500' : 'text-green-500'}`}>
                                {Math.round(result.probability)}%
                            </div>
                            <div className="text-[10px] text-gray-500 uppercase tracking-widest font-black">probability</div>
                        </div>
                    </div>
                    <h3 className={`text-4xl font-bold tracking-tighter text-center ${result.is_fake ? 'text-red-500' : 'text-green-500'}`}>
                      {result.is_fake ? "AI Generated / Fake" : "Authentic Media"}
                    </h3>

                    {/* CONFIDENCE BADGE */}
                    {result.confidence && confidenceStyle(result.confidence) && (
                      <div className={`mt-4 px-4 py-1.5 rounded-full border text-sm font-semibold tracking-wide
                        ${confidenceStyle(result.confidence).bg}
                        ${confidenceStyle(result.confidence).text}
                        ${confidenceStyle(result.confidence).border}`}>
                        {confidenceStyle(result.confidence).label}
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
        <img src={footerBar} alt="glow bar" className="w-full h-8 object-cover block" />
      </footer>
    </div>
  );
}
