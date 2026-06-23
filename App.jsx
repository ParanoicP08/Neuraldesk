import { useState, useRef, useEffect, useCallback } from "react";

function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);
}
function buildTFVector(tokens) {
  const tf = {};
  for (const t of tokens) tf[t] = (tf[t] || 0) + 1;
  return tf;
}
function cosineSim(a, b) {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  let dot = 0, magA = 0, magB = 0;
  for (const k of keys) {
    const va = a[k] || 0, vb = b[k] || 0;
    dot += va * vb; magA += va * va; magB += vb * vb;
  }
  if (!magA || !magB) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}
function chunkText(text, chunkSize = 200, overlap = 40) {
  const words = text.split(/\s+/);
  const chunks = [];
  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    const chunk = words.slice(i, i + chunkSize).join(" ");
    if (chunk.trim()) chunks.push(chunk);
    if (i + chunkSize >= words.length) break;
  }
  return chunks;
}
class VectorStore {
  constructor() { this.docs = []; }
  add(text, meta = {}) {
    const chunks = chunkText(text);
    chunks.forEach((chunk, i) => {
      this.docs.push({ chunk, vec: buildTFVector(tokenize(chunk)), meta: { ...meta, chunkIndex: i } });
    });
    return chunks.length;
  }
  query(q, topK = 3) {
    const qvec = buildTFVector(tokenize(q));
    return this.docs
      .map(d => ({ ...d, score: cosineSim(qvec, d.vec) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .filter(d => d.score > 0);
  }
  clear() { this.docs = []; }
  get count() { return this.docs.length; }
}
const store = new VectorStore();

async function* streamOllama(model, messages, signal) {
  const res = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({ model, messages, stream: true }),
  });
  if (!res.ok) throw new Error(`Ollama error ${res.status}: ${await res.text()}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const line of dec.decode(value).split("\n").filter(Boolean)) {
      try {
        const j = JSON.parse(line);
        if (j.message?.content) yield j.message.content;
        if (j.done) return;
      } catch {}
    }
  }
}

async function listModels() {
  const res = await fetch("http://localhost:11434/api/tags");
  if (!res.ok) return [];
  const data = await res.json();
  return data.models?.map(m => m.name) || [];
}

const C = {
  bg: "#0d0d10", surface: "#16161a", surfaceHigh: "#1e1e24",
  border: "#2a2a35", accent: "#7c6af7", accentDim: "#3d3578",
  accentGlow: "rgba(124,106,247,0.15)", text: "#e8e8f0",
  textMid: "#9898b0", textDim: "#5a5a70", green: "#4ade80", red: "#f87171",
};

export default function App() {
  const [mode, setMode] = useState("rag");
  const [model, setModel] = useState("deepseek-r1:7b");
  const [models, setModels] = useState(["deepseek-r1:7b"]);
  const [ollamaOnline, setOllamaOnline] = useState(null);
  const [docs, setDocs] = useState([]);
  const [chunkCount, setChunkCount] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [toast, setToast] = useState(null);

  const abortRef = useRef(null);
  const bottomRef = useRef(null);
  const storeRef = useRef(store);

  useEffect(() => {
    (async () => {
      try {
        const m = await listModels();
        setModels(m.length ? m : ["llama3"]);
        if (m.length) setModel(m[0]);
        setOllamaOnline(true);
      } catch { setOllamaOnline(false); }
    })();
  }, []);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  };

  const ingest = useCallback((text, name) => {
    if (!text.trim()) return;
    const n = storeRef.current.add(text, { name });
    setChunkCount(storeRef.current.count);
    setDocs(prev => [...prev, { name, size: text.length, chunks: n }]);
    showToast(`"${name}" indexed — ${n} chunks`);
  }, []);

  const handleFile = async (file) => {
    const text = await file.text();
    ingest(text, file.name);
  };

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const removeDoc = (idx) => {
    storeRef.current.clear();
    setDocs(prev => prev.filter((_, i) => i !== idx));
    setChunkCount(0);
    showToast("Document removed");
  };

  const send = async () => {
    const q = input.trim();
    if (!q || streaming) return;
    if (!ollamaOnline) { showToast("Ollama is offline"); return; }

    setInput("");
    const userMsg = { role: "user", content: q };
    setMessages(prev => [...prev, userMsg]);

    let systemPrompt = "You are a helpful assistant.";
    let sources = [];

    if (mode === "rag" && storeRef.current.count > 0) {
      const hits = storeRef.current.query(q, 3);
      if (hits.length) {
        const context = hits.map((h, i) => `[${i + 1}] ${h.chunk}`).join("\n\n");
        sources = [...new Set(hits.map(h => h.meta.name).filter(Boolean))];
        systemPrompt = `You are a helpful assistant. Answer using the context below. If the answer isn't in the context, say so.\n\nCONTEXT:\n${context}`;
      }
    }

    const history = [...messages, userMsg].map(m => ({ role: m.role, content: m.content }));
    const apiMessages = sources.length
      ? [{ role: "user", content: systemPrompt + "\n\nQuestion: " + q }]
      : history;

    setStreaming(true);
    setMessages(prev => [...prev, { role: "assistant", content: "", sources, streaming: true }]);

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      let full = "";
      for await (const token of streamOllama(model, apiMessages, ctrl.signal)) {
        full += token;
        const isThinking = /<think>/.test(full) && !/<\/think>/.test(full);
        const visible = full.replace(/<think>[\s\S]*?<\/think>/g, "").trimStart();
        setMessages(prev => {
          const copy = [...prev];
          copy[copy.length - 1] = { ...copy[copy.length - 1], content: isThinking ? "" : visible, thinking: isThinking };
          return copy;
        });
      }
      setMessages(prev => {
        const copy = [...prev];
        const last = copy[copy.length - 1];
        const cleaned = last.content.replace(/<think>[\s\S]*?<\/think>/g, "").trimStart();
        copy[copy.length - 1] = { ...last, content: cleaned, streaming: false, thinking: false };
        return copy;
      });
    } catch (e) {
      if (e.name !== "AbortError") {
        setMessages(prev => {
          const copy = [...prev];
          copy[copy.length - 1] = {
            ...copy[copy.length - 1],
            content: `Error: ${e.message}\n\nMake sure Ollama is running:\n  ollama serve\n  ollama pull ${model}`,
            streaming: false, error: true,
          };
          return copy;
        });
      }
    } finally { setStreaming(false); }
  };

  return (
    <div style={{minHeight:"100vh",background:C.bg,color:C.text,fontFamily:"Inter,sans-serif",fontSize:14}}>
      <div style={{display:"flex",alignItems:"center",gap:10,padding:"0 18px",height:48,background:C.surface,borderBottom:`1px solid ${C.border}`}}>
        <div style={{width:7,height:7,borderRadius:"50%",background:C.accent,boxShadow:`0 0 8px ${C.accent}`}}/>
        <span style={{fontWeight:600}}>NeuralDesk</span>
        <div style={{marginLeft:"auto",display:"flex",alignItems:"center",gap:8}}>
          <div style={{width:6,height:6,borderRadius:"50%",background:ollamaOnline?C.green:C.red}}/>
          <span style={{fontSize:11,color:ollamaOnline?C.green:C.red}}>{ollamaOnline===null?"checking…":ollamaOnline?"online":"offline"}</span>
          <select style={{background:C.surfaceHigh,border:`1px solid ${C.border}`,color:C.text,padding:"4px 8px",borderRadius:6,fontSize:11}} value={model} onChange={e=>setModel(e.target.value)}>
            {models.map(m=><option key={m} value={m}>{m}</option>)}
          </select>
        </div>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"240px 1fr",height:"calc(100vh - 48px)"}}>
        <div style={{background:C.surface,borderRight:`1px solid ${C.border}`,padding:16,display:"flex",flexDirection:"column",gap:12}}>
          <div style={{fontSize:10,fontWeight:600,letterSpacing:"1.5px",textTransform:"uppercase",color:C.textDim}}>Documents</div>
          <div
            onDragOver={e=>{e.preventDefault();setDragging(true);}}
            onDragLeave={()=>setDragging(false)}
            onDrop={handleDrop}
            style={{border:`1px dashed ${dragging?C.accent:C.border}`,borderRadius:8,padding:"24px 12px",textAlign:"center",color:C.textDim,fontSize:12,cursor:"pointer",position:"relative",background:dragging?C.accentGlow:"transparent"}}
          >
            <input type="file" accept=".txt,.md,.csv,.json,.html" style={{position:"absolute",inset:0,opacity:0,cursor:"pointer",width:"100%",height:"100%"}} onChange={e=>e.target.files[0]&&handleFile(e.target.files[0])}/>
            <div style={{fontSize:20,marginBottom:6}}>📂</div>
            <div>Drop file or click</div>
            <div style={{fontSize:10,color:C.textDim,marginTop:4}}>.txt .md .csv .json .html</div>
          </div>
          <div style={{flex:1,overflowY:"auto",display:"flex",flexDirection:"column",gap:6}}>
            {docs.length===0
              ? <div style={{fontSize:11,color:C.textDim,textAlign:"center",padding:"12px 0"}}>No documents yet</div>
              : docs.map((d,i)=>(
                <div key={i} style={{background:C.surfaceHigh,border:`1px solid ${C.border}`,borderRadius:6,padding:"7px 9px",display:"flex",alignItems:"center",gap:7}}>
                  <div style={{flex:1,minWidth:0}}>
                    <div style={{fontSize:11,fontWeight:500,whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{d.name}</div>
                    <div style={{fontSize:10,color:C.textDim}}>{d.chunks} chunks · {(d.size/1000).toFixed(1)}KB</div>
                  </div>
                  <button onClick={()=>removeDoc(i)} style={{background:"none",border:"none",color:C.textDim,cursor:"pointer",fontSize:13}}>✕</button>
                </div>
              ))
            }
          </div>
        </div>

        <div style={{display:"flex",flexDirection:"column",overflow:"hidden"}}>
          <div style={{display:"flex",gap:2,padding:"6px 14px",borderBottom:`1px solid ${C.border}`,background:C.surface}}>
            {["rag","chat"].map(m=>(
              <button key={m} onClick={()=>setMode(m)} style={{padding:"4px 11px",borderRadius:5,border:mode===m?`1px solid ${C.accentDim}`:"none",background:mode===m?C.accentGlow:"transparent",color:mode===m?C.accent:C.textMid,fontSize:12,cursor:"pointer"}}>
                {m==="rag"?"⚡ RAG":"💬 Chat"}
              </button>
            ))}
            {mode==="rag"&&chunkCount>0&&<span style={{marginLeft:"auto",fontSize:11,color:C.textDim}}>searching <span style={{color:C.accent}}>{chunkCount}</span> chunks</span>}
          </div>

          <div style={{flex:1,overflowY:"auto",padding:"20px 22px",display:"flex",flexDirection:"column",gap:14}}>
            {messages.length===0&&(
              <div style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:8,color:C.textDim,textAlign:"center"}}>
                <div style={{fontSize:36,opacity:0.35}}>{mode==="rag"?"🧠":"💬"}</div>
                <div style={{fontSize:15,fontWeight:500,color:C.textMid}}>{mode==="rag"?"Ask your documents":"Chat with local LLM"}</div>
              </div>
            )}
            {messages.map((m,i)=>(
              <div key={i} style={{display:"flex",gap:10,maxWidth:740,alignSelf:m.role==="user"?"flex-end":"flex-start",flexDirection:m.role==="user"?"row-reverse":"row"}}>
                <div style={{width:26,height:26,borderRadius:5,display:"flex",alignItems:"center",justifyContent:"center",fontSize:12,flexShrink:0,background:m.role==="assistant"?C.accentGlow:C.surfaceHigh,border:`1px solid ${m.role==="assistant"?C.accentDim:C.border}`}}>
                  {m.role==="assistant"?"🤖":"👤"}
                </div>
                <div style={{display:"flex",flexDirection:"column",gap:4,maxWidth:620}}>
                  {m.thinking&&<div style={{fontSize:11,color:C.accent,opacity:0.7}}>⚙ Thinking…</div>}
                  {!m.thinking&&(
                    <div style={{padding:"9px 13px",borderRadius:9,fontSize:13.5,lineHeight:1.65,whiteSpace:"pre-wrap",wordBreak:"break-word",background:m.role==="assistant"?C.surfaceHigh:C.accentDim,border:`1px solid ${m.role==="assistant"?C.border:C.accent+"44"}`,color:m.error?C.red:C.text}}>
                      {m.content}
                    </div>
                  )}
                  {m.sources?.length>0&&(
                    <div style={{display:"flex",flexWrap:"wrap",gap:4}}>
                      {m.sources.map((s,si)=><span key={si} style={{background:C.accentGlow,border:`1px solid ${C.accentDim}`,color:C.accent,fontSize:10,padding:"1px 6px",borderRadius:4}}>{s}</span>)}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef}/>
          </div>

          <div style={{padding:"10px 14px 14px",borderTop:`1px solid ${C.border}`,background:C.surface}}>
            <div style={{display:"flex",gap:8,background:C.surfaceHigh,border:`1px solid ${C.border}`,borderRadius:9,padding:"7px 9px"}}>
              <textarea
                value={input}
                onChange={e=>setInput(e.target.value)}
                onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}}}
                placeholder="Ask something…"
                style={{flex:1,background:"transparent",border:"none",color:C.text,fontFamily:"Inter,sans-serif",fontSize:13.5,outline:"none",resize:"none",maxHeight:100,lineHeight:1.5}}
              />
              <button onClick={send} disabled={streaming||!input.trim()} style={{background:C.accent,border:"none",borderRadius:6,width:30,height:30,display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",color:"#fff",fontSize:13,alignSelf:"flex-end"}}>
                ↑
              </button>
            </div>
          </div>
        </div>
      </div>
      {toast&&<div style={{position:"fixed",bottom:20,left:"50%",transform:"translateX(-50%)",background:C.surfaceHigh,border:`1px solid ${C.border}`,color:C.text,padding:"7px 14px",borderRadius:7,fontSize:12}}>{toast}</div>}
    </div>
  );
      }
