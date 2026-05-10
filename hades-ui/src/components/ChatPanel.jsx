import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Upload, FileText, CheckCircle, AlertTriangle } from 'lucide-react';
import { useDropzone } from 'react-dropzone';

const ChatPanel = ({ setGraphData, refreshTelemetry, refreshGraphState, setActiveNode }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [pdfInfo, setPdfInfo] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const onDrop = useCallback(async (acceptedFiles, fileRejections) => {
    console.log('onDrop triggered');
    console.log('Accepted files:', acceptedFiles);
    console.log('Rejected files:', fileRejections);
    
    if (fileRejections.length > 0) {
      console.error('File rejected:', fileRejections[0].errors);
    }

    const file = acceptedFiles[0];
    if (!file) {
      console.warn('No file accepted');
      return;
    }

    console.log('file dropped:', file.name);
    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log('Starting fetch /api/upload with FormData field "file"');
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      console.log('Response status:', res.status);
      const data = await res.json();
      console.log('Upload response data:', data);
      setPdfInfo({
        name: file.name,
        triples: data.triple_count,
        nodes: data.node_count,
      });
      setGraphData(data.graph_data);
      refreshTelemetry();
    } catch (err) {
      console.error('Upload failed with error:', err);
    }
  }, [setGraphData, refreshTelemetry]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop, 
    accept: { 
      'application/pdf': ['.pdf'],
      'application/x-pdf': ['.pdf'],
      'application/octet-stream': ['.pdf']
    },
    multiple: false 
  });

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;

    const userPrompt = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userPrompt }]);
    setIsStreaming(true);
    setIsReconnecting(false);

    let assistantMsg = { role: 'assistant', content: '', status: 'streaming' };
    setMessages(prev => [...prev, assistantMsg]);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: userPrompt }),
      });

      if (!response.ok) throw new Error('Network response was not ok');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6);
            if (dataStr === '[DONE]') {
              done = true;
              break;
            }

            try {
              const data = JSON.parse(dataStr);
              if (data.token) {
                assistantMsg.content += data.token;
                setMessages(prev => {
                  const last = [...prev];
                  last[last.length - 1] = { ...assistantMsg };
                  return last;
                });
              } else if (data.error) {
                assistantMsg.status = 'error';
                setMessages(prev => {
                  const last = [...prev];
                  last[last.length - 1] = { ...assistantMsg };
                  return last;
                });
              }
            } catch (e) {
              console.error('Parse error:', e, dataStr);
            }
          }
        }
      }

      assistantMsg.status = 'complete';
      setMessages(prev => {
        const last = [...prev];
        last[last.length - 1] = { ...assistantMsg };
        return last;
      });

      // Post-chat updates
      await refreshTelemetry();
      await refreshGraphState();
      
      // Determine verdict from telemetry logs
      const telRes = await fetch('/api/telemetry');
      const telData = await telRes.json();
      const logs = telData.cerberus_log || [];
      const lastLogs = logs.slice(-5);
      
      let finalVerdict = 'VERIFIED';
      if (lastLogs.some(l => l.includes('CONTRADICTION'))) finalVerdict = 'BLOCKED';
      else if (lastLogs.some(l => l.includes('NEUTRAL'))) finalVerdict = 'NEUTRAL';
      
      setMessages(prev => {
        const last = [...prev];
        const lastMsg = last[last.length - 1];
        if (lastMsg.role === 'assistant') {
          lastMsg.verdict = finalVerdict;
          lastMsg.status = 'complete'; // Ensure status is synced
        }
        return last;
      });
    } catch (err) {
      console.error('Chat failed:', err);
      setIsReconnecting(true);
      // Simple retry logic could go here
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="pane chat-pane">
      <div className="message-thread">
        {messages.map((msg, i) => (
          <div 
            key={i} 
            className={`message ${msg.role}`}
            style={msg.role === 'assistant' ? {
              color: msg.verdict === 'BLOCKED' ? '#ff4444' : msg.verdict === 'NEUTRAL' ? '#ffb400' : '#ffffff',
              borderLeft: `2px solid ${msg.verdict === 'BLOCKED' ? '#ff4444' : msg.verdict === 'NEUTRAL' ? '#ffb400' : '#00e676'}`
            } : {}}
          >
            {msg.content}
            {msg.status === 'streaming' && <span className="cursor" />}
            
            {(msg.status === 'complete' || msg.verdict) && (
              <div style={{ marginTop: '8px' }}>
                {msg.verdict === 'BLOCKED' ? (
                  <>
                    <div className="verification-badge badge-blocked">
                      <AlertTriangle size={10} /> ✗ BLOCKED
                    </div>
                    <div style={{ fontSize: '0.6rem', fontStyle: 'italic', color: '#ff4444', marginTop: '4px' }}>
                      Blocked: contradicts source document
                    </div>
                  </>
                ) : msg.verdict === 'NEUTRAL' ? (
                  <>
                    <div className="verification-badge badge-neutral" style={{ backgroundColor: 'rgba(255, 180, 0, 0.1)', color: '#ffb400', border: '1px solid rgba(255, 180, 0, 0.2)' }}>
                      <AlertTriangle size={10} /> ⚠ NEUTRAL
                    </div>
                    <div style={{ fontSize: '0.6rem', fontStyle: 'italic', color: '#ffb400', marginTop: '4px' }}>
                      Source: model knowledge — not verified against document
                    </div>
                  </>
                ) : (
                  <div className="verification-badge badge-clean">
                    <CheckCircle size={10} /> ✓ VERIFIED
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        {isReconnecting && (
          <div style={{ color: 'var(--accent-amber)', fontSize: '0.7rem', textAlign: 'center', padding: '1rem' }}>
            Connection lost. Reconnecting...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        {!pdfInfo ? (
          <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
            <input {...getInputProps()} />
            {isDragActive ? (
              <p>Drop the PDF here...</p>
            ) : (
              <p>Drag & drop PDF to ingest into L2 graph</p>
            )}
          </div>
        ) : (
          <div style={{ padding: '0.5rem', fontSize: '0.7rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px', border: '1px solid var(--border)', borderRadius: '4px', marginBottom: '0.75rem' }}>
            <FileText size={14} color="var(--accent-blue)" />
            <span>{pdfInfo.name}</span>
            <span style={{ color: 'var(--text-muted)' }}>| {pdfInfo.triples} triples | {pdfInfo.nodes} nodes</span>
          </div>
        )}

        <div className="chat-input-wrapper">
          <textarea 
            className="chat-input"
            rows="1"
            placeholder="Ask HADES about the document..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
          />
          <button 
            onClick={handleSend}
            style={{ background: 'none', border: 'none', color: input.trim() ? 'var(--accent-blue)' : 'var(--text-muted)', cursor: 'pointer' }}
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatPanel;
