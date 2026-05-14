import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Upload, FileText, CheckCircle, AlertTriangle } from 'lucide-react';
import { useDropzone } from 'react-dropzone';

const ChatPanel = ({ setGraphData, refreshTelemetry, refreshGraphState, setActiveNode, telemetry }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [pdfInfo, setPdfInfo] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('idle'); // idle, uploading, success, error
  const [uploadFileName, setUploadFileName] = useState('');
  const [uploadStats, setUploadStats] = useState(null);
  const [estimatedTime, setEstimatedTime] = useState(25);
  const messagesEndRef = useRef(null);

  // Estimated time countdown
  useEffect(() => {
    let timer;
    if (uploadStatus === 'uploading') {
      setEstimatedTime(25); // Reset to 25s
      timer = setInterval(() => {
        setEstimatedTime(prev => (prev > 1 ? prev - 1 : 1));
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [uploadStatus]);

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

    setUploadStatus('uploading');
    setUploadFileName(file.name);
    
    // Start polling telemetry while uploading to update StatusBar
    const telemetryInterval = setInterval(refreshTelemetry, 500);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) throw new Error('Upload failed');
      
      const data = await res.json();
      setUploadStatus('success');
      setUploadStats({
        triples: data.triple_count,
        nodes: data.node_count
      });
      setPdfInfo({
        name: file.name,
        triples: data.triple_count,
        nodes: data.node_count,
      });
      
      // Add system message to chat
      setMessages(prev => [...prev, {
        role: 'system',
        content: `— ${file.name} loaded into L2 memory — ${data.triple_count} triples, ${data.node_count} nodes —`
      }]);

      setGraphData(data.graph_data);
      refreshTelemetry();
    } catch (err) {
      console.error('Upload failed:', err);
      setUploadStatus('error');
    } finally {
      clearInterval(telemetryInterval);
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

    let streamVerdict = null;

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
              } else if (data.verdict) {
                // Server sends the authoritative verdict from Cerberus
                streamVerdict = data.verdict;
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
      
      // Use the verdict from the SSE stream if available, otherwise fallback to telemetry
      let finalVerdict = streamVerdict || 'VERIFIED';
      if (!streamVerdict) {
        try {
          const telRes = await fetch('/api/telemetry');
          const telData = await telRes.json();
          const logs = telData.cerberus_log || [];
          const lastLogs = logs.slice(-5);
          
          if (lastLogs.some(l => l.toUpperCase().includes('CONTRADICTION'))) finalVerdict = 'BLOCKED';
          else if (lastLogs.some(l => l.toUpperCase().includes('NEUTRAL'))) finalVerdict = 'NEUTRAL';
        } catch (e) {
          console.error('Telemetry fallback failed:', e);
        }
      }
      
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
            } : msg.role === 'system' ? {
              alignSelf: 'center',
              backgroundColor: 'transparent',
              border: 'none',
              color: 'var(--text-secondary)',
              fontStyle: 'italic',
              fontSize: '0.75rem',
              textAlign: 'center',
              maxWidth: '100%',
              padding: '0.5rem',
              opacity: 0.8
            } : {}}
          >
            {msg.role === 'assistant' ? msg.content.replace(/\{"tool":\s*".*?"\}/g, '').trim() : msg.content}
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
        {uploadStatus === 'idle' && (
          <>
            {(telemetry?.pipeline_stage === 'complete' || telemetry?.l1_status === 'pdf_loaded' || pdfInfo) ? (
              <div style={{ 
                padding: '0.5rem 0.75rem', 
                fontSize: '0.7rem', 
                color: 'var(--accent-blue)', 
                display: 'flex', 
                alignItems: 'center', 
                gap: '8px', 
                border: '1px solid rgba(77, 159, 255, 0.2)', 
                backgroundColor: 'rgba(77, 159, 255, 0.05)',
                borderRadius: '4px', 
                marginBottom: '0.75rem',
                flexShrink: 0
              }}>
                <FileText size={14} />
                <span style={{ fontWeight: '500' }}>Active Document in Memory</span>
                <span style={{ color: 'var(--text-muted)' }}>— Ready for queries</span>
              </div>
            ) : (
              <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`} style={{
                border: isDragActive ? '1px solid #4d9fff' : '1px dashed var(--border-accent)',
                backgroundColor: isDragActive ? 'rgba(77, 159, 255, 0.05)' : 'transparent',
                flexShrink: 0
              }}>
                <input {...getInputProps()} />
                {isDragActive ? (
                  <p style={{ color: '#4d9fff' }}>Drop to ingest</p>
                ) : (
                  <p>Drag & drop PDF to ingest into L2 graph</p>
                )}
              </div>
            )}
          </>
        )}

        {uploadStatus === 'uploading' && (
          <div className="dropzone uploading" style={{ 
            border: '1px solid var(--border-accent)', 
            backgroundColor: 'rgba(255, 255, 255, 0.02)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1.5rem',
            gap: '12px'
          }}>
            <div className="spinner" style={{ borderTopColor: '#4d9fff' }}></div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontWeight: '600', color: 'var(--text-primary)', marginBottom: '4px' }}>{uploadFileName}</div>
              <div style={{ 
                fontSize: '0.75rem', 
                color: '#4d9fff', 
                textTransform: 'uppercase', 
                letterSpacing: '1px',
                fontWeight: '700'
              }}>
                {telemetry?.pipeline_stage || "INITIALIZING..."}
              </div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Estimated completion: ~{estimatedTime}s
              </div>
            </div>
          </div>
        )}

        {uploadStatus === 'success' && uploadStats && (
          <div style={{ 
            padding: '0.5rem 0.75rem', 
            fontSize: '0.7rem', 
            color: 'var(--accent-green)', 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px', 
            border: '1px solid rgba(0, 230, 118, 0.3)', 
            backgroundColor: 'rgba(0, 230, 118, 0.05)',
            borderRadius: '4px', 
            marginBottom: '0.75rem',
            flexShrink: 0
          }}>
            <CheckCircle size={14} />
            <span style={{ fontWeight: '500' }}>{uploadFileName} ingested</span>
            <span style={{ color: 'var(--text-muted)' }}>— {uploadStats.triples} triples → {uploadStats.nodes} nodes</span>
          </div>
        )}

        {uploadStatus === 'error' && (
          <div {...getRootProps()} className="dropzone error" style={{ 
            borderColor: '#ff4444', 
            backgroundColor: 'rgba(255, 68, 68, 0.05)',
            color: '#ff4444'
          }}>
            <input {...getInputProps()} />
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
              <AlertTriangle size={14} />
              <span>✗ Upload failed — please try again</span>
            </div>
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
