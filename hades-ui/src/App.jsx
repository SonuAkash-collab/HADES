import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';
import GraphPanel from './components/GraphPanel';
import StatusBar from './components/StatusBar';

import { Menu, BarChart2, X } from 'lucide-react';

function App() {
  const [telemetry, setTelemetry] = useState(null);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [l1Nodes, setL1Nodes] = useState([]);
  const [activeNode, setActiveNode] = useState(null);
  
  // New State
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isGraphOpen, setIsGraphOpen] = useState(true);
  const [theme, setTheme] = useState(localStorage.getItem('hades_theme') || 'dark');
  const [showShortcuts, setShowShortcuts] = useState(!localStorage.getItem('hades_visited'));

  const fetchTelemetry = async () => {
    try {
      const res = await fetch('/api/telemetry');
      const data = await res.json();
      setTelemetry(data);
    } catch (err) {
      console.error('Failed to fetch telemetry:', err);
    }
  };

  const fetchGraphState = async () => {
    try {
      const res = await fetch('/api/graph/state');
      const data = await res.json();
      setL1Nodes(data.l1_nodes || []);
    } catch (err) {
      console.error('Failed to fetch graph state:', err);
    }
  };

  const flushCache = async () => {
    try {
      await fetch('/api/cache/flush', { method: 'POST' });
      await fetchTelemetry();
      await fetchGraphState();
    } catch (err) {
      console.error('Flush failed:', err);
    }
  };

  // Theme Sync
  useEffect(() => {
    document.body.setAttribute('data-theme', theme);
    localStorage.setItem('hades_theme', theme);
  }, [theme]);

  // Global Shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        document.querySelector('.chat-input')?.focus();
      }
      if (e.ctrlKey && e.key === '/') {
        e.preventDefault();
        setIsSidebarOpen(prev => !prev);
      }
      if (e.ctrlKey && e.key === 'g') {
        e.preventDefault();
        setIsGraphOpen(prev => !prev);
      }
      if (e.ctrlKey && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        flushCache();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Poll telemetry every 2 seconds if processing
  useEffect(() => {
    const interval = setInterval(() => {
      const stage = telemetry?.pipeline_stage;
      if (stage && stage !== 'idle' && stage !== 'complete') {
        fetchTelemetry();
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [telemetry]);

  useEffect(() => {
    fetchTelemetry();
    fetchGraphState();
  }, []);

  const closeShortcuts = () => {
    setShowShortcuts(false);
    localStorage.setItem('hades_visited', 'true');
  };

  return (
    <div className={`app-container ${!isSidebarOpen ? 'sidebar-collapsed' : ''} ${!isGraphOpen ? 'graph-collapsed' : ''}`}>
      <Sidebar 
        telemetry={telemetry} 
        refreshTelemetry={fetchTelemetry} 
        theme={theme} 
        setTheme={setTheme}
        className={isSidebarOpen ? 'open' : ''}
      />
      <ChatPanel 
        setGraphData={setGraphData} 
        refreshTelemetry={fetchTelemetry} 
        refreshGraphState={fetchGraphState}
        setActiveNode={setActiveNode}
      />
      <div className={`pane-wrapper graph-drawer ${isGraphOpen ? 'open' : ''}`}>
        <GraphPanel 
          data={graphData} 
          l1Nodes={l1Nodes} 
          activeNode={activeNode} 
          telemetry={telemetry}
        />
      </div>

      {/* Floating Drawers for Mobile/Tablet */}
      <button className="drawer-toggle sidebar-toggle" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
        <Menu size={20} />
      </button>
      <button className="drawer-toggle graph-toggle" onClick={() => setIsGraphOpen(!isGraphOpen)}>
        <BarChart2 size={20} />
      </button>

      {/* Shortcuts Modal */}
      {showShortcuts && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2 className="modal-title">HADES TERMINAL SHORTCUTS</h2>
            <div className="shortcut-list">
              <div className="shortcut-item"><span>Focus Chat</span><span className="shortcut-key">Ctrl + K</span></div>
              <div className="shortcut-item"><span>Toggle Sidebar</span><span className="shortcut-key">Ctrl + /</span></div>
              <div className="shortcut-item"><span>Toggle Graph</span><span className="shortcut-key">Ctrl + G</span></div>
              <div className="shortcut-item"><span>Flush Cache</span><span className="shortcut-key">Ctrl + Shift + C</span></div>
            </div>
            <button className="modal-close-btn" onClick={closeShortcuts}>Initialize</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
