import React, { useState, useEffect, useRef, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';
import GraphPanel from './components/GraphPanel';
import StatusBar from './components/StatusBar';

import { Menu, BarChart2, X } from 'lucide-react';

const MIN_SIDEBAR = 150;
const MIN_CHAT = 400;
const MIN_GRAPH = 200;
const DEFAULT_SIDEBAR = 220;
const DEFAULT_GRAPH = 320;

function loadPaneWidths() {
  try {
    const saved = localStorage.getItem('hades_pane_widths');
    if (saved) {
      const parsed = JSON.parse(saved);
      if (parsed.sidebar >= MIN_SIDEBAR && parsed.graph >= MIN_GRAPH) {
        return parsed;
      }
    }
  } catch {}
  return { sidebar: DEFAULT_SIDEBAR, graph: DEFAULT_GRAPH };
}

function App() {
  const [telemetry, setTelemetry] = useState(null);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [l1Nodes, setL1Nodes] = useState([]);
  const [activeNode, setActiveNode] = useState(null);
  
  // Panel visibility
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isGraphOpen, setIsGraphOpen] = useState(true);
  const [theme, setTheme] = useState(localStorage.getItem('hades_theme') || 'dark');
  const [showShortcuts, setShowShortcuts] = useState(!localStorage.getItem('hades_visited'));

  // Resizable pane widths
  const [paneWidths, setPaneWidths] = useState(loadPaneWidths);
  const dragRef = useRef(null); // { handle: 'left'|'right', startX, startWidths }

  const savePaneWidths = useCallback((widths) => {
    localStorage.setItem('hades_pane_widths', JSON.stringify(widths));
  }, []);

  const getGridColumns = () => {
    const sidebar = isSidebarOpen ? `${paneWidths.sidebar}px` : '0px';
    const graph = isGraphOpen ? `${paneWidths.graph}px` : '0px';
    // 4px for each drag handle
    const handleLeft = isSidebarOpen ? '4px' : '0px';
    const handleRight = isGraphOpen ? '4px' : '0px';
    return `${sidebar} ${handleLeft} 1fr ${handleRight} ${graph}`;
  };

  const handleMouseDown = useCallback((handle, e) => {
    e.preventDefault();
    dragRef.current = {
      handle,
      startX: e.clientX,
      startWidths: { ...paneWidths },
    };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, [paneWidths]);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!dragRef.current) return;
      const { handle, startX, startWidths } = dragRef.current;
      const delta = e.clientX - startX;

      if (handle === 'left') {
        const newSidebar = Math.max(MIN_SIDEBAR, startWidths.sidebar + delta);
        setPaneWidths(prev => ({ ...prev, sidebar: newSidebar }));
      } else if (handle === 'right') {
        const newGraph = Math.max(MIN_GRAPH, startWidths.graph - delta);
        setPaneWidths(prev => ({ ...prev, graph: newGraph }));
      }
    };

    const handleMouseUp = () => {
      if (!dragRef.current) return;
      dragRef.current = null;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      // Persist and trigger D3 resize
      setPaneWidths(prev => {
        savePaneWidths(prev);
        return prev;
      });
      window.dispatchEvent(new Event('resize'));
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [savePaneWidths]);

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

  // Poll telemetry every 500ms if processing
  useEffect(() => {
    const interval = setInterval(() => {
      const stage = telemetry?.pipeline_stage;
      if (stage && stage !== 'idle' && stage !== 'complete') {
        fetchTelemetry();
      }
    }, 500);
    return () => clearInterval(interval);
  }, [telemetry]);

  const fetchGraphData = async () => {
    try {
      const res = await fetch('/api/graph/data');
      const data = await res.json();
      if (data.nodes && data.nodes.length > 0) {
        setGraphData(data);
      }
    } catch (err) {
      console.error('Failed to fetch graph data:', err);
    }
  };

  useEffect(() => {
    fetchTelemetry();
    fetchGraphState();
    fetchGraphData();
  }, []);

  const closeShortcuts = () => {
    setShowShortcuts(false);
    localStorage.setItem('hades_visited', 'true');
  };

  return (
    <div
      className="app-container"
      style={{ gridTemplateColumns: getGridColumns() }}
    >
      {isSidebarOpen && (
        <Sidebar 
          telemetry={telemetry} 
          refreshTelemetry={fetchTelemetry} 
          theme={theme} 
          setTheme={setTheme}
          className="open"
        />
      )}
      {!isSidebarOpen && <div style={{ width: 0, overflow: 'hidden' }} />}

      {/* Left drag handle */}
      {isSidebarOpen && (
        <div
          className="resize-handle"
          onMouseDown={(e) => handleMouseDown('left', e)}
        />
      )}
      {!isSidebarOpen && <div />}

      <ChatPanel 
        setGraphData={setGraphData} 
        refreshTelemetry={fetchTelemetry} 
        refreshGraphState={fetchGraphState}
        setActiveNode={setActiveNode}
        telemetry={telemetry}
      />

      {/* Right drag handle */}
      {isGraphOpen && (
        <div
          className="resize-handle"
          onMouseDown={(e) => handleMouseDown('right', e)}
        />
      )}
      {!isGraphOpen && <div />}

      {isGraphOpen && (
        <div className="pane-wrapper graph-drawer open">
          <GraphPanel 
            data={graphData} 
            l1Nodes={l1Nodes} 
            activeNode={activeNode} 
            telemetry={telemetry}
          />
        </div>
      )}
      {!isGraphOpen && <div style={{ width: 0, overflow: 'hidden' }} />}

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

