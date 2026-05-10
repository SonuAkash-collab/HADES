import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Activity, Database, AlertCircle, Shield, Settings, Sun, Moon } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const CollapsiblePanel = ({ title, icon: Icon, children, id }) => {
  const [isOpen, setIsOpen] = useState(() => {
    const saved = localStorage.getItem(`panel_${id}`);
    return saved === null ? true : saved === 'true';
  });

  const toggle = () => {
    const next = !isOpen;
    setIsOpen(next);
    localStorage.setItem(`panel_${id}`, next);
  };

  return (
    <div className="collapsible-panel">
      <button className="collapsible-trigger" onClick={toggle}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Icon size={14} />
          {title}
        </span>
        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            style={{ overflow: 'hidden' }}
          >
            <div className="collapsible-content">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const Sidebar = ({ telemetry, theme, setTheme, className }) => {
  const [model, setModel] = useState('qwen3:0.6b');

  return (
    <div className={`pane sidebar ${className}`}>
      <div className="sidebar-sticky-header" style={{ position: 'sticky', top: 0, backgroundColor: 'var(--bg-secondary)', zIndex: 10 }}>
        <div className="sidebar-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>HADES TERMINAL</span>
          <button 
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', display: 'flex', padding: '2px' }}
          >
            {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
          </button>
        </div>
        
        <div style={{ padding: '1rem' }}>
          <select 
            className="chat-input" 
            style={{ width: '100%', padding: '0.5rem', background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: '4px' }}
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            <option value="qwen3:0.6b">qwen3:0.6b</option>
            <option value="phi3.5">phi3.5</option>
            <option value="llama3.2:3b">llama3.2:3b</option>
          </select>
        </div>
      </div>

      <div className="sidebar-scroll-content" style={{ flex: 1, overflowY: 'auto' }}>
        <CollapsiblePanel title="Telemetry" icon={Activity} id="telemetry">
        <div style={{ display: 'grid', gap: '8px' }}>
          <div>STATUS: <span style={{ color: 'var(--accent-green)' }}>{telemetry?.l1_status || 'IDLE'}</span></div>
          <div>TOOL CALLS: <span style={{ color: 'var(--accent-blue)' }}>{telemetry?.tool_calls || 0}</span></div>
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel title="Memory Fault Log" icon={AlertCircle} id="faults">
        <div style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
          {telemetry?.memory_faults?.length > 0 ? (
            telemetry.memory_faults.map((f, i) => (
              <div key={i} style={{ marginBottom: '4px', borderLeft: '1px solid var(--border)', paddingLeft: '4px' }}>
                ▸ {f}
              </div>
            ))
          ) : (
            <div style={{ color: 'var(--text-muted)' }}>No faults recorded</div>
          )}
        </div>
      </CollapsiblePanel>

      <CollapsiblePanel title="Cerberus Gate Log" icon={Shield} id="cerberus">
        <div style={{ fontSize: '0.65rem', fontFamily: 'var(--font-mono)' }}>
          {telemetry?.cerberus_log?.length > 0 ? (
            telemetry.cerberus_log.map((entry, i) => {
              const isClean = entry.includes('CLEAN');
              const color = isClean ? 'var(--accent-green)' : entry.includes('CONTRADICTION') ? 'var(--accent-red)' : 'var(--accent-amber)';
              return (
                <div key={i} style={{ marginBottom: '4px', color, borderLeft: `2px solid ${color}`, paddingLeft: '6px' }}>
                  {entry}
                </div>
              );
            })
          ) : (
            <div style={{ color: 'var(--text-muted)' }}>No gate logs</div>
          )}
        </div>
      </CollapsiblePanel>

        <CollapsiblePanel title="Settings" icon={Settings} id="settings">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input type="checkbox" id="deep-res" />
            <label htmlFor="deep-res">Deep Entity Resolution</label>
          </div>
        </CollapsiblePanel>
      </div>
    </div>
  );
};

export default Sidebar;
