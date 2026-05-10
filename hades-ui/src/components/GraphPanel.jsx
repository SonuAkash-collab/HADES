import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { ZoomIn, ZoomOut, Maximize2, Info, Maximize, ExternalLink } from 'lucide-react';
import StatusBar from './StatusBar';

const GraphPanel = ({ data, l1Nodes: propsL1Nodes, activeNode: propsActiveNode, telemetry }) => {
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const tooltipRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const simulationRef = useRef(null);
  
  // Internal state for live updates
  const [l1Nodes, setL1Nodes] = useState(propsL1Nodes || []);
  const [activeNode, setActiveNode] = useState(propsActiveNode || null);

  const handleFullscreen = () => {
    if (containerRef.current.requestFullscreen) {
      containerRef.current.requestFullscreen();
    }
  };

  const handlePopout = () => {
    localStorage.setItem('hades_graph_data', JSON.stringify({ data, l1Nodes, activeNode }));
    window.open('/popout.html', 'HADES Graph Pop-out', 'width=1000,height=800');
  };

  // Sync props to internal state
  useEffect(() => {
    setL1Nodes(propsL1Nodes);
    setActiveNode(propsActiveNode);
  }, [propsL1Nodes, propsActiveNode]);

  // Live state updates from server when pipeline completes
  useEffect(() => {
    if (telemetry?.pipeline_stage === 'complete') {
      const fetchState = async () => {
        try {
          const res = await fetch('/api/graph/state');
          const state = await res.json();
          setL1Nodes(state.l1_nodes || []);
          setActiveNode(state.active_node || null);
        } catch (err) {
          console.error('Failed to fetch live graph state:', err);
        }
      };
      fetchState();
    }
  }, [telemetry?.pipeline_stage]);

  useEffect(() => {
    if (!data || !data.nodes || data.nodes.length === 0) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    svg.selectAll('*').remove();

    // Definitions (Filters, Markers, Gradients)
    const defs = svg.append('defs');

    // Radial Gradient Background
    const gradient = defs.append('radialGradient')
      .attr('id', 'bg-gradient')
      .attr('cx', '50%')
      .attr('cy', '50%')
      .attr('r', '50%');
    gradient.append('stop').attr('offset', '0%').attr('stop-color', '#0d1117');
    gradient.append('stop').attr('offset', '100%').attr('stop-color', '#111827');

    svg.insert('rect', ':first-child')
      .attr('width', '100%')
      .attr('height', '100%')
      .attr('fill', 'url(#bg-gradient)');

    // Green Glow Filter
    const glowFilter = defs.append('filter')
      .attr('id', 'green-glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');
    glowFilter.append('feGaussianBlur')
      .attr('stdDeviation', '3')
      .attr('result', 'blur');
    glowFilter.append('feComposite')
      .attr('in', 'SourceGraphic')
      .attr('in2', 'blur')
      .attr('operator', 'over');

    // Directional Arrow Marker
    defs.append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '-0 -5 10 10')
      .attr('refX', 20) // Positioned near node
      .attr('refY', 0)
      .attr('orient', 'auto')
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('xoverflow', 'visible')
      .append('svg:path')
      .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
      .attr('fill', '#2a3a5c')
      .style('stroke', 'none');

    const g = svg.append('g');

    const zoom = d3.zoom()
      .scaleExtent([0.1, 8])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        setZoomLevel(event.transform.k);
      });

    svg.call(zoom);

    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id(d => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(40));

    simulationRef.current = simulation;

    const link = g.append('g')
      .selectAll('line')
      .data(data.links)
      .enter().append('line')
      .attr('stroke', '#2a3a5c')
      .attr('stroke-width', 1)
      .attr('opacity', 0.6)
      .attr('marker-end', 'url(#arrowhead)');

    const node = g.append('g')
      .selectAll('g')
      .data(data.nodes)
      .enter().append('g')
      .attr('class', 'node-group')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended))
      .on('click', (event, d) => setSelectedNode(d))
      .on('mouseover', (event, d) => {
        const tooltip = d3.select(tooltipRef.current);
        const tier = l1Nodes.includes(d.id) ? 'L1 CACHE' : 'L2 MEMORY';
        tooltip.style('display', 'block')
          .html(`<div class="tooltip-title">${d.label}</div><div class="tooltip-tier">${tier}</div>`);
      })
      .on('mousemove', (event) => {
        d3.select(tooltipRef.current)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY + 10) + 'px');
      })
      .on('mouseout', () => {
        d3.select(tooltipRef.current).style('display', 'none');
      });

    // Node circles initialized
    node.append('circle')
      .attr('class', 'main-circle')
      .attr('r', 6)
      .attr('fill', '#1a2a4a')
      .attr('stroke', '#2a3a5c')
      .attr('stroke-width', 1);

    // Pulse ring placeholder
    node.append('circle')
      .attr('class', 'pulse-ring')
      .attr('r', 6)
      .attr('fill', 'none')
      .attr('opacity', 0)
      .style('pointer-events', 'none');

    const labels = node.append('text')
      .attr('dx', 12)
      .attr('dy', 4)
      .text(d => d.label)
      .attr('font-size', '10px')
      .attr('fill', 'var(--text-secondary)')
      .attr('font-family', 'var(--font-mono)')
      .style('pointer-events', 'none')
      .style('visibility', zoomLevel > 1.5 ? 'visible' : 'hidden');

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    // Initial attribute sync
    updateVisuals(l1Nodes, activeNode, false);

    const resizeObserver = new ResizeObserver(() => {
      if (containerRef.current) {
        const w = containerRef.current.clientWidth;
        const h = containerRef.current.clientHeight;
        svg.attr('width', w).attr('height', h);
      }
    });
    resizeObserver.observe(containerRef.current);
    
    return () => {
      simulation.stop();
      resizeObserver.disconnect();
    };
  }, [data]);

  // Transitions for live updates
  const updateVisuals = (l1, active, animate = true) => {
    const svg = d3.select(svgRef.current);
    if (svg.empty()) return;

    const nodes = svg.selectAll('.node-group');
    const transition = d3.transition().duration(animate ? 400 : 0);

    nodes.each(function(d) {
      const g = d3.select(this);
      const circle = g.select('.main-circle');
      const pulseRing = g.select('.pulse-ring');
      
      const isActive = d.id === active;
      const isL1 = l1.includes(d.id);

      circle.transition(transition)
        .attr('r', isActive ? 10 : (isL1 ? 8 : 6))
        .attr('fill', isActive ? '#2a1a00' : (isL1 ? '#0a2a1a' : '#1a2a4a'))
        .attr('stroke', isActive ? '#ffb400' : (isL1 ? '#00e676' : '#2a3a5c'))
        .attr('stroke-width', isActive ? 2 : (isL1 ? 1.5 : 1))
        .attr('class', isL1 && !isActive ? 'main-circle glow-green' : 'main-circle');

      if (isActive && animate) {
        pulseRing
          .attr('class', 'pulse-ring node-pulse-ring')
          .attr('r', 10)
          .attr('opacity', 1);
      } else {
        pulseRing
          .attr('class', 'pulse-ring')
          .attr('opacity', 0);
      }
    });
  };

  useEffect(() => {
    updateVisuals(l1Nodes, activeNode, true);
  }, [l1Nodes, activeNode]);

  useEffect(() => {
    d3.select(svgRef.current).selectAll('text')
      .style('visibility', zoomLevel > 1.5 ? 'visible' : 'hidden');
  }, [zoomLevel]);

  const handleReset = () => {
    d3.select(svgRef.current).transition().duration(750).call(
      d3.zoom().transform,
      d3.zoomIdentity
    );
  };

  return (
    <div className="pane graph-pane">
      <div className="graph-container" ref={containerRef}>
        <svg ref={svgRef}></svg>
        <div className="graph-tooltip" ref={tooltipRef}></div>
        
        <div className="graph-toolbar">
          <button className="toolbar-btn" title="Zoom In" onClick={() => d3.select(svgRef.current).transition().call(d3.zoom().scaleBy, 1.3)}><ZoomIn size={14} /></button>
          <button className="toolbar-btn" title="Zoom Out" onClick={() => d3.select(svgRef.current).transition().call(d3.zoom().scaleBy, 0.7)}><ZoomOut size={14} /></button>
          <button className="toolbar-btn" title="Reset View" onClick={handleReset}><Maximize2 size={14} /></button>
          <button className="toolbar-btn" title="Full Screen" onClick={handleFullscreen}><Maximize size={14} /></button>
          <button className="toolbar-btn" title="Pop-out Window" onClick={handlePopout}><ExternalLink size={14} /></button>
        </div>
      </div>

      <StatusBar telemetry={telemetry} />

      <div className="node-detail">
        <div className="detail-title">
          <Info size={12} style={{ verticalAlign: 'middle', marginRight: '4px' }} />
          Node Inspector
        </div>
        <div className="detail-content">
          {selectedNode ? (
            <div>
              <div style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>{selectedNode.label}</div>
              <div style={{ display: 'grid', gap: '4px' }}>
                <div>Tier: <span style={{ color: l1Nodes.includes(selectedNode.id) ? 'var(--accent-green)' : 'var(--accent-blue)' }}>
                  {l1Nodes.includes(selectedNode.id) ? 'L1 CACHE' : 'L2 MEMORY'}
                </span></div>
                <div>ID: {selectedNode.id}</div>
              </div>
            </div>
          ) : (
            <div style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>Select a node to view details</div>
          )}
        </div>
      </div>
    </div>
  );
};
export default GraphPanel;
