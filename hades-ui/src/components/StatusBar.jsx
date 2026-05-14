import React from 'react';

const STAGE_ORDER = [
  'idle',
  'uploading',
  'reading pdf',
  'healing layout',
  'extracting triples',
  'building L2 graph',
  'compressing',
  'querying',
  'verifying',
  'writing back',
  'complete'
];

function matchStageIndex(currentStage) {
  // Exact match first
  const exact = STAGE_ORDER.indexOf(currentStage);
  if (exact !== -1) return exact;
  // Prefix match (e.g. "extracting triples (p.3)" matches "extracting triples")
  for (let i = STAGE_ORDER.length - 1; i >= 0; i--) {
    if (currentStage.startsWith(STAGE_ORDER[i])) return i;
  }
  // Substring match (e.g. "compressing with charon" matches "compressing")
  for (let i = STAGE_ORDER.length - 1; i >= 0; i--) {
    if (currentStage.toLowerCase().includes(STAGE_ORDER[i])) return i;
  }
  return 0;
}

const StatusBar = ({ telemetry }) => {
  const currentStage = telemetry?.pipeline_stage || 'idle';
  const stageIndex = matchStageIndex(currentStage);
  const progress = ((stageIndex + 1) / STAGE_ORDER.length) * 100;
  
  const isProcessing = currentStage !== 'idle' && currentStage !== 'complete';
  const isVerifying = currentStage.includes('verif');

  return (
    <div className="status-bar">
      <div className="status-row">
        {isProcessing ? (
          <div className="spinner"></div>
        ) : (
          <div className="status-dot dot-idle"></div>
        )}
        <span className={isVerifying ? 'status-text-pulse' : ''} style={{ textTransform: 'uppercase' }}>
          {isProcessing ? `PIPELINE: ${currentStage}` : `SYSTEM: ${currentStage}`}
        </span>
      </div>
      
      <div className="progress-container">
        <div 
          className="progress-bar" 
          style={{ 
            width: isProcessing ? `${progress}%` : (currentStage === 'complete' ? '100%' : '0%'),
            backgroundColor: currentStage === 'complete' ? 'var(--accent-green)' : 'var(--accent-blue)'
          }}
        ></div>
      </div>
    </div>
  );
};

export default StatusBar;
