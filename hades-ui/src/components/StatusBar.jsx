import React from 'react';

const STAGES = [
  'idle',
  'uploading',
  'extracting triples',
  'building L2 graph',
  'compressing (Charon)',
  'querying',
  'verifying (Cerberus)',
  'writing back',
  'complete'
];

const StatusBar = ({ telemetry }) => {
  const currentStage = telemetry?.pipeline_stage || 'idle';
  const stageIndex = STAGES.indexOf(currentStage);
  const progress = ((stageIndex + 1) / STAGES.length) * 100;
  
  const isProcessing = currentStage !== 'idle' && currentStage !== 'complete';
  const isVerifying = currentStage === 'verifying (Cerberus)';

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
