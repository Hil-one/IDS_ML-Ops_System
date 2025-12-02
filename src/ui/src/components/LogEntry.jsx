import './LogEntry.css'

function LogEntry({ log }) {
  const isAttack = log.prediction === 'suspicious'
  const timestamp = new Date(log.timestamp).toLocaleTimeString()
  const protocol = log.original_log?.features?.protocol_type || 'unknown'
  const service = log.original_log?.features?.service || 'unknown'
  const srcBytes = log.original_log?.features?.src_bytes || 0
  const dstBytes = log.original_log?.features?.dst_bytes || 0

  return (
    <div className={`log-entry ${isAttack ? 'attack' : 'normal'}`}>
      <div className="log-entry-indicator"></div>

      <div className="log-entry-content">
        <div className="log-entry-header">
          <span className="log-id">#{log.id}</span>
          <span className="log-timestamp">{timestamp}</span>
          <span className={`log-prediction ${isAttack ? 'attack' : 'normal'}`}>
            {isAttack ? '🔴 SUSPICIOUS' : '🟢 NORMAL'}
          </span>
        </div>

        <div className="log-entry-details">
          <div className="log-detail">
            <span className="detail-label">Protocol:</span>
            <span className="detail-value">{protocol.toUpperCase()}</span>
          </div>
          <div className="log-detail">
            <span className="detail-label">Service:</span>
            <span className="detail-value">{service}</span>
          </div>
          <div className="log-detail">
            <span className="detail-label">Bytes:</span>
            <span className="detail-value">
              {srcBytes.toLocaleString()} → {dstBytes.toLocaleString()}
            </span>
          </div>
        </div>

        <div className="confidence-bar-container">
          <div className="confidence-label">
            Confidence: {(log.score * 100).toFixed(1)}%
          </div>
          <div className="confidence-bar">
            <div
              className={`confidence-fill ${isAttack ? 'attack' : 'normal'}`}
              style={{ width: `${log.score * 100}%` }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LogEntry
