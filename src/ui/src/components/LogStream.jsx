import { useEffect, useRef, useState } from 'react'
import LogEntry from './LogEntry'
import './LogStream.css'

function LogStream({ logs }) {
  const streamRef = useRef(null)
  const [autoScroll, setAutoScroll] = useState(true)

  useEffect(() => {
    if (autoScroll && streamRef.current) {
      streamRef.current.scrollTop = 0
    }
  }, [logs, autoScroll])

  const handleScroll = (e) => {
    const { scrollTop } = e.target
    // If user scrolls away from top, disable auto-scroll
    setAutoScroll(scrollTop === 0)
  }

  return (
    <div className="log-stream">
      <div className="log-stream-header">
        <h2>Real-Time Traffic Log</h2>
        <div className="log-stream-controls">
          <button
            className={`auto-scroll-btn ${autoScroll ? 'active' : ''}`}
            onClick={() => setAutoScroll(!autoScroll)}
          >
            {autoScroll ? '⏸️ Pause' : '▶️ Resume'} Auto-scroll
          </button>
          <span className="log-count">{logs.length} logs</span>
        </div>
      </div>

      <div
        className="log-stream-content"
        ref={streamRef}
        onScroll={handleScroll}
      >
        {logs.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📡</div>
            <p>Waiting for traffic data...</p>
            <p className="empty-subtitle">
              Classification results will appear here in real-time
            </p>
          </div>
        ) : (
          <div className="log-entries">
            {logs.map((log, index) => (
              <LogEntry key={`${log.id}-${index}`} log={log} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default LogStream
