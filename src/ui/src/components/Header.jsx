import './Header.css'

function Header({ connected, totalProcessed, messagesPerSecond }) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="logo">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
            <path
              d="M12 2L4 6v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V6l-8-4z"
              fill="currentColor"
              opacity="0.2"
            />
            <path
              d="M12 2L4 6v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V6l-8-4z"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <h1>IDS Traffic Monitor</h1>
        </div>
      </div>

      <div className="header-center">
        <div className="status-indicator">
          <div className={`status-dot ${connected ? 'connected' : 'disconnected'}`}></div>
          <span>{connected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </div>

      <div className="header-right">
        <div className="metric">
          <span className="metric-label">Messages Processed</span>
          <span className="metric-value">{totalProcessed.toLocaleString()}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Rate</span>
          <span className="metric-value">{messagesPerSecond.toFixed(1)} msg/s</span>
        </div>
      </div>
    </header>
  )
}

export default Header
