import { useEffect } from 'react'
import './AlertBanner.css'

function AlertBanner({ alert, onDismiss }) {
  useEffect(() => {
    const timer = setTimeout(() => {
      onDismiss()
    }, 5000) // Auto-dismiss after 5 seconds

    return () => clearTimeout(timer)
  }, [alert, onDismiss])

  if (!alert) return null

  return (
    <div className="alert-banner">
      <div className="alert-banner-content">
        <div className="alert-banner-icon">🚨</div>
        <div className="alert-banner-text">
          <div className="alert-banner-title">
            Security Alert Detected!
          </div>
          <div className="alert-banner-subtitle">
            Suspicious {alert.protocol.toUpperCase()} traffic detected with{' '}
            {(alert.score * 100).toFixed(0)}% confidence (ID: #{alert.id})
          </div>
        </div>
        <button className="alert-banner-close" onClick={onDismiss}>
          ✕
        </button>
      </div>
    </div>
  )
}

export default AlertBanner
