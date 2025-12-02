import { useState, useEffect } from 'react'
import { io } from 'socket.io-client'
import Header from './components/Header'
import LogStream from './components/LogStream'
import StatsPanel from './components/StatsPanel'
import AlertBanner from './components/AlertBanner'
import './App.css'

function App() {
  const [socket, setSocket] = useState(null)
  const [connected, setConnected] = useState(false)
  const [logs, setLogs] = useState([])
  const [stats, setStats] = useState({
    totalProcessed: 0,
    attacksDetected: 0,
    normalDetected: 0,
    messagesPerSecond: 0,
    attackRate: 0,
    protocolDistribution: { tcp: 0, udp: 0, icmp: 0, other: 0 },
    recentAlerts: []
  })
  const [currentAlert, setCurrentAlert] = useState(null)

  useEffect(() => {
    // Connect to backend WebSocket
    const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000'
    const newSocket = io(backendUrl, {
      transports: ['websocket', 'polling'],
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    })

    newSocket.on('connect', () => {
      console.log('Connected to backend')
      setConnected(true)
    })

    newSocket.on('disconnect', () => {
      console.log('Disconnected from backend')
      setConnected(false)
    })

    newSocket.on('classification_result', (data) => {
      // Add new log entry
      setLogs(prevLogs => {
        const newLogs = [data, ...prevLogs]
        // Keep only last 100 logs
        return newLogs.slice(0, 100)
      })

      // Update statistics
      setStats(prevStats => {
        const newStats = { ...prevStats }
        newStats.totalProcessed += 1

        if (data.prediction === 'suspicious') {
          newStats.attacksDetected += 1

          // Show alert banner
          setCurrentAlert({
            id: data.id,
            timestamp: data.timestamp,
            score: data.score,
            protocol: data.original_log?.features?.protocol_type || 'unknown'
          })

          // Add to recent alerts (keep last 5)
          newStats.recentAlerts = [
            {
              id: data.id,
              timestamp: data.timestamp,
              score: data.score,
              protocol: data.original_log?.features?.protocol_type || 'unknown'
            },
            ...newStats.recentAlerts
          ].slice(0, 5)
        } else {
          newStats.normalDetected += 1
        }

        // Calculate attack rate
        newStats.attackRate = (newStats.attacksDetected / newStats.totalProcessed * 100).toFixed(2)

        // Update protocol distribution
        const protocol = data.original_log?.features?.protocol_type
        if (protocol) {
          if (['tcp', 'udp', 'icmp'].includes(protocol)) {
            newStats.protocolDistribution[protocol] =
              (newStats.protocolDistribution[protocol] || 0) + 1
          } else {
            newStats.protocolDistribution.other =
              (newStats.protocolDistribution.other || 0) + 1
          }
        }

        return newStats
      })
    })

    newSocket.on('stats_update', (data) => {
      setStats(prevStats => ({
        ...prevStats,
        messagesPerSecond: data.messagesPerSecond || 0
      }))
    })

    setSocket(newSocket)

    // Cleanup on unmount
    return () => {
      if (newSocket) {
        newSocket.close()
      }
    }
  }, [])

  const dismissAlert = () => {
    setCurrentAlert(null)
  }

  return (
    <div className="app">
      <Header
        connected={connected}
        totalProcessed={stats.totalProcessed}
        messagesPerSecond={stats.messagesPerSecond}
      />

      {currentAlert && (
        <AlertBanner alert={currentAlert} onDismiss={dismissAlert} />
      )}

      <div className="dashboard">
        <div className="main-content">
          <LogStream logs={logs} />
        </div>

        <div className="sidebar">
          <StatsPanel stats={stats} />
        </div>
      </div>
    </div>
  )
}

export default App
