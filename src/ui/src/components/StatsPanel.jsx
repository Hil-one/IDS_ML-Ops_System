import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import './StatsPanel.css'

function StatsPanel({ stats }) {
  const protocolData = [
    { name: 'TCP', value: stats.protocolDistribution.tcp },
    { name: 'UDP', value: stats.protocolDistribution.udp },
    { name: 'ICMP', value: stats.protocolDistribution.icmp },
    { name: 'Other', value: stats.protocolDistribution.other },
  ].filter(item => item.value > 0)

  const COLORS = ['#1d9bf0', '#00ba7c', '#ff9500', '#8b5cf6']

  return (
    <div className="stats-panel">
      <div className="stats-card">
        <h3 className="stats-card-title">Detection Summary</h3>
        <div className="stats-grid">
          <div className="stat-item">
            <div className="stat-label">Attack Rate</div>
            <div className="stat-value attack-rate">{stats.attackRate}%</div>
          </div>
          <div className="stat-item">
            <div className="stat-label">Attacks Detected</div>
            <div className="stat-value attacks">
              {stats.attacksDetected.toLocaleString()}
            </div>
          </div>
          <div className="stat-item">
            <div className="stat-label">Normal Traffic</div>
            <div className="stat-value normal">
              {stats.normalDetected.toLocaleString()}
            </div>
          </div>
        </div>
      </div>

      {protocolData.length > 0 && (
        <div className="stats-card">
          <h3 className="stats-card-title">Protocol Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={protocolData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) =>
                  `${name}: ${(percent * 100).toFixed(0)}%`
                }
                outerRadius={70}
                fill="#8884d8"
                dataKey="value"
              >
                {protocolData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="stats-card">
        <h3 className="stats-card-title">Recent Alerts</h3>
        <div className="recent-alerts">
          {stats.recentAlerts.length === 0 ? (
            <div className="no-alerts">No recent attacks detected</div>
          ) : (
            stats.recentAlerts.map((alert) => (
              <div key={alert.id} className="alert-item">
                <div className="alert-icon">⚠️</div>
                <div className="alert-info">
                  <div className="alert-id">Alert #{alert.id}</div>
                  <div className="alert-details">
                    {alert.protocol.toUpperCase()} •{' '}
                    {new Date(alert.timestamp).toLocaleTimeString()} •
                    Confidence: {(alert.score * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default StatsPanel
