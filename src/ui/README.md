# IDS Traffic Monitor UI

Real-time web dashboard for monitoring network intrusion detection system activity. Built with React + Vite, this UI displays live classification results, statistics, and security alerts.

## Features

### **Real-Time Dashboard**
- Live traffic log stream with color-coded entries
- Green (🟢) for normal traffic
- Red (🔴) for detected attacks
- Auto-scrolling with pause/resume control

### **Statistics Panel**
- Attack detection rate percentage
- Total attacks vs normal traffic counts
- Protocol distribution pie chart (TCP, UDP, ICMP)
- Messages per second throughput

### **Alert System**
- Sliding alert banner for detected attacks
- Recent alerts list with timestamps
- Auto-dismiss after 5 seconds
- Confidence scores for each detection

### **Modern UI**
- Dark theme optimized for long monitoring sessions
- Responsive design (desktop and tablet)
- Smooth animations and transitions
- Professional security operations aesthetic

## Architecture

```
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│   Classifier    │ ───▶  │  Redis Pub/Sub   │ ───▶  │  UI Backend     │
│    Service      │       │   (Channel)      │       │   (WebSocket)   │
└─────────────────┘       └──────────────────┘       └─────────────────┘
                                                              │
                                                              │ Socket.IO
                                                              ▼
                                                      ┌─────────────────┐
                                                      │  React Frontend │
                                                      │   (Dashboard)   │
                                                      └─────────────────┘
```

**Components:**

1. **Frontend (React + Vite)**
   - Dashboard UI with real-time updates
   - WebSocket client connection
   - Charts and visualizations

2. **Backend (Python Flask + Socket.IO)**
   - Redis pub/sub → WebSocket bridge
   - Subscribes to `classification_results` channel
   - Broadcasts to connected web clients

## Prerequisites

- **Node.js** 18+ (for frontend)
- **Python** 3.11+ (for backend)
- **Redis** running and accessible
- **Classifier service** publishing results

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# From project root
docker-compose up ui-frontend ui-backend
```

The UI will be available at http://localhost:3000

### Option 2: Local Development

#### Backend Setup

```bash
# Navigate to UI directory
cd src/ui

# Install Python dependencies
pip install -r backend/requirements.txt

# Start the backend
python backend/server.py
```

Backend runs on http://localhost:5000

#### Frontend Setup

```bash
# In a new terminal, from src/ui directory
cd src/ui

# Install Node dependencies
npm install

# Start development server
npm run dev
```

Frontend runs on http://localhost:3000

## Configuration

### Environment Variables

Create `.env` file (copy from `.env.example`):

```env
# Frontend
VITE_BACKEND_URL=http://localhost:5000

# Backend
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_RESULTS_CHANNEL=classification_results
SERVER_PORT=5000
LOG_LEVEL=INFO
```

### Docker Configuration

The UI is configured in `docker-compose.yml`:

```yaml
services:
  ui-backend:
    build:
      context: ./src/ui
      dockerfile: Dockerfile.backend
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    ports:
      - "5000:5000"

  ui-frontend:
    build:
      context: ./src/ui
      dockerfile: Dockerfile
    ports:
      - "3000:80"
```

## Project Structure

```
src/ui/
├── backend/
│   ├── server.py           # WebSocket bridge service
│   └── requirements.txt    # Python dependencies
├── src/
│   ├── components/
│   │   ├── Header.jsx      # Top navigation bar
│   │   ├── LogStream.jsx   # Real-time log display
│   │   ├── LogEntry.jsx    # Individual log entry
│   │   ├── StatsPanel.jsx  # Statistics sidebar
│   │   └── AlertBanner.jsx # Security alert popup
│   ├── App.jsx             # Main application
│   ├── main.jsx            # Entry point
│   └── index.css           # Global styles
├── Dockerfile              # Frontend production build
├── Dockerfile.backend      # Backend service
├── nginx.conf              # Nginx configuration
├── package.json            # Node dependencies
├── vite.config.js          # Vite configuration
└── README.md               # This file
```

## Development

### Running in Development Mode

```bash
# Backend (with hot reload)
cd src/ui
python backend/server.py

# Frontend (with hot reload)
npm run dev
```

Changes to React components will hot-reload automatically.

### Building for Production

```bash
# Build frontend
npm run build

# Output will be in dist/
```

### Linting

```bash
npm run lint
```

## API Endpoints

### Backend WebSocket Events

**Client → Server:**
- `connect` - Establish connection
- `disconnect` - Close connection

**Server → Client:**
- `classification_result` - New classification from Redis
  ```json
  {
    "id": 123,
    "timestamp": "2024-12-02T10:30:00Z",
    "prediction": "suspicious" | "normal",
    "score": 0.85,
    "original_log": { ... }
  }
  ```

- `stats_update` - Periodic statistics
  ```json
  {
    "messagesPerSecond": 42.5,
    "totalMessages": 1523,
    "connectedClients": 2
  }
  ```

### Backend HTTP Endpoints

- `GET /health` - Health check
  ```json
  {
    "status": "healthy",
    "connected_clients": 2,
    "uptime_seconds": 3600,
    "messages_processed": 1523
  }
  ```

## Customization

### Changing Colors

Edit `src/index.css` CSS variables:

```css
:root {
  --color-normal: #00ba7c;    /* Green for normal traffic */
  --color-attack: #f4212e;    /* Red for attacks */
  --color-warning: #ff9500;   /* Orange for warnings */
  --color-info: #1d9bf0;      /* Blue for info */
}
```

### Adjusting Alert Duration

Edit `src/components/AlertBanner.jsx`:

```javascript
useEffect(() => {
  const timer = setTimeout(() => {
    onDismiss()
  }, 5000) // Change duration here (milliseconds)
  // ...
}, [alert, onDismiss])
```

### Changing Log Retention

Edit `src/App.jsx`:

```javascript
setLogs(prevLogs => {
  const newLogs = [data, ...prevLogs]
  return newLogs.slice(0, 100) // Keep last 100 logs
})
```

## Troubleshooting

### Frontend not connecting to backend

**Symptoms:** "Disconnected" status in header

**Solutions:**
1. Check backend is running: `curl http://localhost:5000/health`
2. Verify `VITE_BACKEND_URL` in `.env`
3. Check browser console for WebSocket errors
4. Ensure CORS is enabled on backend

### No data appearing

**Symptoms:** Empty log stream

**Solutions:**
1. Verify classifier service is running and publishing
2. Check Redis channel name matches: `classification_results`
3. Check backend logs: `docker logs ui-backend`
4. Test Redis: `redis-cli SUBSCRIBE classification_results`

### Chart not rendering

**Symptoms:** Protocol distribution chart missing

**Solutions:**
1. Ensure `recharts` is installed: `npm install recharts`
2. Check browser console for errors
3. Verify data format matches expected structure

## Performance

### Metrics

- **Frontend Bundle Size:** ~200KB gzipped
- **WebSocket Latency:** <10ms (local network)
- **Memory Usage:** ~50MB (browser)
- **Max Concurrent Clients:** 100+ per backend instance

### Optimization Tips

1. **Limit log retention:** Keep only last 100-200 entries
2. **Throttle updates:** Backend sends stats every 1 second
3. **Use production build:** `npm run build` for optimized bundle
4. **Enable gzip:** nginx configuration includes gzip compression

## Security Considerations

⚠️ **Important for Production:**

1. **Change SECRET_KEY** in backend `.env`
2. **Enable HTTPS** for WebSocket connections
3. **Add authentication** if exposing to internet
4. **Rate limit** WebSocket connections
5. **Validate** all incoming data
6. **Use environment variables** for sensitive config

## Testing

### Manual Testing

1. Start all services (Redis, Classifier, Generator, UI)
2. Open browser to http://localhost:3000
3. Verify connection status shows "Connected"
4. Generate traffic using traffic generator
5. Confirm logs appear in real-time
6. Check stats update correctly
7. Trigger attack to test alert banner

### Integration Testing

```bash
# Test backend health
curl http://localhost:5000/health

# Test WebSocket connection (using wscat)
npm install -g wscat
wscat -c ws://localhost:5000/socket.io/
```

## Known Issues

- **Auto-scroll:** Pauses when user manually scrolls (by design)
- **Chart performance:** May lag with >1000 messages (use log retention)
- **Mobile:** Optimized for desktop/tablet, phone support limited

## Future Enhancements

- [ ] Export logs to CSV/JSON
- [ ] Filtering by protocol/service
- [ ] Historical data visualization
- [ ] Dark/Light theme toggle
- [ ] Alert sound notifications
- [ ] User authentication
- [ ] Multi-dashboard support
- [ ] Grafana integration

## Contributing

When adding new features:

1. Follow existing component structure
2. Use CSS custom properties for colors
3. Add PropTypes for type checking
4. Update this README with new features
5. Test on Chrome, Firefox, Safari

## License

Part of the IDS ML-Ops System project.

## Support

For issues or questions:
- Check troubleshooting section above
- Review backend logs: `docker logs ui-backend`
- Check browser console for frontend errors
- Verify all services are running: `docker-compose ps`
