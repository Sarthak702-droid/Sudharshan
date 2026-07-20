package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	_ "github.com/mattn/go-sqlite3"
)

// --- Database Schema Structures ---
type Camera struct {
	ID         string    `json:"id"`
	Name       string    `json:"name"`
	RTSPURL    string    `json:"rtsp_url"`
	ZoneID     string    `json:"zone_id"`
	Status     string    `json:"status"`
	FPS        int       `json:"fps"`
	Resolution string    `json:"resolution"`
	CreatedAt  time.Time `json:"created_at"`
}

type Zone struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	Polygon   string    `json:"polygon_geojson"`
	CreatedAt time.Time `json:"created_at"`
}

type GridMetric struct {
	Timestamp       string  `json:"timestamp"`
	CameraID        string  `json:"camera_id"`
	ZoneID          string  `json:"zone_id"`
	FrameID         int     `json:"frame_id"`
	GridID          string  `json:"grid_id"`
	Count           float64 `json:"count"`
	Density         float64 `json:"density"`
	FlowX           float64 `json:"flow_x"`
	FlowY           float64 `json:"flow_y"`
	DirectionDeg    float64 `json:"direction_deg"`
	DirectionLabel  string  `json:"direction_label"`
	RelativeSpeed   float64 `json:"relative_speed"`
	SpeedLevel      string  `json:"speed_level"`
	Coherence       float64 `json:"coherence"`
	ReverseScore    float64 `json:"reverse_score"`
	ConflictScore   float64 `json:"conflict_score"`
	CongestionScore float64 `json:"congestion_score"`
	RiskLevel       string  `json:"risk_level"`
	Confidence      float64 `json:"confidence"`
	// Robust AI Features
	TurbulenceScore          float64 `json:"turbulence_score"`
	SpeedSurgeWarning        bool    `json:"speed_surge_warning"`
	StasisWarning            bool    `json:"stasis_warning"`
	TurbulenceWarning        bool    `json:"turbulence_warning"`
	CrowdPresent             bool    `json:"crowd_present"`
	CrowdClass               string  `json:"crowd_class"`
	CrowdProbability         float64 `json:"crowd_probability"`
	FlowQuality              float64 `json:"flow_quality"`
	ValidFlowRatio           float64 `json:"valid_flow_ratio"`
	AlertEligible            bool    `json:"alert_eligible"`
	PhysicalCalibrated       bool    `json:"physical_calibrated"`
	SpeedMPS                 float64 `json:"speed_mps"`
	DensityPeopleM2          float64 `json:"density_people_m2"`
	Divergence               float64 `json:"divergence"`
	Acceleration             float64 `json:"acceleration"`
	PredictedCount           float64 `json:"predicted_count"`
	PredictedCongestionScore float64 `json:"predicted_congestion_score"`
	PredictedRiskLevel       string  `json:"predicted_risk_level"`
	TrendSlope               float64 `json:"trend_slope"`
}

type Alert struct {
	ID             int64          `json:"id"`
	Timestamp      string         `json:"timestamp"`
	CameraID       string         `json:"camera_id"`
	GridID         string         `json:"grid_id"`
	Severity       string         `json:"severity"`
	Type           string         `json:"type"`
	Status         string         `json:"status"` // NEW, ACKNOWLEDGED, RESOLVED
	AcknowledgedBy sql.NullString `json:"acknowledged_by,omitempty"`
	AcknowledgedAt sql.NullTime   `json:"acknowledged_at,omitempty"`
	ResolvedAt     sql.NullTime   `json:"resolved_at,omitempty"`
	Notes          string         `json:"notes"`
}

// --- WebSocket Broadcast Hub ---
type Hub struct {
	clients    map[*websocket.Conn]bool
	broadcast  chan []byte
	register   chan *websocket.Conn
	unregister chan *websocket.Conn
	mu         sync.Mutex
}

func newHub() *Hub {
	return &Hub{
		clients:    make(map[*websocket.Conn]bool),
		broadcast:  make(chan []byte, 16),
		register:   make(chan *websocket.Conn),
		unregister: make(chan *websocket.Conn),
	}
}

func (h *Hub) run() {
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			h.mu.Unlock()
			log.Printf("WebSocket client connected. Total clients: %d", len(h.clients))
		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				client.Close()
			}
			h.mu.Unlock()
			log.Printf("WebSocket client disconnected. Total clients: %d", len(h.clients))
		case message := <-h.broadcast:
			h.mu.Lock()
			for client := range h.clients {
				client.SetWriteDeadline(time.Now().Add(2 * time.Second))
				err := client.WriteMessage(websocket.TextMessage, message)
				if err != nil {
					log.Printf("WebSocket error, closing connection: %v", err)
					client.Close()
					delete(h.clients, client)
				}
			}
			h.mu.Unlock()
		}
	}
}

func (h *Hub) clientCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.clients)
}

// Global Application State
var (
	db       *sql.DB
	hub      *Hub
	upgrader = websocket.Upgrader{
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
		CheckOrigin: func(r *http.Request) bool {
			return true // Enable CORS for development
		},
	}
)

func initDB() {
	var err error
	dbPath := "./sudharshan.db"
	db, err = sql.Open("sqlite3", dbPath)
	if err != nil {
		log.Fatalf("Error opening database file: %v", err)
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000; PRAGMA foreign_keys=ON;"); err != nil {
		log.Printf("Warning: Could not enable all SQLite safety pragmas: %v", err)
	}

	// Create tables if they do not exist
	schemas := []string{
		`CREATE TABLE IF NOT EXISTS cameras (
			id TEXT PRIMARY KEY,
			name TEXT,
			rtsp_url TEXT,
			zone_id TEXT,
			status TEXT,
			fps INTEGER,
			resolution TEXT,
			created_at DATETIME
		);`,
		`CREATE TABLE IF NOT EXISTS zones (
			id TEXT PRIMARY KEY,
			name TEXT,
			polygon_geojson TEXT,
			created_at DATETIME
		);`,
		`CREATE TABLE IF NOT EXISTS grid_metrics (
			metric_key TEXT UNIQUE,
			timestamp TEXT,
			camera_id TEXT,
			zone_id TEXT,
			frame_id INTEGER,
			grid_id TEXT,
			count REAL,
			density REAL,
			flow_x REAL,
			flow_y REAL,
			direction_deg REAL,
			direction_label TEXT,
			relative_speed REAL,
			speed_level TEXT,
			coherence REAL,
			reverse_score REAL,
			conflict_score REAL,
			congestion_score REAL,
			risk_level TEXT,
			confidence REAL,
			turbulence_score REAL,
			speed_surge_warning INTEGER,
			stasis_warning INTEGER,
			turbulence_warning INTEGER,
			crowd_present INTEGER,
			crowd_class TEXT,
			crowd_probability REAL,
			flow_quality REAL,
			valid_flow_ratio REAL,
			alert_eligible INTEGER,
			physical_calibrated INTEGER,
			speed_mps REAL,
			density_people_m2 REAL,
			divergence REAL,
			acceleration REAL,
			predicted_count REAL,
			predicted_congestion_score REAL,
			predicted_risk_level TEXT,
			trend_slope REAL
		);`,
		`CREATE TABLE IF NOT EXISTS alerts (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT,
			camera_id TEXT,
			grid_id TEXT,
			severity TEXT,
			type TEXT,
			status TEXT,
			acknowledged_by TEXT,
			acknowledged_at DATETIME,
			resolved_at DATETIME,
			notes TEXT
		);`,
	}

	for _, s := range schemas {
		_, err := db.Exec(s)
		if err != nil {
			log.Fatalf("Error running schema creation: %v", err)
		}
	}

	// Migration logic: Add columns if they do not exist
	alterations := []struct {
		column string
		sql    string
	}{
		{"speed_surge_warning", "ALTER TABLE grid_metrics ADD COLUMN speed_surge_warning INTEGER DEFAULT 0;"},
		{"stasis_warning", "ALTER TABLE grid_metrics ADD COLUMN stasis_warning INTEGER DEFAULT 0;"},
		{"turbulence_warning", "ALTER TABLE grid_metrics ADD COLUMN turbulence_warning INTEGER DEFAULT 0;"},
		{"crowd_present", "ALTER TABLE grid_metrics ADD COLUMN crowd_present INTEGER DEFAULT 0;"},
		{"metric_key", "ALTER TABLE grid_metrics ADD COLUMN metric_key TEXT;"},
		{"crowd_class", "ALTER TABLE grid_metrics ADD COLUMN crowd_class TEXT DEFAULT 'EMPTY';"},
		{"crowd_probability", "ALTER TABLE grid_metrics ADD COLUMN crowd_probability REAL DEFAULT 0;"},
		{"flow_quality", "ALTER TABLE grid_metrics ADD COLUMN flow_quality REAL DEFAULT 0;"},
		{"valid_flow_ratio", "ALTER TABLE grid_metrics ADD COLUMN valid_flow_ratio REAL DEFAULT 0;"},
		{"alert_eligible", "ALTER TABLE grid_metrics ADD COLUMN alert_eligible INTEGER DEFAULT 0;"},
		{"physical_calibrated", "ALTER TABLE grid_metrics ADD COLUMN physical_calibrated INTEGER DEFAULT 0;"},
		{"speed_mps", "ALTER TABLE grid_metrics ADD COLUMN speed_mps REAL DEFAULT 0;"},
		{"density_people_m2", "ALTER TABLE grid_metrics ADD COLUMN density_people_m2 REAL DEFAULT 0;"},
		{"divergence", "ALTER TABLE grid_metrics ADD COLUMN divergence REAL DEFAULT 0;"},
		{"acceleration", "ALTER TABLE grid_metrics ADD COLUMN acceleration REAL DEFAULT 0;"},
		{"predicted_count", "ALTER TABLE grid_metrics ADD COLUMN predicted_count REAL DEFAULT 0;"},
		{"predicted_congestion_score", "ALTER TABLE grid_metrics ADD COLUMN predicted_congestion_score REAL DEFAULT 0;"},
		{"predicted_risk_level", "ALTER TABLE grid_metrics ADD COLUMN predicted_risk_level TEXT DEFAULT 'GREEN';"},
		{"trend_slope", "ALTER TABLE grid_metrics ADD COLUMN trend_slope REAL DEFAULT 0;"},
	}
	for _, alt := range alterations {
		// Check if column exists by querying pragma
		rows, err := db.Query("PRAGMA table_info(grid_metrics)")
		if err != nil {
			log.Printf("Warning: Could not query table info for migrations: %v", err)
			continue
		}

		hasColumn := false
		for rows.Next() {
			var cid int
			var name, ctype string
			var notnull, pk int
			var dfltVal interface{}
			if err := rows.Scan(&cid, &name, &ctype, &notnull, &dfltVal, &pk); err == nil {
				if name == alt.column {
					hasColumn = true
					break
				}
			}
		}
		rows.Close()

		if !hasColumn {
			log.Printf("Migrating database: Adding column %s to grid_metrics table...", alt.column)
			_, err = db.Exec(alt.sql)
			if err != nil {
				log.Printf("Warning: Failed to add column %s: %v", alt.column, err)
			} else {
				log.Printf("Successfully added column %s to grid_metrics.", alt.column)
			}
		}
	}
	if _, err := db.Exec(`CREATE UNIQUE INDEX IF NOT EXISTS idx_grid_metrics_metric_key ON grid_metrics(metric_key)`); err != nil {
		log.Printf("Warning: Failed to create telemetry idempotency index: %v", err)
	}

	log.Println("SQLite database and schema initialized successfully.")
}

// CORS Middleware
func enableCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PATCH")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// --- Handler Functions ---

func handleHealth(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":    "UP",
		"database":  "CONNECTED",
		"timestamp": time.Now().Format(time.RFC3339),
		"version":   "1.0.0-GoBackend",
		"websocket": hub.clientCount(),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func finite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func validateGridMetric(m GridMetric) error {
	if strings.TrimSpace(m.CameraID) == "" || strings.TrimSpace(m.ZoneID) == "" || strings.TrimSpace(m.GridID) == "" {
		return fmt.Errorf("camera_id, zone_id and grid_id are required")
	}
	if m.FrameID < 0 || m.Count < 0 || m.Density < 0 {
		return fmt.Errorf("frame_id, count and density must be non-negative")
	}
	if !finite(m.Count, m.Density, m.FlowX, m.FlowY, m.DirectionDeg, m.RelativeSpeed,
		m.Coherence, m.ReverseScore, m.ConflictScore, m.CongestionScore, m.Confidence,
		m.TurbulenceScore, m.CrowdProbability, m.FlowQuality, m.ValidFlowRatio,
		m.SpeedMPS, m.DensityPeopleM2, m.Divergence, m.Acceleration,
		m.PredictedCount, m.PredictedCongestionScore, m.TrendSlope) {
		return fmt.Errorf("numeric telemetry must be finite")
	}
	unitValues := map[string]float64{
		"coherence": m.Coherence, "reverse_score": m.ReverseScore,
		"conflict_score": m.ConflictScore, "confidence": m.Confidence,
		"turbulence_score": m.TurbulenceScore, "crowd_probability": m.CrowdProbability,
		"flow_quality": m.FlowQuality, "valid_flow_ratio": m.ValidFlowRatio,
	}
	for name, value := range unitValues {
		if value < 0 || value > 1 {
			return fmt.Errorf("%s must be between 0 and 1", name)
		}
	}
	if m.CongestionScore < 0 || m.CongestionScore > 100 ||
		m.PredictedCongestionScore < 0 || m.PredictedCongestionScore > 100 {
		return fmt.Errorf("congestion scores must be between 0 and 100")
	}
	validRisks := map[string]bool{"GREEN": true, "YELLOW": true, "ORANGE": true, "RED": true}
	if !validRisks[m.RiskLevel] || (m.PredictedRiskLevel != "" && !validRisks[m.PredictedRiskLevel]) {
		return fmt.Errorf("invalid risk level")
	}
	validCrowdClasses := map[string]bool{"": true, "EMPTY": true, "SPARSE": true, "MODERATE": true, "DENSE": true, "CRITICAL": true}
	if !validCrowdClasses[m.CrowdClass] {
		return fmt.Errorf("invalid crowd_class")
	}
	if m.Timestamp != "" {
		if _, err := time.Parse(time.RFC3339, m.Timestamp); err != nil {
			return fmt.Errorf("timestamp must use RFC3339")
		}
	}
	return nil
}

func gridMetricKey(m GridMetric) string {
	return fmt.Sprintf("%s:%s:%d:%s", m.CameraID, m.Timestamp, m.FrameID, m.GridID)
}

func handleTelemetry(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 8<<20)
	var metrics []GridMetric
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	err := decoder.Decode(&metrics)
	if err != nil {
		http.Error(w, "Bad Request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if len(metrics) == 0 || len(metrics) > 5000 {
		http.Error(w, "Batch must contain between 1 and 5000 metrics", http.StatusBadRequest)
		return
	}
	for index, metric := range metrics {
		if err := validateGridMetric(metric); err != nil {
			http.Error(w, fmt.Sprintf("Invalid metric at index %d: %v", index, err), http.StatusBadRequest)
			return
		}
	}

	// Begin SQL transaction
	tx, err := db.Begin()
	if err != nil {
		http.Error(w, "Internal Server Error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	stmt, err := tx.Prepare(`INSERT OR IGNORE INTO grid_metrics (
		metric_key, timestamp, camera_id, zone_id, frame_id, grid_id, count, density,
		flow_x, flow_y, direction_deg, direction_label, relative_speed,
		speed_level, coherence, reverse_score, conflict_score, congestion_score,
		risk_level, confidence, turbulence_score, speed_surge_warning,
		stasis_warning, turbulence_warning, crowd_present, crowd_class,
		crowd_probability, flow_quality, valid_flow_ratio, alert_eligible,
		physical_calibrated, speed_mps, density_people_m2, divergence, acceleration,
		predicted_count, predicted_congestion_score, predicted_risk_level, trend_slope
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		tx.Rollback()
		http.Error(w, "Internal Server Error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer stmt.Close()

	nowStr := time.Now().Format(time.RFC3339)
	insertedMetrics := make([]GridMetric, 0, len(metrics))
	for _, m := range metrics {
		if m.Timestamp == "" {
			m.Timestamp = nowStr
		}

		speedSurgeInt := 0
		if m.SpeedSurgeWarning {
			speedSurgeInt = 1
		}
		stasisInt := 0
		if m.StasisWarning {
			stasisInt = 1
		}
		turbulenceInt := 0
		if m.TurbulenceWarning {
			turbulenceInt = 1
		}
		crowdPresentInt := 0
		if m.CrowdPresent {
			crowdPresentInt = 1
		}
		alertEligibleInt := 0
		if m.AlertEligible {
			alertEligibleInt = 1
		}
		physicalCalibratedInt := 0
		if m.PhysicalCalibrated {
			physicalCalibratedInt = 1
		}
		metricKey := gridMetricKey(m)

		result, execErr := stmt.Exec(
			metricKey, m.Timestamp, m.CameraID, m.ZoneID, m.FrameID, m.GridID, m.Count, m.Density,
			m.FlowX, m.FlowY, m.DirectionDeg, m.DirectionLabel, m.RelativeSpeed,
			m.SpeedLevel, m.Coherence, m.ReverseScore, m.ConflictScore, m.CongestionScore,
			m.RiskLevel, m.Confidence, m.TurbulenceScore, speedSurgeInt,
			stasisInt, turbulenceInt, crowdPresentInt, m.CrowdClass,
			m.CrowdProbability, m.FlowQuality, m.ValidFlowRatio, alertEligibleInt,
			physicalCalibratedInt, m.SpeedMPS, m.DensityPeopleM2, m.Divergence, m.Acceleration,
			m.PredictedCount, m.PredictedCongestionScore, m.PredictedRiskLevel, m.TrendSlope,
		)
		if execErr != nil {
			tx.Rollback()
			http.Error(w, "Database Error: "+execErr.Error(), http.StatusInternalServerError)
			return
		}
		rowsAffected, _ := result.RowsAffected()
		if rowsAffected == 0 {
			continue // Idempotent retry: do not duplicate telemetry or alerts.
		}
		insertedMetrics = append(insertedMetrics, m)

		// Automated Alert Triggers based on new Robust AI Features
		if m.AlertEligible && m.SpeedSurgeWarning {
			_, err = tx.Exec(`INSERT INTO alerts (
				timestamp, camera_id, grid_id, severity, type, status, notes
			) VALUES (?, ?, ?, 'RED', 'SPEED_SURGE', 'NEW', ?)`,
				m.Timestamp, m.CameraID, m.GridID, "[EXPERIMENTAL] Automated alert: Sudden crowd speed surge (possible panic running / stampede risk).")
			if err != nil {
				log.Printf("Error creating speed surge alert: %v", err)
			}
		}

		if m.AlertEligible && m.StasisWarning {
			_, err = tx.Exec(`INSERT INTO alerts (
				timestamp, camera_id, grid_id, severity, type, status, notes
			) VALUES (?, ?, ?, 'RED', 'CROWD_CRUSH_HAZARD', 'NEW', ?)`,
				m.Timestamp, m.CameraID, m.GridID, "[EXPERIMENTAL] Automated alert: Sustained crowd compression / stasis (crush hazard).")
			if err != nil {
				log.Printf("Error creating stasis alert: %v", err)
			}
		}

		if m.AlertEligible && m.TurbulenceWarning {
			_, err = tx.Exec(`INSERT INTO alerts (
				timestamp, camera_id, grid_id, severity, type, status, notes
			) VALUES (?, ?, ?, 'ORANGE', 'FLOW_TURBULENCE', 'NEW', ?)`,
				m.Timestamp, m.CameraID, m.GridID, "[EXPERIMENTAL] Automated alert: Chaotic multidirectional crowd flow conflict (turbulence hazard).")
			if err != nil {
				log.Printf("Error creating turbulence alert: %v", err)
			}
		}

		// Default congestion buildup alert (if not covered by other alarms)
		if m.AlertEligible && !m.SpeedSurgeWarning && !m.StasisWarning && !m.TurbulenceWarning {
			if m.RiskLevel == "ORANGE" || m.RiskLevel == "RED" || m.ReverseScore > 0.5 {
				_, err = tx.Exec(`INSERT INTO alerts (
					timestamp, camera_id, grid_id, severity, type, status, notes
				) VALUES (?, ?, ?, ?, ?, 'NEW', ?)`,
					m.Timestamp, m.CameraID, m.GridID, m.RiskLevel, "CONGESTION_BUILDUP", "[EXPERIMENTAL] Automated density congestion alert.")
				if err != nil {
					log.Printf("Error creating congestion alert: %v", err)
				}
			}
		}
	}

	err = tx.Commit()
	if err != nil {
		http.Error(w, "Database Commit Error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Broadcast metrics to all WebSockets
	if len(insertedMetrics) > 0 {
		dataBytes, _ := json.Marshal(insertedMetrics)
		select {
		case hub.broadcast <- dataBytes:
		default:
			log.Printf("Dropping stale WebSocket telemetry batch; clients are not keeping up")
		}
	}

	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true, "inserted": len(insertedMetrics), "duplicates": len(metrics) - len(insertedMetrics),
	})
}

func handleGetLiveState(w http.ResponseWriter, r *http.Request) {
	// Query latest telemetry metrics per grid including robust AI fields
	rows, err := db.Query(`
		SELECT timestamp, camera_id, zone_id, frame_id, grid_id, count, density,
		       flow_x, flow_y, direction_deg, direction_label, relative_speed,
		       speed_level, coherence, reverse_score, conflict_score, congestion_score,
		       risk_level, confidence, turbulence_score, speed_surge_warning,
		       stasis_warning, turbulence_warning, crowd_present, crowd_class,
		       crowd_probability, flow_quality, valid_flow_ratio, alert_eligible,
		       physical_calibrated, speed_mps, density_people_m2, divergence, acceleration,
		       predicted_count, predicted_congestion_score, predicted_risk_level, trend_slope
		FROM grid_metrics
		WHERE rowid IN (
			SELECT MAX(rowid) FROM grid_metrics GROUP BY camera_id, zone_id, grid_id
		)
	`)
	if err != nil {
		http.Error(w, "Database Error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var list []GridMetric
	for rows.Next() {
		var m GridMetric
		var speedSurgeInt, stasisInt, turbulenceInt, crowdPresentInt, alertEligibleInt, physicalCalibratedInt int
		err = rows.Scan(
			&m.Timestamp, &m.CameraID, &m.ZoneID, &m.FrameID, &m.GridID, &m.Count, &m.Density,
			&m.FlowX, &m.FlowY, &m.DirectionDeg, &m.DirectionLabel, &m.RelativeSpeed,
			&m.SpeedLevel, &m.Coherence, &m.ReverseScore, &m.ConflictScore, &m.CongestionScore,
			&m.RiskLevel, &m.Confidence, &m.TurbulenceScore, &speedSurgeInt,
			&stasisInt, &turbulenceInt, &crowdPresentInt, &m.CrowdClass,
			&m.CrowdProbability, &m.FlowQuality, &m.ValidFlowRatio, &alertEligibleInt,
			&physicalCalibratedInt, &m.SpeedMPS, &m.DensityPeopleM2, &m.Divergence, &m.Acceleration,
			&m.PredictedCount, &m.PredictedCongestionScore, &m.PredictedRiskLevel, &m.TrendSlope,
		)
		if err == nil {
			m.SpeedSurgeWarning = speedSurgeInt == 1
			m.StasisWarning = stasisInt == 1
			m.TurbulenceWarning = turbulenceInt == 1
			m.CrowdPresent = crowdPresentInt == 1
			m.AlertEligible = alertEligibleInt == 1
			m.PhysicalCalibrated = physicalCalibratedInt == 1
			list = append(list, m)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func handleCameras(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		var c Camera
		err := json.NewDecoder(r.Body).Decode(&c)
		if err != nil {
			http.Error(w, "Bad Request", http.StatusBadRequest)
			return
		}
		c.CreatedAt = time.Now()
		_, err = db.Exec(`INSERT INTO cameras (id, name, rtsp_url, zone_id, status, fps, resolution, created_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			c.ID, c.Name, c.RTSPURL, c.ZoneID, c.Status, c.FPS, c.Resolution, c.CreatedAt)
		if err != nil {
			http.Error(w, "Database Error: "+err.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(c)
		return
	}

	rows, err := db.Query(`SELECT id, name, rtsp_url, zone_id, status, fps, resolution, created_at FROM cameras`)
	if err != nil {
		http.Error(w, "Database Error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var cameras []Camera
	for rows.Next() {
		var c Camera
		rows.Scan(&c.ID, &c.Name, &c.RTSPURL, &c.ZoneID, &c.Status, &c.FPS, &c.Resolution, &c.CreatedAt)
		cameras = append(cameras, c)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(cameras)
}

func handleGetAlerts(w http.ResponseWriter, r *http.Request) {
	rows, err := db.Query(`SELECT id, timestamp, camera_id, grid_id, severity, type, status, notes FROM alerts WHERE status != 'RESOLVED' ORDER BY id DESC`)
	if err != nil {
		http.Error(w, "Database Error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var alerts []Alert
	for rows.Next() {
		var a Alert
		rows.Scan(&a.ID, &a.Timestamp, &a.CameraID, &a.GridID, &a.Severity, &a.Type, &a.Status, &a.Notes)
		alerts = append(alerts, a)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(alerts)
}

func handleAcknowledgeAlert(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPatch {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	alertID := r.URL.Path[len("/api/v1/alerts/"):]
	alertID = alertID[:len(alertID)-len("/acknowledge")]

	now := time.Now()
	_, err := db.Exec(`UPDATE alerts SET status = 'ACKNOWLEDGED', acknowledged_by = 'OPERATOR', acknowledged_at = ? WHERE id = ?`, now, alertID)
	if err != nil {
		http.Error(w, "Database Error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"success":true}`))
}

func serveWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket Upgrade Error: %v", err)
		return
	}
	hub.register <- conn

	// Keep-alive read loop to check for clean disconnects
	go func() {
		defer func() {
			hub.unregister <- conn
		}()
		for {
			_, _, err := conn.ReadMessage()
			if err != nil {
				break
			}
		}
	}()
}

func main() {
	log.Println("Starting Sudharshan AI - Go Backend Engine...")

	initDB()

	hub = newHub()
	go hub.run()

	mux := http.NewServeMux()

	// API Gateway routes
	mux.HandleFunc("/api/v1/system/health", handleHealth)
	mux.HandleFunc("/api/v1/telemetry/grid-metrics", handleTelemetry)
	mux.HandleFunc("/api/v1/live/grid-state", func(w http.ResponseWriter, r *http.Request) {
		handleGetLiveState(w, r)
	})
	mux.HandleFunc("/api/v1/cameras", handleCameras)
	mux.HandleFunc("/api/v1/alerts", handleGetAlerts)

	// WebSocket upgrade route
	mux.HandleFunc("/api/v1/live/ws", serveWebSocket)

	// Acknowledge alert routing check
	mux.HandleFunc("/api/v1/alerts/", func(w http.ResponseWriter, r *http.Request) {
		handleAcknowledgeAlert(w, r)
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	serverAddr := ":" + port
	log.Printf("Go server successfully listening on %s", serverAddr)

	err := http.ListenAndServe(serverAddr, enableCORS(mux))
	if err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
