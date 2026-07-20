package main

import (
	"bytes"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

func validMetric() GridMetric {
	return GridMetric{
		Timestamp: "2026-07-21T10:00:00Z", CameraID: "CAM_1", ZoneID: "ZONE_1",
		FrameID: 1, GridID: "G_00_00", RiskLevel: "GREEN",
		PredictedRiskLevel: "GREEN", CrowdClass: "MODERATE",
		Confidence: 0.8, Coherence: 0.9, FlowQuality: 0.7,
		ValidFlowRatio: 0.6, CrowdProbability: 0.75,
	}
}

func TestValidateGridMetric(t *testing.T) {
	m := validMetric()
	if err := validateGridMetric(m); err != nil {
		t.Fatalf("valid metric rejected: %v", err)
	}

	m.Confidence = 1.1
	if err := validateGridMetric(m); err == nil {
		t.Fatal("out-of-range confidence was accepted")
	}
	m = validMetric()
	m.FlowX = math.NaN()
	if err := validateGridMetric(m); err == nil {
		t.Fatal("NaN telemetry was accepted")
	}
	m = validMetric()
	m.Timestamp = "not-a-timestamp"
	if err := validateGridMetric(m); err == nil {
		t.Fatal("invalid timestamp was accepted")
	}
}

func TestGridMetricKeyIsRetrySafeAndSessionScoped(t *testing.T) {
	a := validMetric()
	b := a
	if gridMetricKey(a) != gridMetricKey(b) {
		t.Fatal("identical retry produced a different key")
	}
	b.Timestamp = "2026-07-21T11:00:00Z"
	if gridMetricKey(a) == gridMetricKey(b) {
		t.Fatal("separate processing session produced the same key")
	}
}

func TestTelemetryHandlerIsIdempotent(t *testing.T) {
	originalDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	temporaryDir := t.TempDir()
	if err := os.Chdir(temporaryDir); err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(originalDir)

	initDB()
	defer db.Close()
	hub = newHub()
	go hub.run()

	payload, err := json.Marshal([]GridMetric{validMetric()})
	if err != nil {
		t.Fatal(err)
	}
	for attempt := 0; attempt < 2; attempt++ {
		request := httptest.NewRequest(http.MethodPost, "/api/v1/telemetry/grid-metrics", bytes.NewReader(payload))
		response := httptest.NewRecorder()
		handleTelemetry(response, request)
		if response.Code != http.StatusCreated {
			t.Fatalf("attempt %d failed: status=%d body=%s", attempt+1, response.Code, response.Body.String())
		}
	}

	var count int
	if err := db.QueryRow("SELECT COUNT(*) FROM grid_metrics").Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("retry created duplicate rows: got %d", count)
	}
}
