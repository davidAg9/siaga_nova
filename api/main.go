// SIAGA API: the cross-border health-data interoperability layer (SDG 17).
//
// A single static Go binary with no external dependencies, so it deploys and runs
// anywhere, including the low-resource, intermittently-connected environments the
// hackathon brief describes. It serves one harmonized schema over ASEAN health
// indicators, the 5-year disease forecasts, the model's driver ranking, and a
// lightweight edge-inference endpoint for life-expectancy what-if queries.
//
// Endpoints:
//
//	GET  /health                              liveness
//	GET  /countries                           list of member states + ISO3
//	GET  /indicators?country=&indicator=&from=&to=   harmonized panel rows
//	GET  /forecast?disease=&country=          5-year TB/malaria forecasts
//	GET  /drivers                             SHAP driver ranking (years of LE)
//	POST /predict                             edge life-expectancy inference
//
// Run:  cd siaga/api && go run .        (serves on :8080, override with PORT)
package main

import (
	"encoding/csv"
	"encoding/json"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

var (
	panel     []map[string]any // asean_panel.csv rows
	forecasts []map[string]any // disease_forecasts.csv rows
	drivers   map[string]float64
	surrogate LinearSurrogate
	countries []Country
)

// LinearSurrogate is the standardized ridge approximation of the reference GBM,
// exported from the analytics layer. predict = intercept + sum(coef*(x-mean)/std).
type LinearSurrogate struct {
	Features  []string  `json:"features"`
	Mean      []float64 `json:"mean"`
	Std       []float64 `json:"std"`
	Coef      []float64 `json:"coef"`
	Intercept float64   `json:"intercept"`
	R2        float64   `json:"in_sample_r2"`
}

type Country struct {
	Name string `json:"country"`
	ISO3 string `json:"iso3"`
}

func dataDir() string {
	if d := os.Getenv("SIAGA_DATA"); d != "" {
		return d
	}
	return "data"
}

// loadCSV reads a CSV into a slice of maps, converting numeric-looking cells to
// float64 and leaving blanks out entirely (so JSON omits them rather than sending 0).
func loadCSV(name string) ([]map[string]any, error) {
	f, err := os.Open(filepath.Join(dataDir(), name))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	rows, err := csv.NewReader(f).ReadAll()
	if err != nil || len(rows) == 0 {
		return nil, err
	}
	head := rows[0]
	out := make([]map[string]any, 0, len(rows)-1)
	for _, r := range rows[1:] {
		m := make(map[string]any, len(head))
		for i, h := range head {
			if i >= len(r) || strings.TrimSpace(r[i]) == "" {
				continue
			}
			if v, err := strconv.ParseFloat(strings.TrimSpace(r[i]), 64); err == nil {
				m[h] = v
			} else {
				m[h] = r[i]
			}
		}
		out = append(out, m)
	}
	return out, nil
}

func loadJSON(name string, dst any) error {
	b, err := os.ReadFile(filepath.Join(dataDir(), name))
	if err != nil {
		return err
	}
	return json.Unmarshal(b, dst)
}

func mustLoad() {
	var err error
	if panel, err = loadCSV("asean_panel.csv"); err != nil {
		log.Fatalf("load panel: %v", err)
	}
	if forecasts, err = loadCSV("disease_forecasts.csv"); err != nil {
		log.Fatalf("load forecasts: %v", err)
	}
	if err = loadJSON("shap_importance.json", &drivers); err != nil {
		log.Fatalf("load drivers: %v", err)
	}
	if err = loadJSON("linear_surrogate.json", &surrogate); err != nil {
		log.Fatalf("load surrogate: %v", err)
	}
	// Derive the country list from the panel, preserving first-seen order.
	seen := map[string]bool{}
	for _, r := range panel {
		name, _ := r["country"].(string)
		iso, _ := r["iso3"].(string)
		if name != "" && !seen[name] {
			seen[name] = true
			countries = append(countries, Country{name, iso})
		}
	}
	log.Printf("loaded %d panel rows, %d forecast rows, %d drivers, %d countries",
		len(panel), len(forecasts), len(drivers), len(countries))
}

// writeJSON emits v with permissive CORS so the Streamlit dashboard (a different
// origin) can call the API directly from the browser.
func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	json.NewEncoder(w).Encode(v)
}

func eqFold(a any, b string) bool {
	s, ok := a.(string)
	return ok && strings.EqualFold(s, b)
}

func handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, map[string]any{"status": "ok", "service": "siaga-api",
		"panel_rows": len(panel), "surrogate_r2": surrogate.R2})
}

func handleCountries(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, countries)
}

func handleIndicators(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	country, indicator := q.Get("country"), q.Get("indicator")
	from, _ := strconv.Atoi(q.Get("from"))
	to, _ := strconv.Atoi(q.Get("to"))
	out := make([]map[string]any, 0)
	for _, row := range panel {
		if country != "" && !eqFold(row["country"], country) {
			continue
		}
		if yr, ok := row["year"].(float64); ok {
			if from != 0 && int(yr) < from {
				continue
			}
			if to != 0 && int(yr) > to {
				continue
			}
		}
		if indicator != "" {
			v, ok := row[indicator]
			if !ok {
				continue
			}
			out = append(out, map[string]any{"country": row["country"], "iso3": row["iso3"],
				"year": row["year"], "indicator": indicator, "value": v})
		} else {
			out = append(out, row)
		}
	}
	writeJSON(w, out)
}

func handleForecast(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	disease, country := q.Get("disease"), q.Get("country")
	out := make([]map[string]any, 0)
	for _, row := range forecasts {
		if disease != "" && !eqFold(row["disease"], disease) {
			continue
		}
		if country != "" && !eqFold(row["country"], country) {
			continue
		}
		out = append(out, row)
	}
	writeJSON(w, out)
}

func handleDrivers(w http.ResponseWriter, _ *http.Request) {
	type d struct {
		Feature string  `json:"feature"`
		Impact  float64 `json:"mean_abs_shap_years"`
	}
	out := make([]d, 0, len(drivers))
	for k, v := range drivers {
		out = append(out, d{k, v})
	}
	// simple descending sort by impact
	for i := range out {
		for j := i + 1; j < len(out); j++ {
			if out[j].Impact > out[i].Impact {
				out[i], out[j] = out[j], out[i]
			}
		}
	}
	writeJSON(w, out)
}

// handlePredict runs the standardized linear surrogate on a feature map. Missing
// features fall back to the training mean (contributing zero), matching how the
// edge model is meant to degrade gracefully on partial field data.
func handlePredict(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Features map[string]float64 `json:"features"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Features == nil {
		http.Error(w, `{"error":"expected JSON body {\"features\": {name: value}}"}`, http.StatusBadRequest)
		return
	}
	pred := surrogate.Intercept
	used := 0
	for i, f := range surrogate.Features {
		x, ok := body.Features[f]
		if !ok || surrogate.Std[i] == 0 {
			continue // absent feature contributes 0 (it sits at the mean)
		}
		pred += surrogate.Coef[i] * (x - surrogate.Mean[i]) / surrogate.Std[i]
		used++
	}
	writeJSON(w, map[string]any{
		"predicted_life_expectancy": math.Round(pred*100) / 100,
		"features_used":            used,
		"model":                    "linear_surrogate",
		"note":                     "edge approximation of the reference GBM; see notebook for the primary model",
	})
}

func main() {
	mustLoad()
	loadAPIKeys()
	rl := newRateLimiter(rateFromEnv())
	log.Printf("rate limit: %d requests/min per client IP", rateFromEnv())

	mux := http.NewServeMux()
	// /health is exempt from auth and rate limiting (liveness probes).
	mux.HandleFunc("/health", withMiddleware(rl, true, handleHealth))
	mux.HandleFunc("/countries", withMiddleware(rl, false, handleCountries))
	mux.HandleFunc("/indicators", withMiddleware(rl, false, handleIndicators))
	mux.HandleFunc("/forecast", withMiddleware(rl, false, handleForecast))
	mux.HandleFunc("/drivers", withMiddleware(rl, false, handleDrivers))
	mux.HandleFunc("/predict", withMiddleware(rl, false, handlePredict))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	log.Printf("SIAGA API listening on :%s (data dir: %s)", port, dataDir())
	log.Fatal(http.ListenAndServe(":"+port, mux))
}
