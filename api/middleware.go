// Auth and rate limiting for the SIAGA API. Standard library only.
//
// Configuration (environment variables):
//
//	SIAGA_API_KEYS   comma-separated list of valid API keys. If empty, auth is
//	                 DISABLED (a warning is logged) so local development works.
//	SIAGA_RATE_RPM   allowed requests per minute per client IP (default 60).
//
// /health is always exempt so liveness probes never need a key or a token.
package main

import (
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ---- API key auth --------------------------------------------------------------

var apiKeys map[string]bool

func loadAPIKeys() {
	apiKeys = map[string]bool{}
	for _, k := range strings.Split(os.Getenv("SIAGA_API_KEYS"), ",") {
		if k = strings.TrimSpace(k); k != "" {
			apiKeys[k] = true
		}
	}
	if len(apiKeys) == 0 {
		log.Println("WARNING: SIAGA_API_KEYS is empty, API-key auth is DISABLED (development mode)")
	} else {
		log.Printf("API-key auth enabled (%d key(s))", len(apiKeys))
	}
}

// presentedKey reads the key from "X-API-Key" or "Authorization: Bearer <key>".
func presentedKey(r *http.Request) string {
	if k := r.Header.Get("X-API-Key"); k != "" {
		return k
	}
	if a := r.Header.Get("Authorization"); strings.HasPrefix(a, "Bearer ") {
		return strings.TrimPrefix(a, "Bearer ")
	}
	return ""
}

func authOK(r *http.Request) bool {
	if len(apiKeys) == 0 {
		return true // auth disabled
	}
	return apiKeys[presentedKey(r)]
}

// ---- token-bucket rate limiter (per client IP) ---------------------------------

type bucket struct {
	tokens float64
	last   time.Time
}

type rateLimiter struct {
	mu       sync.Mutex
	buckets  map[string]*bucket
	rps      float64 // refill rate, tokens per second
	burst    float64 // bucket capacity
}

func newRateLimiter(rpm int) *rateLimiter {
	rl := &rateLimiter{buckets: map[string]*bucket{}, rps: float64(rpm) / 60.0, burst: float64(rpm)}
	go rl.reap()
	return rl
}

// allow consumes one token for ip, refilling by elapsed time. Returns false when
// the bucket is empty (client is over the limit).
func (rl *rateLimiter) allow(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	now := time.Now()
	b := rl.buckets[ip]
	if b == nil {
		b = &bucket{tokens: rl.burst, last: now}
		rl.buckets[ip] = b
	}
	b.tokens += now.Sub(b.last).Seconds() * rl.rps
	if b.tokens > rl.burst {
		b.tokens = rl.burst
	}
	b.last = now
	if b.tokens < 1 {
		return false
	}
	b.tokens--
	return true
}

// reap drops idle buckets so the map cannot grow without bound.
func (rl *rateLimiter) reap() {
	for range time.Tick(5 * time.Minute) {
		rl.mu.Lock()
		for ip, b := range rl.buckets {
			if time.Since(b.last) > 10*time.Minute {
				delete(rl.buckets, ip)
			}
		}
		rl.mu.Unlock()
	}
}

func clientIP(r *http.Request) string {
	if fwd := r.Header.Get("X-Forwarded-For"); fwd != "" {
		return strings.TrimSpace(strings.Split(fwd, ",")[0])
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// ---- middleware chain ----------------------------------------------------------

// withMiddleware wraps a handler with CORS, auth, and rate limiting. /health is
// exempt from auth and limiting so orchestrators can always probe liveness.
func withMiddleware(rl *rateLimiter, exempt bool, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, X-API-Key, Authorization")
		if r.Method == http.MethodOptions {
			return
		}
		if !exempt {
			if !authOK(r) {
				writeErr(w, http.StatusUnauthorized, "missing or invalid API key")
				return
			}
			if !rl.allow(clientIP(r)) {
				w.Header().Set("Retry-After", "60")
				writeErr(w, http.StatusTooManyRequests, "rate limit exceeded, retry in 60s")
				return
			}
		}
		h(w, r)
	}
}

func writeErr(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write([]byte(`{"error":"` + msg + `"}`))
}

func rateFromEnv() int {
	if v := os.Getenv("SIAGA_RATE_RPM"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return 60
}
