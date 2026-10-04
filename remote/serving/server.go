// Package serving owns the dormant rollback facade's HTTP boundary.
// Audio, inference and action authority remain outside this package.
package serving

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"
)

const (
	MaxChatBytes  = 16 * 1024
	MaxReplyBytes = 64 * 1024
	ChatDeadline  = 30 * time.Second
	maxPeers      = 4096
)

var ErrBackendBusy = errors.New("backend busy")

type Backend interface {
	Generate(context.Context, string) (string, error)
}
type BackendFunc func(context.Context, string) (string, error)

func (f BackendFunc) Generate(ctx context.Context, s string) (string, error) { return f(ctx, s) }

type Config struct {
	RemoteToken                   string
	AllowNoAuth                   bool
	LiveKitURL, APIKey, APISecret string
	RollbackVoice                 bool
	WebDir                        string
	ChatBackend                   Backend
	Now                           func() time.Time
}

type Server struct {
	cfg     Config
	limiter limiter
	active  chan struct{}
	web     *os.Root
}

func New(cfg Config) (*Server, error) {
	cfg.RemoteToken = trimPythonSpace(cfg.RemoteToken)
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	s := &Server{cfg: cfg, active: make(chan struct{}, 1)}
	s.limiter.hits = make(map[string][]time.Time)
	if cfg.WebDir != "" {
		root, err := os.OpenRoot(cfg.WebDir)
		if err != nil {
			return nil, errors.New("web directory unavailable")
		}
		s.web = root
	}
	return s, nil
}
func (s *Server) Close() error {
	if s.web != nil {
		return s.web.Close()
	}
	return nil
}

func SanitizeRoomName(name string) string {
	var b strings.Builder
	bad := false
	// Python lower expands capital dotted I before the ASCII room filter.
	lowered := strings.ToLower(strings.ReplaceAll(trimPythonSpace(name), "İ", "i\u0307"))
	for _, r := range lowered {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '_' || r == '-' {
			if bad {
				b.WriteByte('-')
				bad = false
			}
			b.WriteRune(r)
		} else {
			bad = true
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "assistant"
	}
	return out
}
func SanitizeIdentity(name string) string {
	var b strings.Builder
	space := false
	for _, r := range trimPythonSpace(name) {
		if pythonSpace(r) {
			space = true
			continue
		}
		if space {
			b.WriteByte('-')
			space = false
		}
		if r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '_' || r == '-' {
			b.WriteRune(r)
		}
	}
	out := b.String()
	if out == "" {
		return "user"
	}
	return out
}

// MintAccessToken implements the reviewed LiveKit API 1.2.0 HS256 claims.
// TTL is exact and bounded; an invalid TTL publishes no JWT.
func MintAccessToken(key, secret, identity, room string, ttl time.Duration, now time.Time) (string, error) {
	if key == "" || secret == "" || identity == "" || room == "" || ttl <= 0 || ttl > time.Hour || ttl%time.Second != 0 {
		return "", errors.New("token configuration unavailable")
	}
	claims := struct {
		Iss   string         `json:"iss"`
		Sub   string         `json:"sub"`
		Name  string         `json:"name"`
		Nbf   int64          `json:"nbf"`
		Exp   int64          `json:"exp"`
		Video map[string]any `json:"video"`
	}{key, identity, identity, now.Unix(), now.Unix() + int64(ttl/time.Second), map[string]any{
		"roomJoin": true, "room": room, "canPublish": true, "canSubscribe": true, "canPublishData": true,
	}}
	raw, err := json.Marshal(claims)
	if err != nil {
		return "", errors.New("token mint failed")
	}
	enc := base64.RawURLEncoding
	body := enc.EncodeToString([]byte(`{"alg":"HS256","typ":"JWT"}`)) + "." + enc.EncodeToString(raw)
	mac := hmac.New(sha256.New, []byte(secret))
	_, _ = mac.Write([]byte(body))
	return body + "." + enc.EncodeToString(mac.Sum(nil)), nil
}

func loopbackLiveKit(raw string) bool {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.User != nil || (u.Scheme != "ws" && u.Scheme != "wss") || u.Hostname() == "" || u.Fragment != "" {
		return false
	}
	h := strings.TrimSuffix(strings.ToLower(u.Hostname()), ".")
	if h == "localhost" {
		return true
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}

func (s *Server) auth(w http.ResponseWriter, r *http.Request) bool {
	if s.cfg.RemoteToken == "" {
		if s.cfg.AllowNoAuth {
			return true
		}
		detail(w, 401, "remote auth not configured (set SPEAKER_REMOTE_TOKEN)")
		return false
	}
	headers := r.Header.Values("Authorization")
	if len(headers) == 1 {
		header := trimPythonSpace(headers[0])
		split := strings.IndexFunc(header, pythonSpace)
		if split > 0 && strings.EqualFold(header[:split], "bearer") {
			// Hash equalizes lengths before constant-time comparison.
			got, want := sha256.Sum256([]byte(trimPythonSpace(header[split:]))), sha256.Sum256([]byte(s.cfg.RemoteToken))
			if subtle.ConstantTimeCompare(got[:], want[:]) == 1 {
				return true
			}
		}
	}
	w.Header().Set("WWW-Authenticate", "Bearer")
	detail(w, 401, "missing or invalid bearer token")
	return false
}

func jsonResponse(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(value)
}
func detail(w http.ResponseWriter, status int, message string) {
	jsonResponse(w, status, map[string]string{"detail": message})
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Referrer-Policy", "no-referrer")
	w.Header().Set("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self' https: wss:; media-src 'self' blob: mediastream:; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
	switch r.URL.Path {
	case "/healthz":
		if r.Method != "GET" && r.Method != "HEAD" {
			method(w, "GET, HEAD")
			return
		}
		jsonResponse(w, 200, map[string]bool{"ok": true})
	case "/token":
		if r.Method != "GET" {
			method(w, "GET")
			return
		}
		if !s.auth(w, r) {
			return
		}
		// Rollback is machine-local only. This facade cannot open the still-unmet
		// trusted-LAN live-A/B gate, even when a setup grant exists.
		if !s.cfg.RollbackVoice || !loopbackLiveKit(s.cfg.LiveKitURL) {
			detail(w, 403, "voice transport unavailable (rollback is loopback only)")
			return
		}
		identity, room := SanitizeIdentity(r.URL.Query().Get("identity")), SanitizeRoomName(r.URL.Query().Get("room"))
		if len(identity) > 128 || len(room) > 128 {
			detail(w, 400, "identity or room too long")
			return
		}
		token, err := MintAccessToken(s.cfg.APIKey, s.cfg.APISecret, identity, room, time.Hour, s.cfg.Now())
		if err != nil {
			detail(w, 500, "failed to mint access token")
			return
		}
		jsonResponse(w, 200, map[string]string{"token": token, "url": s.cfg.LiveKitURL, "room": room, "identity": identity})
	case "/chat":
		if r.Method != "POST" {
			method(w, "POST")
			return
		}
		if !s.auth(w, r) {
			return
		}
		s.chat(w, r)
	default:
		s.static(w, r)
	}
}
func method(w http.ResponseWriter, allowed string) {
	w.Header().Set("Allow", allowed)
	detail(w, 405, "method not allowed")
}

// Parse one JSON value. Legacy non-object/empty/null messages remain empty;
// non-string and duplicate object members fail closed instead of panicking.
func messageBody(raw []byte) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	if !utf8.Valid(raw) {
		return "", errors.New("invalid JSON body")
	}
	d := json.NewDecoder(strings.NewReader(string(raw)))
	var payload any
	if err := d.Decode(&payload); err != nil {
		return "", err
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return "", errors.New("trailing data")
	}
	object, ok := payload.(map[string]any)
	if !ok {
		return "", nil
	}
	// A second token pass detects duplicates before any inference admission.
	d = json.NewDecoder(strings.NewReader(string(raw)))
	_, _ = d.Token()
	keys := make(map[string]bool)
	for d.More() {
		k, err := d.Token()
		if err != nil {
			return "", err
		}
		key := k.(string)
		if keys[key] {
			return "", errors.New("duplicate member")
		}
		keys[key] = true
		if err := d.Decode(new(any)); err != nil {
			return "", err
		}
	}
	m := object["message"]
	if m == nil {
		return "", nil
	}
	text, ok := m.(string)
	if !ok {
		return "", errors.New("message must be string")
	}
	return trimPythonSpace(text), nil
}

func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	peer, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		peer = r.RemoteAddr
	}
	if peer == "" {
		peer = "unknown"
	}
	if !s.limiter.allow(peer, s.cfg.Now()) {
		detail(w, 429, "rate limit exceeded")
		return
	}
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, MaxChatBytes))
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			detail(w, 413, "request body too large")
		} else {
			detail(w, 400, "invalid JSON body")
		}
		return
	}
	text, err := messageBody(raw)
	if err != nil {
		detail(w, 400, "invalid JSON body")
		return
	}
	if text == "" {
		jsonResponse(w, 200, map[string]string{"reply": ""})
		return
	}
	if s.cfg.ChatBackend == nil {
		detail(w, 503, "chat backend unavailable")
		return
	}
	select {
	case s.active <- struct{}{}:
	default:
		detail(w, 503, "chat backend busy")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), ChatDeadline)
	defer cancel()
	type result struct {
		reply string
		err   error
	}
	done := make(chan result, 1)
	go func() {
		var out result
		completed := false
		defer func() {
			_ = recover()
			if !completed {
				out = result{err: errors.New("backend failed")}
			}
			<-s.active // A timed-out uncooperative source retains BUSY until it returns.
			done <- out
		}()
		out.reply, out.err = s.cfg.ChatBackend.Generate(ctx, text)
		completed = true
	}()
	select {
	case <-ctx.Done():
		detail(w, 504, "chat backend timeout")
	case out := <-done:
		if ctx.Err() != nil {
			detail(w, 504, "chat backend timeout")
			return
		}
		if errors.Is(out.err, ErrBackendBusy) {
			detail(w, 503, "chat backend busy")
			return
		}
		if out.err != nil || !utf8.ValidString(out.reply) || len(out.reply) > MaxReplyBytes {
			detail(w, 500, "chat backend error")
			return
		}
		var body bytes.Buffer
		enc := json.NewEncoder(&body)
		enc.SetEscapeHTML(false)
		if enc.Encode(map[string]string{"reply": trimPythonSpace(out.reply)}) != nil || body.Len() > MaxReplyBytes {
			detail(w, 500, "chat backend error")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		w.WriteHeader(200)
		_, _ = w.Write(body.Bytes())
	}
}

func (s *Server) static(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" && r.Method != "HEAD" {
		method(w, "GET, HEAD")
		return
	}
	var filename string
	switch r.URL.Path {
	case "/", "/index.html":
		filename = "index.html"
	case "/app.js":
		filename = "app.js"
	case "/livekit-client.umd.min.js":
		filename = "livekit-client.umd.min.js"
	default:
		http.NotFound(w, r)
		return
	}
	if s.web == nil {
		http.NotFound(w, r)
		return
	}
	info, err := s.web.Lstat(filename)
	if err != nil || !info.Mode().IsRegular() {
		http.NotFound(w, r)
		return
	}
	f, err := s.web.Open(filename)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer f.Close()
	openedInfo, err := f.Stat()
	if err != nil || !openedInfo.Mode().IsRegular() || !os.SameFile(info, openedInfo) {
		http.NotFound(w, r)
		return
	}
	http.ServeContent(w, r, filename, openedInfo.ModTime(), f)
}

type limiter struct {
	mu   sync.Mutex
	hits map[string][]time.Time
}

func (l *limiter) allow(key string, now time.Time) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	cutoff := now.Add(-time.Minute)
	trim := func(hits []time.Time) []time.Time {
		n := 0
		for n < len(hits) && !hits[n].After(cutoff) {
			n++
		}
		return hits[n:]
	}
	hits, exists := l.hits[key]
	hits = trim(hits)
	if !exists && len(l.hits) >= maxPeers {
		for k, h := range l.hits {
			if h = trim(h); len(h) == 0 {
				delete(l.hits, k)
			} else {
				l.hits[k] = h
			}
		}
		if len(l.hits) >= maxPeers {
			return false
		}
	}
	if len(hits) >= 30 {
		l.hits[key] = hits
		return false
	}
	// Copy trims retained backing capacity as well as retained event count.
	next := make([]time.Time, len(hits)+1)
	copy(next, hits)
	next[len(hits)] = now
	l.hits[key] = next
	return true
}

// Preserve Python str.strip/re \s compatibility for the four ASCII information
// separators, as well as Unicode whitespace, at the migrated string boundary.
func pythonSpace(r rune) bool         { return unicode.IsSpace(r) || r >= 0x1c && r <= 0x1f }
func trimPythonSpace(s string) string { return strings.TrimFunc(s, pythonSpace) }
