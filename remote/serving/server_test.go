package serving

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// These fixture rows preserve the original Python sanitizer tests. HTTP, JWT,
// concurrency and bounds below test observable contracts rather than helpers.
func TestSanitizeRoomNameFromPythonCases(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"My Room!", "my-room"}, {"  Assistant Room 1 ", "assistant-room-1"},
		{"", "assistant"}, {"***", "assistant"}, {"a__b--c", "a__b--c"},
	} {
		if got := SanitizeRoomName(tc.in); got != tc.want {
			t.Errorf("SanitizeRoomName(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestSanitizeIdentityFromPythonCases(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"Alice Smith", "Alice-Smith"}, {"", "user"}, {"  bob  ", "bob"},
		{"a/b\\c!", "abc"}, {"Alice\t \nSmith", "Alice-Smith"}, {"***", "user"},
	} {
		if got := SanitizeIdentity(tc.in); got != tc.want {
			t.Errorf("SanitizeIdentity(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func newContractServer(t *testing.T, cfg Config) *Server {
	t.Helper()
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return s
}

func serveContract(s http.Handler, method, path, body, auth string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	r.RemoteAddr = "192.0.2.10:12345"
	if auth != "" {
		r.Header.Set("Authorization", auth)
	}
	w := httptest.NewRecorder()
	s.ServeHTTP(w, r)
	return w
}

func assertJSON(t *testing.T, w *httptest.ResponseRecorder, status int, want any) {
	t.Helper()
	if w.Code != status {
		t.Fatalf("HTTP %d, want %d; body %q", w.Code, status, w.Body.String())
	}
	if !strings.HasPrefix(w.Header().Get("Content-Type"), "application/json") {
		t.Errorf("JSON response Content-Type = %q", w.Header().Get("Content-Type"))
	}
	got := map[string]any{}
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("invalid JSON response: %v", err)
	}
	encoded, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	expected := map[string]any{}
	if err := json.Unmarshal(encoded, &expected); err != nil {
		t.Fatal(err)
	}
	actualJSON, _ := json.Marshal(got)
	expectedJSON, _ := json.Marshal(expected)
	if !bytes.Equal(actualJSON, expectedJSON) {
		t.Errorf("JSON = %s, want %s", actualJSON, expectedJSON)
	}
}

func assertDetail(t *testing.T, w *httptest.ResponseRecorder, status int, detail string) {
	t.Helper()
	assertJSON(t, w, status, map[string]any{"detail": detail})
}

func TestHealthIsPublicAndDoesNotConstructBackend(t *testing.T) {
	var calls atomic.Int32
	s := newContractServer(t, Config{ChatBackend: BackendFunc(func(context.Context, string) (string, error) {
		calls.Add(1)
		return "", errors.New("should not be called")
	})})
	assertJSON(t, serveContract(s, "GET", "/healthz", "", ""), 200, map[string]any{"ok": true})
	if calls.Load() != 0 {
		t.Fatal("health check invoked inference")
	}
}

func TestAuthDenyByDefaultAndWhitespaceTokenIsUnset(t *testing.T) {
	for _, token := range []string{"", " \t\n "} {
		s := newContractServer(t, Config{RemoteToken: token})
		for _, route := range []struct{ method, path string }{{"GET", "/token"}, {"POST", "/chat"}} {
			w := serveContract(s, route.method, route.path, `{ "message": "hi" }`, "Bearer supplied")
			assertDetail(t, w, 401, "remote auth not configured (set SPEAKER_REMOTE_TOKEN)")
		}
	}
}

func TestConfiguredAuthCannotBeBypassedByNoAuthOptIn(t *testing.T) {
	s := newContractServer(t, Config{RemoteToken: " contract-token ", AllowNoAuth: true})
	for _, auth := range []string{"", "Basic contract-token", "Bearer", "Bearer wrong", "Bearer contract-token wrong"} {
		w := serveContract(s, "POST", "/chat", "{}", auth)
		assertDetail(t, w, 401, "missing or invalid bearer token")
		if got := w.Header().Get("WWW-Authenticate"); got != "Bearer" {
			t.Errorf("WWW-Authenticate = %q", got)
		}
	}
	for _, auth := range []string{"Bearer contract-token", "bEaReR\t contract-token  "} {
		assertJSON(t, serveContract(s, "POST", "/chat", "{}", auth), 200, map[string]any{"reply": ""})
	}
}

func TestExplicitNoAuthOptInOnlyAppliesWhenTokenUnset(t *testing.T) {
	s := newContractServer(t, Config{AllowNoAuth: true})
	assertJSON(t, serveContract(s, "POST", "/chat", "{}", ""), 200, map[string]any{"reply": ""})
	// Development auth opt-in cannot activate the dormant voice facade.
	if w := serveContract(s, "GET", "/token", "", ""); w.Code != 403 {
		t.Fatalf("default dormant /token = %d, want 403", w.Code)
	}
}

func decodeAndVerifyJWT(t *testing.T, token, key, secret, identity, room string, ttl time.Duration, now time.Time) {
	t.Helper()
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		t.Fatalf("JWT segment count = %d", len(parts))
	}
	decode := func(part string) map[string]any {
		raw, err := base64.RawURLEncoding.DecodeString(part)
		if err != nil {
			t.Fatalf("invalid JWT base64: %v", err)
		}
		value := map[string]any{}
		if err := json.Unmarshal(raw, &value); err != nil {
			t.Fatalf("invalid JWT JSON: %v", err)
		}
		return value
	}
	header := decode(parts[0])
	if header["alg"] != "HS256" || header["typ"] != "JWT" {
		t.Errorf("JWT header = %#v", header)
	}
	mac := hmac.New(sha256.New, []byte(secret))
	_, _ = mac.Write([]byte(parts[0] + "." + parts[1]))
	signature, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || !hmac.Equal(signature, mac.Sum(nil)) {
		t.Fatal("JWT signature does not verify against configured synthetic key")
	}
	claims := decode(parts[1])
	for name, want := range map[string]string{"iss": key, "sub": identity, "name": identity} {
		if claims[name] != want {
			t.Errorf("claim %s = %v, want %q", name, claims[name], want)
		}
	}
	nbf, ok := claims["nbf"].(float64)
	if !ok || nbf != float64(now.Unix()) {
		t.Errorf("nbf = %v, want %d", claims["nbf"], now.Unix())
	}
	exp, ok := claims["exp"].(float64)
	if !ok || exp-nbf != ttl.Seconds() {
		t.Errorf("exp-nbf = %v, want %v", exp-nbf, ttl.Seconds())
	}
	video, ok := claims["video"].(map[string]any)
	if !ok {
		t.Fatal("missing video grants")
	}
	if video["room"] != room {
		t.Errorf("room = %v, want %q", video["room"], room)
	}
	for _, name := range []string{"roomJoin", "canPublish", "canSubscribe", "canPublishData"} {
		if video[name] != true {
			t.Errorf("video grant %s = %v, want true", name, video[name])
		}
	}
}

func TestMintAccessTokenObservableSDKContractAndExactTTL(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	for _, ttl := range []time.Duration{time.Hour, 90 * time.Second} {
		token, err := MintAccessToken("synthetic-key", "synthetic-secret", "publisher", "room", ttl, now)
		if err != nil {
			t.Fatal(err)
		}
		decodeAndVerifyJWT(t, token, "synthetic-key", "synthetic-secret", "publisher", "room", ttl, now)
	}
}

func TestMintAccessTokenConfigurationAndTTLFailuresPublishNoToken(t *testing.T) {
	for _, tc := range []struct {
		key, secret string
		ttl         time.Duration
	}{{"", "synthetic-secret", time.Hour}, {"synthetic-key", "", time.Hour},
		{"synthetic-key", "synthetic-secret", 0}, {"synthetic-key", "synthetic-secret", -time.Second},
		{"synthetic-key", "synthetic-secret", time.Hour + time.Second}, {"synthetic-key", "synthetic-secret", time.Second + time.Nanosecond}} {
		token, err := MintAccessToken(tc.key, tc.secret, "publisher", "room", tc.ttl, time.Unix(1_800_000_000, 0))
		if err == nil || token != "" {
			t.Errorf("invalid mint configuration published token or lacked error")
		}
	}
}

func TestTokenRollbackHTTPContractUsesGoJWTAndSanitizedNames(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	s := newContractServer(t, Config{RemoteToken: "contract-token", RollbackVoice: true,
		LiveKitURL: "ws://127.0.0.1:7880", APIKey: "synthetic-key", APISecret: "synthetic-secret",
		Now: func() time.Time { return now }})
	for _, tc := range []struct{ query, identity, room string }{
		{"", "user", "assistant"}, {"?identity=Alice%20Smith&room=My%20Room%21", "Alice-Smith", "my-room"},
	} {
		w := serveContract(s, "GET", "/token"+tc.query, "", "Bearer contract-token")
		if w.Code != 200 {
			t.Fatalf("/token = %d: %s", w.Code, w.Body.String())
		}
		payload := map[string]string{}
		if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
			t.Fatal(err)
		}
		if len(payload) != 4 || payload["url"] != "ws://127.0.0.1:7880" || payload["room"] != tc.room || payload["identity"] != tc.identity {
			t.Fatalf("token response shape/binding = %#v", payload)
		}
		decodeAndVerifyJWT(t, payload["token"], "synthetic-key", "synthetic-secret", tc.identity, tc.room, time.Hour, now)
	}
}

func TestTokenErrorDetailFromOriginalPythonContractIsGeneric(t *testing.T) {
	s := newContractServer(t, Config{AllowNoAuth: true, RollbackVoice: true, LiveKitURL: "ws://127.0.0.1:7880"})
	assertDetail(t, serveContract(s, "GET", "/token", "", ""), 500, "failed to mint access token")
}

func TestChatEmptyAndNonobjectPayloadsDoNotCallBackend(t *testing.T) {
	var calls atomic.Int32
	s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(context.Context, string) (string, error) {
		calls.Add(1)
		return "should not be called", nil
	})})
	for _, body := range []string{"", "{}", "null", "[]", "42", "true", `{"other":"value"}`, `{"message":null}`, `{"message":" \t\n "}`} {
		assertJSON(t, serveContract(s, "POST", "/chat", body, ""), 200, map[string]any{"reply": ""})
	}
	if calls.Load() != 0 {
		t.Fatalf("empty inputs called backend %d times", calls.Load())
	}
}

func TestChatTrimsTextAndReplyAndAcceptsUnrelatedObjectFields(t *testing.T) {
	s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(_ context.Context, message string) (string, error) {
		if message != "hello" {
			t.Errorf("backend message = %q, want hello", message)
		}
		return " \tanswer\n", nil
	})})
	assertJSON(t, serveContract(s, "POST", "/chat", `{"message":"  hello  ","other":"compatible"}`, ""), 200, map[string]any{"reply": "answer"})
}

func TestChatRejectsInvalidJSONAndNonStringMessagesBeforeInference(t *testing.T) {
	var calls atomic.Int32
	s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(context.Context, string) (string, error) {
		calls.Add(1)
		return "", nil
	})})
	for _, body := range []string{"{", `{"message":"hi"} trailing`, `{"message":`, `{"message":"\xff"}`} {
		if w := serveContract(s, "POST", "/chat", body, ""); w.Code != 400 {
			t.Errorf("invalid body %q got HTTP %d", body, w.Code)
		}
	}
	for _, body := range []string{`{"message":1}`, `{"message":true}`, `{"message":[]}`, `{"message":{}}`} {
		if w := serveContract(s, "POST", "/chat", body, ""); w.Code != 400 {
			t.Errorf("non-string message %s got HTTP %d", body, w.Code)
		}
	}
	if calls.Load() != 0 {
		t.Fatalf("rejected input invoked backend %d times", calls.Load())
	}
}

func TestChatBodyByteLimitIncludesStreamingBodiesAndExactBoundary(t *testing.T) {
	s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(context.Context, string) (string, error) { return "ok", nil })})
	prefix, suffix := `{"message":"`, `"}`
	atLimit := prefix + strings.Repeat("x", MaxChatBytes-len(prefix)-len(suffix)) + suffix
	assertJSON(t, serveContract(s, "POST", "/chat", atLimit, ""), 200, map[string]any{"reply": "ok"})
	for _, declared := range []int64{-1, int64(MaxChatBytes + 1)} {
		r := httptest.NewRequest("POST", "/chat", io.NopCloser(strings.NewReader(atLimit+" ")))
		r.ContentLength = declared
		r.RemoteAddr = "192.0.2.10:12345"
		w := httptest.NewRecorder()
		s.ServeHTTP(w, r)
		assertDetail(t, w, 413, "request body too large")
	}
}

func TestChatErrorDetailFromOriginalPythonContractIsGeneric(t *testing.T) {
	s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(context.Context, string) (string, error) {
		return "partial-private-output", errors.New("private backend at http://10.0.0.5:11434 refused /private/model.gguf")
	})})
	assertDetail(t, serveContract(s, "POST", "/chat", `{"message":"hello"}`, ""), 500, "chat backend error")
}

func TestChatOutputIsBoundedAndInvalidUTF8FailsClosed(t *testing.T) {
	for _, reply := range []string{strings.Repeat("x", MaxReplyBytes+1), string([]byte{0xff, 0xfe})} {
		s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(context.Context, string) (string, error) { return reply, nil })})
		w := serveContract(s, "POST", "/chat", `{"message":"hello"}`, "")
		assertDetail(t, w, 500, "chat backend error")
		if strings.Contains(w.Body.String(), strings.Repeat("x", 256)) {
			t.Error("oversized output escaped")
		}
	}
}

func TestChatRateLimitCountsAuthorizedMalformedAndEmptyRequests(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	s := newContractServer(t, Config{RemoteToken: "contract-token", Now: func() time.Time { return now }})
	for i := 0; i < 40; i++ {
		if w := serveContract(s, "POST", "/chat", "{}", "Bearer wrong"); w.Code != 401 {
			t.Fatalf("unauthorized request %d = %d", i, w.Code)
		}
	}
	for i := 0; i < 30; i++ {
		body, want := "{}", 200
		if i%2 == 0 {
			body, want = "{", 400
		}
		if w := serveContract(s, "POST", "/chat", body, "Bearer contract-token"); w.Code != want {
			t.Fatalf("admitted request %d = %d, want %d", i, w.Code, want)
		}
	}
	assertDetail(t, serveContract(s, "POST", "/chat", "{}", "Bearer contract-token"), 429, "rate limit exceeded")
	now = now.Add(time.Minute)
	assertJSON(t, serveContract(s, "POST", "/chat", "{}", "Bearer contract-token"), 200, map[string]any{"reply": ""})
}

func TestChatRateLimitUsesPeerHostAndIgnoresForwardedHeaders(t *testing.T) {
	s := newContractServer(t, Config{AllowNoAuth: true})
	for i := 0; i < 31; i++ {
		r := httptest.NewRequest("POST", "/chat", strings.NewReader("{}"))
		r.RemoteAddr = fmt.Sprintf("192.0.2.10:%d", 10000+i)
		r.Header.Set("X-Forwarded-For", fmt.Sprintf("198.51.100.%d", i+1))
		w := httptest.NewRecorder()
		s.ServeHTTP(w, r)
		want := 200
		if i == 30 {
			want = 429
		}
		if w.Code != want {
			t.Fatalf("request %d = %d, want %d", i, w.Code, want)
		}
	}
	r := httptest.NewRequest("POST", "/chat", strings.NewReader("{}"))
	r.RemoteAddr = "192.0.2.11:10000"
	w := httptest.NewRecorder()
	s.ServeHTTP(w, r)
	assertJSON(t, w, 200, map[string]any{"reply": ""})
}

func TestChatConcurrentLimiterAdmitsExactlyThirty(t *testing.T) {
	s := newContractServer(t, Config{AllowNoAuth: true})
	var accepted, rejected atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < 60; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w := serveContract(s, "POST", "/chat", "{}", "")
			switch w.Code {
			case 200:
				accepted.Add(1)
			case 429:
				rejected.Add(1)
			default:
				t.Errorf("concurrent request got HTTP %d", w.Code)
			}
		}()
	}
	wg.Wait()
	if accepted.Load() != 30 || rejected.Load() != 30 {
		t.Fatalf("accepted/rejected = %d/%d, want 30/30", accepted.Load(), rejected.Load())
	}
}

func TestChatBackendAdmissionIsNoWaitBusyAndReopensAfterTrueReturn(t *testing.T) {
	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	var calls atomic.Int32
	s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(context.Context, string) (string, error) {
		if calls.Add(1) == 1 {
			entered <- struct{}{}
			<-release
		}
		return "ok", nil
	})})
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- serveContract(s, "POST", "/chat", `{"message":"first"}`, "") }()
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("backend was not entered")
	}
	busyDone := make(chan *httptest.ResponseRecorder, 1)
	go func() { busyDone <- serveContract(s, "POST", "/chat", `{"message":"second"}`, "") }()
	select {
	case w := <-busyDone:
		if w.Code != 503 {
			t.Errorf("busy response = %d, want 503", w.Code)
		}
	case <-time.After(time.Second):
		close(release)
		t.Fatal("busy request waited behind active inference")
	}
	if calls.Load() != 1 {
		t.Errorf("busy request entered backend: calls=%d", calls.Load())
	}
	close(release)
	select {
	case w := <-done:
		assertJSON(t, w, 200, map[string]any{"reply": "ok"})
	case <-time.After(2 * time.Second):
		t.Fatal("released backend did not settle")
	}
	assertJSON(t, serveContract(s, "POST", "/chat", `{"message":"third"}`, ""), 200, map[string]any{"reply": "ok"})
}

func TestChatPassesBoundedDeadlineToBackend(t *testing.T) {
	s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(ctx context.Context, _ string) (string, error) {
		deadline, ok := ctx.Deadline()
		if !ok {
			t.Error("backend context has no deadline")
		} else if remaining := time.Until(deadline); remaining <= 0 || remaining > 30*time.Second {
			t.Errorf("backend deadline remaining = %v, want (0,30s]", remaining)
		}
		return "ok", nil
	})})
	assertJSON(t, serveContract(s, "POST", "/chat", `{"message":"hello"}`, ""), 200, map[string]any{"reply": "ok"})
}

func TestMethodContractsDoNotFallThroughToStatic(t *testing.T) {
	s := newContractServer(t, Config{AllowNoAuth: true})
	for _, tc := range []struct{ method, path string }{{"POST", "/healthz"}, {"POST", "/token"}, {"GET", "/chat"}, {"PUT", "/chat"}} {
		if w := serveContract(s, tc.method, tc.path, "{}", ""); w.Code != 405 {
			t.Errorf("%s %s = %d, want 405", tc.method, tc.path, w.Code)
		}
	}
}

func actualWebFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for _, name := range []string{"index.html", "app.js", "livekit-client.umd.min.js"} {
		data, err := os.ReadFile(filepath.Join("..", "..", "web", name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestStaticServesShippedAssetsAndPreservesCSPAndSDKIntegrity(t *testing.T) {
	dir := actualWebFixture(t)
	s := newContractServer(t, Config{WebDir: dir})
	for _, name := range []string{"/", "/index.html", "/app.js", "/livekit-client.umd.min.js"} {
		w := serveContract(s, "GET", name, "", "")
		if w.Code != 200 {
			t.Fatalf("static %s = %d", name, w.Code)
		}
		fileName := strings.TrimPrefix(name, "/")
		if fileName == "" {
			fileName = "index.html"
		}
		want, _ := os.ReadFile(filepath.Join(dir, fileName))
		if !bytes.Equal(w.Body.Bytes(), want) {
			t.Errorf("served %s differs from shipped fixture", name)
		}
		if w.Header().Get("X-Content-Type-Options") != "nosniff" {
			t.Errorf("%s missing nosniff", name)
		}
	}
	html, _ := os.ReadFile(filepath.Join(dir, "index.html"))
	sdk, _ := os.ReadFile(filepath.Join(dir, "livekit-client.umd.min.js"))
	digest := sha512.Sum384(sdk)
	integrity := "sha384-" + base64.StdEncoding.EncodeToString(digest[:])
	if !strings.Contains(string(html), `integrity="`+integrity+`"`) {
		t.Fatal("shipped HTML SRI does not match actual vendored SDK")
	}
	w := serveContract(s, "GET", "/", "", "")
	csp := w.Header().Get("Content-Security-Policy")
	for _, required := range []string{"default-src 'none'", "script-src 'self'", "base-uri 'none'", "frame-ancestors 'none'"} {
		if !strings.Contains(csp, required) {
			t.Errorf("HTTP CSP lacks %q: %q", required, csp)
		}
	}
}

func TestStaticAllowlistPreventsConfigTraversalAndDirectoryListing(t *testing.T) {
	dir := actualWebFixture(t)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte("private-static-sentinel"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, "nested"), 0700); err != nil {
		t.Fatal(err)
	}
	s := newContractServer(t, Config{WebDir: dir})
	for _, path := range []string{"/config.json", "/.env", "/nested/", "/missing", "/../config.json", "/%2e%2e/config.json", "/chat/missing", "/healthz/missing", "/token/missing"} {
		w := serveContract(s, "GET", path, "", "")
		if w.Code == 200 || strings.Contains(w.Body.String(), "private-static-sentinel") {
			t.Errorf("forbidden static path %s escaped with HTTP %d", path, w.Code)
		}
	}
	assertJSON(t, serveContract(s, "GET", "/healthz", "", ""), 200, map[string]any{"ok": true})
	assertDetail(t, serveContract(s, "POST", "/chat", "{}", ""), 401, "remote auth not configured (set SPEAKER_REMOTE_TOKEN)")
}

func TestStaticRejectsAllowedFilenameSymlinkOutsideWebRoot(t *testing.T) {
	dir := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside.js")
	if err := os.WriteFile(outside, []byte("private-outside-sentinel"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "app.js")); err != nil {
		t.Fatal(err)
	}
	s := newContractServer(t, Config{WebDir: dir})
	w := serveContract(s, "GET", "/app.js", "", "")
	if w.Code == 200 || strings.Contains(w.Body.String(), "private-outside-sentinel") {
		t.Fatal("allowed filename followed an outside-root symlink")
	}
}

func TestCanceledUncooperativeBackendRetainsBusyUntilActualReturn(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	returned := make(chan struct{})
	var calls atomic.Int32
	s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(_ context.Context, _ string) (string, error) {
		if calls.Add(1) == 1 {
			close(entered)
			<-release // Deliberately ignore cancellation until the source releases.
			close(returned)
		}
		return "ok", nil
	})})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r := httptest.NewRequest("POST", "/chat", strings.NewReader(`{"message":"first"}`)).WithContext(ctx)
	r.RemoteAddr = "192.0.2.10:12345"
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		w := httptest.NewRecorder()
		s.ServeHTTP(w, r)
		done <- w
	}()
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("backend was not entered")
	}
	cancel()
	select {
	case w := <-done:
		assertDetail(t, w, 504, "chat backend timeout")
	case <-time.After(time.Second):
		close(release)
		t.Fatal("canceled HTTP request waited for uncooperative source")
	}
	assertDetail(t, serveContract(s, "POST", "/chat", `{"message":"successor"}`, ""), 503, "chat backend busy")
	if calls.Load() != 1 {
		t.Errorf("cancellation admitted successor before source return: %d calls", calls.Load())
	}
	close(release)
	select {
	case <-returned:
	case <-time.After(time.Second):
		t.Fatal("released source did not return")
	}
	// Source return and the enclosing owner's terminal receipt can differ by a
	// scheduler step. Observe reopening through HTTP without assuming that step.
	for attempt := 0; attempt < 20; attempt++ {
		w := serveContract(s, "POST", "/chat", `{"message":"after-return"}`, "")
		if w.Code == 200 {
			assertJSON(t, w, 200, map[string]any{"reply": "ok"})
			return
		}
		if w.Code != 503 {
			t.Fatalf("post-return admission got HTTP %d", w.Code)
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("backend remained busy after actual return")
}

func TestBackendBusyAndPanicAreBoundedGenericResponses(t *testing.T) {
	for _, tc := range []struct {
		name    string
		backend BackendFunc
		status  int
		detail  string
	}{
		{"busy", func(context.Context, string) (string, error) { return "", ErrBackendBusy }, 503, "chat backend busy"},
		{"panic", func(context.Context, string) (string, error) { panic("private-backend-panic-sentinel") }, 500, "chat backend error"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: tc.backend})
			w := serveContract(s, "POST", "/chat", `{"message":"hello"}`, "")
			assertDetail(t, w, tc.status, tc.detail)
			if strings.Contains(w.Body.String(), "private-backend-panic-sentinel") {
				t.Error("panic detail escaped")
			}
		})
	}
}

func TestTokenLoopbackRollbackDoesNotOpenLANOrCloudAudioGate(t *testing.T) {
	for _, target := range []string{
		"", "wss://192.168.1.20:7880", "ws://192.168.1.20:7880", "wss://voice.home.arpa",
		"wss://example.com", "wss://project.livekit.cloud", "https://127.0.0.1:7880",
		"ws://user:private@127.0.0.1:7880", "ws://127.0.0.1:7880/#fragment",
		"ws://134744072:7880", "ws://0x08080808:7880", "ws://0177.0.0.1:7880",
	} {
		s := newContractServer(t, Config{AllowNoAuth: true, RollbackVoice: true,
			LiveKitURL: target, APIKey: "synthetic-key", APISecret: "synthetic-secret"})
		w := serveContract(s, "GET", "/token", "", "")
		if w.Code != 403 || strings.Contains(w.Body.String(), `"token":`) {
			t.Errorf("disallowed target %q got HTTP %d / token", target, w.Code)
		}
	}
	for _, target := range []string{"ws://127.0.0.1:7880", "wss://localhost:7880", "ws://[::1]:7880"} {
		s := newContractServer(t, Config{AllowNoAuth: true, RollbackVoice: true,
			LiveKitURL: target, APIKey: "synthetic-key", APISecret: "synthetic-secret"})
		if w := serveContract(s, "GET", "/token", "", ""); w.Code != 200 {
			t.Errorf("allowed loopback %q got HTTP %d", target, w.Code)
		}
	}
}

func TestRateLimiterPeerCapacityFailsClosedAndReclaimsExpiredPeers(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	s := newContractServer(t, Config{AllowNoAuth: true, Now: func() time.Time { return now }})
	requestPeer := func(peer int) *httptest.ResponseRecorder {
		r := httptest.NewRequest("POST", "/chat", strings.NewReader("{}"))
		r.RemoteAddr = fmt.Sprintf("198.18.%d.%d:10000", peer/256, peer%256)
		w := httptest.NewRecorder()
		s.ServeHTTP(w, r)
		return w
	}
	for peer := 0; peer < 4096; peer++ {
		if w := requestPeer(peer); w.Code != 200 {
			t.Fatalf("peer %d got HTTP %d before capacity", peer, w.Code)
		}
	}
	assertDetail(t, requestPeer(4096), 429, "rate limit exceeded")
	assertJSON(t, requestPeer(0), 200, map[string]any{"reply": ""})
	now = now.Add(time.Minute)
	assertJSON(t, requestPeer(4096), 200, map[string]any{"reply": ""})
}

func TestChatDuplicateObjectMembersAndInvalidRawUTF8FailBeforeBackend(t *testing.T) {
	var calls atomic.Int32
	s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(context.Context, string) (string, error) {
		calls.Add(1)
		return "", nil
	})})
	for _, body := range []string{
		`{"message":"first","message":"second"}`,
		`{"other":1,"other":2,"message":"hello"}`,
		"{\"message\":\"\xff\"}",
	} {
		assertDetail(t, serveContract(s, "POST", "/chat", body, ""), 400, "invalid JSON body")
	}
	if calls.Load() != 0 {
		t.Errorf("invalid duplicate/encoding request reached backend %d times", calls.Load())
	}
}

func TestChatOutputByteLimitAcceptsExactBoundaryAndCountsUTF8Bytes(t *testing.T) {
	for _, tc := range []struct {
		name, reply string
		status      int
	}{
		{"exact-ascii", strings.Repeat("x", MaxReplyBytes-len("{\"reply\":\"\"}\n")), 200},
		{"exact-utf8", strings.Repeat("é", (MaxReplyBytes-13)/2), 200},
		{"oversized-utf8", strings.Repeat("é", MaxReplyBytes/2+1), 500},
		{"oversized-before-trim", strings.Repeat(" ", MaxReplyBytes) + "x", 500},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(context.Context, string) (string, error) { return tc.reply, nil })})
			w := serveContract(s, "POST", "/chat", `{"message":"hello"}`, "")
			if tc.status == 200 {
				assertJSON(t, w, 200, map[string]any{"reply": tc.reply})
			} else {
				assertDetail(t, w, tc.status, "chat backend error")
			}
		})
	}
}

func TestChatMissingBackendFailsClosedAndDoesNotAffectPublicHealth(t *testing.T) {
	s := newContractServer(t, Config{AllowNoAuth: true})
	assertDetail(t, serveContract(s, "POST", "/chat", `{"message":"hello"}`, ""), 503, "chat backend unavailable")
	assertJSON(t, serveContract(s, "GET", "/healthz", "", ""), 200, map[string]any{"ok": true})
}

func TestDuplicateAuthorizationHeadersFailClosed(t *testing.T) {
	s := newContractServer(t, Config{RemoteToken: "contract-token"})
	r := httptest.NewRequest("POST", "/chat", strings.NewReader("{}"))
	r.Header.Add("Authorization", "Bearer contract-token")
	r.Header.Add("Authorization", "Bearer contract-token")
	w := httptest.NewRecorder()
	s.ServeHTTP(w, r)
	assertDetail(t, w, 401, "missing or invalid bearer token")
}

func TestChatSerializedReplyBoundIncludesControlEscaping(t *testing.T) {
	for _, tc := range []struct {
		reply  string
		status int
	}{
		{strings.Repeat("<", 11000), 200}, {strings.Repeat("\x00", 11000), 500},
		{strings.Repeat("x", MaxReplyBytes-13), 200}, {strings.Repeat("x", MaxReplyBytes-12), 500},
	} {
		s, err := New(Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(context.Context, string) (string, error) { return tc.reply, nil })})
		if err != nil {
			t.Fatal(err)
		}
		r := httptest.NewRequest("POST", "/chat", strings.NewReader(`{"message":"synthetic"}`))
		w := httptest.NewRecorder()
		s.ServeHTTP(w, r)
		_ = s.Close()
		if w.Code != tc.status || w.Body.Len() > MaxReplyBytes {
			t.Fatalf("status=%d bytes=%d", w.Code, w.Body.Len())
		}
	}
}

func TestPythonUnicodeWhitespaceAndDottedLowerCompatibility(t *testing.T) {
	if SanitizeIdentity("\x1cAlice\x1dSmith\x1f") != "Alice-Smith" {
		t.Fatal("Python whitespace identity contract changed")
	}
	if SanitizeRoomName("\x1cİA\x1f") != "i-a" {
		t.Fatal("Python dotted-I lowercase contract changed")
	}
	if text, err := messageBody([]byte(`{"message":"\u001c\u001f"}`)); err != nil || text != "" {
		t.Fatal("Python empty text contract changed")
	}
	s := newContractServer(t, Config{RemoteToken: " token with spaces "})
	if w := serveContract(s, "POST", "/chat", `{}`, "Bearer token with spaces"); w.Code != 200 {
		t.Fatal("exact bearer comparison contract changed")
	}
}

func TestBackendNilPanicFailsClosedUnderLegacyRuntimeSetting(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(context.Context, string) (string, error) { panic(nil) })})
	assertDetail(t, serveContract(s, "POST", "/chat", `{"message":"synthetic"}`, ""), 500, "chat backend error")
}
