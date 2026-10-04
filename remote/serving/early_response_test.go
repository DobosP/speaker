package serving

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const earlyTCPResponseBudget = 400 * time.Millisecond

type earlyBodyReadCounter struct {
	io.ReadCloser
	reads *atomic.Int64
}

func (b *earlyBodyReadCounter) Read(p []byte) (int, error) {
	b.reads.Add(1)
	return b.ReadCloser.Read(p)
}

type earlyTCPFixture struct {
	address string
	reads   atomic.Int64
	calls   atomic.Int64
	handled chan struct{}
}

// Use an actual net/http server: an in-process ResponseRecorder does not expose
// the server's automatic unread-body draining before a small response flush.
func newEarlyTCPFixture(t *testing.T, cfg Config) *earlyTCPFixture {
	t.Helper()
	f := &earlyTCPFixture{handled: make(chan struct{}, 64)}
	cfg.ChatBackend = BackendFunc(func(context.Context, string) (string, error) {
		f.calls.Add(1)
		return "backend must not be invoked by early-response cases", nil
	})
	handler := newContractServer(t, cfg)
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	f.address = listener.Addr().String()
	server := &http.Server{
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      35 * time.Second,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Body != nil && r.Body != http.NoBody {
				r.Body = &earlyBodyReadCounter{ReadCloser: r.Body, reads: &f.reads}
			}
			handler.ServeHTTP(w, r)
			f.handled <- struct{}{}
		}),
	}
	stopped := make(chan error, 1)
	go func() { stopped <- server.Serve(listener) }()
	t.Cleanup(func() {
		if err := server.Close(); err != nil {
			t.Errorf("close TCP server: %v", err)
		}
		if err := <-stopped; err != nil && !errors.Is(err, http.ErrServerClosed) {
			t.Errorf("TCP server: %v", err)
		}
	})
	return f
}

// Send headers only, despite announcing a body. The client never supplies a
// chunk, body byte, or terminating chunk. A final response must finish and the
// server must close its connection before the client's short read deadline.
func (f *earlyTCPFixture) stalledHeaders(t *testing.T, method, path, extraHeaders string, chunked bool, contentLength ...int) (*http.Response, []byte) {
	t.Helper()
	conn, err := net.DialTimeout("tcp4", f.address, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	framing := "Content-Length: 1000\r\n"
	if len(contentLength) != 0 {
		framing = "Content-Length: " + strconv.Itoa(contentLength[0]) + "\r\n"
	}
	if chunked {
		framing = "Transfer-Encoding: chunked\r\n"
	}
	request := method + " " + path + " HTTP/1.1\r\nHost: localhost\r\n" + framing + extraHeaders + "\r\n"
	start := time.Now()
	if err := conn.SetDeadline(start.Add(earlyTCPResponseBudget)); err != nil {
		t.Fatal(err)
	}
	if _, err := io.WriteString(conn, request); err != nil {
		t.Fatal(err)
	}
	var wire bytes.Buffer
	reader := bufio.NewReader(io.TeeReader(conn, &wire))
	response, err := http.ReadResponse(reader, &http.Request{Method: method})
	if err != nil {
		t.Fatalf("response did not arrive within %v with body withheld: %v", earlyTCPResponseBudget, err)
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusContinue {
		t.Fatal("server sent 100 Continue before an early final rejection")
	}
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatalf("full response body did not arrive within %v: %v", earlyTCPResponseBudget, err)
	}
	wireHeaders, _, _ := strings.Cut(wire.String(), "\r\n\r\n")
	if !response.Close || !strings.Contains(strings.ToLower(wireHeaders), "\r\nconnection: close") {
		t.Errorf("unread request body response did not announce Connection: close on the wire: close=%v headers=%q", response.Close, wireHeaders)
	}
	if _, err := reader.ReadByte(); !errors.Is(err, io.EOF) {
		t.Fatalf("server did not promptly close the stalled-body connection: %v", err)
	}
	if elapsed := time.Since(start); elapsed >= earlyTCPResponseBudget {
		t.Errorf("early full response and connection close took %v, bound %v", elapsed, earlyTCPResponseBudget)
	}
	select {
	case <-f.handled:
	case <-time.After(earlyTCPResponseBudget):
		t.Fatal("handler did not finish after its full TCP response")
	}
	if reads := f.reads.Load(); reads != 0 {
		t.Errorf("early response read the underlying request body %d times", reads)
	}
	if calls := f.calls.Load(); calls != 0 {
		t.Errorf("early response invoked backend %d times", calls)
	}
	return response, body
}

func assertEarlyTCPJSON(t *testing.T, response *http.Response, body []byte, status int, fragment string) {
	t.Helper()
	if response.StatusCode != status {
		t.Fatalf("TCP response = %d, want %d; body %q", response.StatusCode, status, body)
	}
	if !strings.HasPrefix(response.Header.Get("Content-Type"), "application/json") {
		t.Errorf("early response is not JSON: %q", response.Header.Get("Content-Type"))
	}
	if !strings.Contains(string(body), fragment) {
		t.Errorf("early TCP body %q lacks %q", body, fragment)
	}
}

func TestEarlyTCPAuthRejectsStalledContentLengthWithoutBodyRead(t *testing.T) {
	for _, tc := range []struct {
		name, token, headers, detail string
	}{
		{"unset", "", "", "remote auth not configured"},
		{"missing", "contract-token", "", "missing or invalid bearer token"},
		{"invalid", "contract-token", "Authorization: Bearer wrong\r\n", "missing or invalid bearer token"},
		{"duplicate", "contract-token", "Authorization: Bearer contract-token\r\nAuthorization: Bearer contract-token\r\n", "missing or invalid bearer token"},
		{"wrong-scheme", "contract-token", "Authorization: Basic contract-token\r\n", "missing or invalid bearer token"},
		{"missing-bearer-value", "contract-token", "Authorization: Bearer\r\n", "missing or invalid bearer token"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newEarlyTCPFixture(t, Config{RemoteToken: tc.token})
			response, body := f.stalledHeaders(t, "POST", "/chat", tc.headers, false)
			assertEarlyTCPJSON(t, response, body, 401, tc.detail)
		})
	}
}

func TestEarlyTCPAuthRejectsStalledChunkedBodyWithoutBodyRead(t *testing.T) {
	for _, headers := range []string{"", "Authorization: Bearer wrong\r\n"} {
		f := newEarlyTCPFixture(t, Config{RemoteToken: "contract-token"})
		response, body := f.stalledHeaders(t, "POST", "/chat", headers, true)
		assertEarlyTCPJSON(t, response, body, 401, "missing or invalid bearer token")
	}
}

func TestEarlyTCPExpectContinueDoesNotPrecedeAuthRejection(t *testing.T) {
	for _, chunked := range []bool{false, true} {
		f := newEarlyTCPFixture(t, Config{RemoteToken: "contract-token"})
		response, body := f.stalledHeaders(t, "POST", "/chat", "Expect: 100-continue\r\nAuthorization: Bearer wrong\r\n", chunked)
		assertEarlyTCPJSON(t, response, body, 401, "missing or invalid bearer token")
	}
}

func TestEarlyTCPWrongMethodsRejectWithoutReadingAnnouncedBody(t *testing.T) {
	for _, tc := range []struct{ method, path string }{
		{"POST", "/token"}, {"PUT", "/chat"}, {"POST", "/healthz"},
		{"POST", "/app.js"}, {"POST", "/missing-static"},
	} {
		t.Run(tc.method+tc.path, func(t *testing.T) {
			f := newEarlyTCPFixture(t, Config{AllowNoAuth: true})
			response, body := f.stalledHeaders(t, tc.method, tc.path, "", false)
			if strings.HasPrefix(tc.path, "/app") || tc.path == "/missing-static" {
				if response.StatusCode != 404 && response.StatusCode != 405 {
					t.Errorf("wrong static method = %d, want 404 or 405", response.StatusCode)
				}
			} else {
				assertEarlyTCPJSON(t, response, body, 405, "method not allowed")
			}
		})
	}
}

func TestEarlyTCPHealthAndTokenIgnoreUnexpectedStalledBodies(t *testing.T) {
	for _, tc := range []struct {
		name, method, path, headers string
		cfg                         Config
		status                      int
		fragment                    string
	}{
		{"health", "GET", "/healthz", "", Config{}, 200, `"ok":true`},
		{"token-unauthenticated", "GET", "/token", "", Config{RemoteToken: "contract-token"}, 401, "missing or invalid bearer token"},
		{"token-dormant", "GET", "/token", "Authorization: Bearer contract-token\r\n", Config{RemoteToken: "contract-token"}, 403, "voice transport unavailable"},
		{"token-rollback", "GET", "/token", "Authorization: Bearer contract-token\r\n", Config{
			RemoteToken: "contract-token", RollbackVoice: true, LiveKitURL: "ws://127.0.0.1:7880",
			APIKey: "synthetic-key", APISecret: "synthetic-secret",
		}, 200, `"token":`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newEarlyTCPFixture(t, tc.cfg)
			response, body := f.stalledHeaders(t, tc.method, tc.path, tc.headers, false)
			assertEarlyTCPJSON(t, response, body, tc.status, tc.fragment)
		})
	}
}

func TestEarlyTCPRateRejectionPrecedesReadingStalledBody(t *testing.T) {
	f := newEarlyTCPFixture(t, Config{RemoteToken: "contract-token"})
	client := &http.Client{Timeout: time.Second}
	defer client.CloseIdleConnections()
	for i := 0; i < 30; i++ {
		request, err := http.NewRequest("POST", "http://"+f.address+"/chat", nil)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Authorization", "Bearer contract-token")
		response, err := client.Do(request)
		if err != nil {
			t.Fatalf("prime rate limiter request %d: %v", i, err)
		}
		_, err = io.Copy(io.Discard, response.Body)
		response.Body.Close()
		if err != nil || response.StatusCode != 200 {
			t.Fatalf("prime rate limiter request %d: status=%d error=%v", i, response.StatusCode, err)
		}
		select {
		case <-f.handled:
		case <-time.After(time.Second):
			t.Fatal("priming handler did not finish")
		}
	}
	if reads := f.reads.Load(); reads != 0 {
		t.Fatalf("bodyless limiter priming performed request body reads: %d", reads)
	}
	response, body := f.stalledHeaders(t, "POST", "/chat", "Authorization: Bearer contract-token\r\n", false)
	assertEarlyTCPJSON(t, response, body, 429, "rate limit exceeded")
}

func TestEarlyTCPDeclaredOversizeRejectsBeforeReadingStalledBody(t *testing.T) {
	f := newEarlyTCPFixture(t, Config{RemoteToken: "contract-token"})
	response, body := f.stalledHeaders(t, "POST", "/chat", "Authorization: Bearer contract-token\r\n", false, MaxChatBytes+1)
	assertEarlyTCPJSON(t, response, body, 413, "request body too large")
}
