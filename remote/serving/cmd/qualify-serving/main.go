// qualify-serving exercises the actual binary using synthetic loopback only.
// Its ordinary path requires Go and the binary, with no Python/Rust/model.
package main

import (
	"bufio"
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const auth = "synthetic-qualification-bearer"
const key = "synthetic-qualification-key"
const secret = "synthetic-qualification-secret-with-thirty-two-bytes"

type fixture struct {
	cmd    *exec.Cmd
	client *http.Client
	base   string
	closed bool
}

func launch(binary, repo, python string) (*fixture, error) {
	reserved, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	port := reserved.Addr().(*net.TCPAddr).Port
	_ = reserved.Close()
	args := []string{"--port", strconv.Itoa(port), "--web-dir", filepath.Join(repo, "web"), "--config", filepath.Join(repo, "config.json"), "--rollback-voice"}
	if python != "" {
		args = append(args, "--repo-dir", repo, "--chat-python", python, "--chat-echo")
	}
	cmd := exec.Command(binary, args...)
	cmd.Dir = repo
	cmd.Env = []string{"PATH=/usr/bin:/bin", "SPEAKER_REMOTE_TOKEN=" + auth, "LIVEKIT_API_KEY=" + key, "LIVEKIT_API_SECRET=" + secret, "LIVEKIT_URL=ws://127.0.0.1:7880"}
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if err = cmd.Start(); err != nil {
		return nil, errors.New("qualification start failed")
	}
	f := &fixture{cmd: cmd, client: &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{Proxy: nil}, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}, base: "http://127.0.0.1:" + strconv.Itoa(port)}
	for i := 0; i < 250; i++ {
		status, _, _, err := f.request("GET", "/healthz", nil, false)
		if err == nil && status == 200 {
			return f, nil
		}
		time.Sleep(20 * time.Millisecond)
	}
	f.close()
	return nil, errors.New("qualification readiness failed")
}
func (f *fixture) close() {
	if f.closed {
		return
	}
	f.closed = true
	f.client.CloseIdleConnections()
	_ = f.cmd.Process.Signal(os.Interrupt)
	done := make(chan struct{})
	go func() { _ = f.cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		_ = f.cmd.Process.Kill()
		<-done
	}
}
func (f *fixture) request(method, path string, body []byte, authenticated bool) (int, http.Header, []byte, error) {
	r, err := http.NewRequest(method, f.base+path, bytes.NewReader(body))
	if err != nil {
		return 0, nil, nil, err
	}
	r.Header.Set("Content-Type", "application/json")
	if authenticated {
		r.Header.Set("Authorization", "Bearer "+auth)
	}
	response, err := f.client.Do(r)
	if err != nil {
		return 0, nil, nil, err
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(response.Body, 2*1024*1024))
	return response.StatusCode, response.Header, raw, err
}
func require(condition bool) error {
	if !condition {
		return errors.New("qualification contract failed")
	}
	return nil
}
func verify(f *fixture, binary, repo string, pipe bool) error {
	if err := verifyStalledUnauthorized(f); err != nil {
		return err
	}
	for _, tc := range []struct {
		method, path, body string
		auth               bool
		status             int
	}{
		{"GET", "/healthz", "", false, 200}, {"GET", "/token", "", false, 401}, {"POST", "/chat", `{"message":"synthetic"}`, false, 401},
		{"GET", "/config.json", "", true, 404}, {"GET", "/../config.json", "", true, 404}, {"GET", "/token/", "", true, 404}, {"GET", "/docs/", "", true, 404},
		{"POST", "/chat", `{"message":42}`, true, 400},
		{"POST", "/chat", `{"message":"\ud800"}`, true, 400},
		{"POST", "/chat", `{"message":"\udc00"}`, true, 400},
		{"POST", "/chat", `{"message":"safe","ignored":{"bad":"\ud800"}}`, true, 400}, {"POST", "/chat", strings.Repeat("x", 16*1024+1), true, 413},
	} {
		status, _, _, err := f.request(tc.method, tc.path, []byte(tc.body), tc.auth)
		if err != nil {
			return err
		}
		if err = require(status == tc.status); err != nil {
			return err
		}
	}
	status, _, raw, err := f.request("POST", "/chat", []byte(`{"message":"  synthetic turn  "}`), true)
	if err != nil {
		return err
	}
	if pipe {
		if err = require(status == 200 && string(raw) == "{\"reply\":\"You said: synthetic turn\"}\n"); err != nil {
			return err
		}
	} else {
		if err = require(status == 503); err != nil {
			return err
		}
	}
	if pipe {
		for _, tc := range []struct{ raw, want string }{
			{`{"message":"\ud83d\ude42"}`, "You said: 🙂"},
			{`{"message":"�"}`, "You said: �"},
			{`{"message":"\\ud800"}`, `You said: \ud800`},
		} {
			status, _, body, err := f.request("POST", "/chat", []byte(tc.raw), true)
			if err != nil {
				return err
			}
			var reply map[string]string
			if json.Unmarshal(body, &reply) != nil || status != 200 || reply["reply"] != tc.want {
				return errors.New("Unicode text preservation failed")
			}
		}
	}
	for _, name := range []string{"index.html", "app.js", "livekit-client.umd.min.js"} {
		path := "/" + name
		if name == "index.html" {
			path = "/"
		}
		status, headers, raw, err := f.request("GET", path, nil, false)
		if err != nil {
			return err
		}
		expected, err := os.ReadFile(filepath.Join(repo, "web", name))
		if err != nil {
			return err
		}
		if err = require(status == 200 && bytes.Equal(raw, expected) && headers.Get("X-Content-Type-Options") == "nosniff" && strings.Contains(headers.Get("Content-Security-Policy"), "frame-ancestors 'none'")); err != nil {
			return err
		}
	}
	status, _, raw, err = f.request("GET", "/token?identity=Alice+Smith&room=My+Room!", nil, true)
	if err != nil {
		return err
	}
	var token map[string]string
	if json.Unmarshal(raw, &token) != nil || status != 200 || len(token) != 4 {
		return errors.New("token contract failed")
	}
	parts := strings.Split(token["token"], ".")
	if len(parts) != 3 {
		return errors.New("token shape failed")
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil {
		return err
	}
	mac := hmac.New(sha256.New, []byte(secret))
	_, _ = mac.Write([]byte(parts[0] + "." + parts[1]))
	if !hmac.Equal(signature, mac.Sum(nil)) {
		return errors.New("token signature failed")
	}
	claimsRaw, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return err
	}
	var claims struct {
		Iss, Sub, Name string
		Nbf, Exp       int64
		Video          map[string]any
	}
	if json.Unmarshal(claimsRaw, &claims) != nil {
		return errors.New("token claims failed")
	}
	if err = require(claims.Iss == key && claims.Sub == "Alice-Smith" && claims.Name == claims.Sub && claims.Exp-claims.Nbf == 3600 && claims.Video["room"] == "my-room" && len(claims.Video) == 5 && claims.Video["roomJoin"] == true && claims.Video["canPublish"] == true && claims.Video["canSubscribe"] == true && claims.Video["canPublishData"] == true); err != nil {
		return err
	}
	probe := exec.Command(binary, "--probe-url", f.base+"/healthz")
	probe.Env = []string{"PATH=/usr/bin:/bin"}
	if probe.Run() != nil {
		return errors.New("native probe failed")
	}
	return nil
}
func rss(pid int) (idle, peak int64) {
	raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 {
			n, _ := strconv.ParseInt(fields[1], 10, 64)
			switch fields[0] {
			case "VmRSS:":
				idle = n
			case "VmHWM:":
				peak = n
			}
		}
	}
	return
}
func cpu(pid int) int64 {
	raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return 0
	}
	end := strings.LastIndex(string(raw), ")")
	if end < 0 {
		return 0
	}
	fields := strings.Fields(string(raw)[end+1:])
	if len(fields) < 13 {
		return 0
	}
	a, _ := strconv.ParseInt(fields[11], 10, 64)
	b, _ := strconv.ParseInt(fields[12], 10, 64)
	return a + b
}
func measure(f *fixture) (map[string]any, error) {
	const n = 800
	for i := 0; i < 20; i++ {
		_, _, _, _ = f.request("GET", "/healthz", nil, false)
	}
	idle, _ := rss(f.cmd.Process.Pid)
	startCPU := cpu(f.cmd.Process.Pid)
	elapsed := make([]float64, 0, n)
	var bodyBytes int
	start := time.Now()
	for i := 0; i < n; i++ {
		path := "/healthz"
		if i%5 == 4 {
			path = "/"
		}
		begin := time.Now()
		status, _, body, err := f.request("GET", path, nil, false)
		if err != nil || status != 200 {
			return nil, errors.New("resource request failed")
		}
		elapsed = append(elapsed, float64(time.Since(begin))/float64(time.Millisecond))
		bodyBytes += len(body)
	}
	wall := time.Since(start).Seconds()
	ticks := cpu(f.cmd.Process.Pid) - startCPU
	_, peak := rss(f.cmd.Process.Pid)
	sort.Float64s(elapsed)
	result := map[string]any{"requests": n, "concurrency": 1, "mix": "80% healthz / 20% shipped index, warm sequential keepalive", "response_body_bytes": bodyBytes, "p50_ms": elapsed[n*50/100], "p95_ms": elapsed[n*95/100], "p99_ms": elapsed[n*99/100], "throughput_rps": float64(n) / wall, "idle_rss_kib": idle, "peak_rss_kib": peak, "server_cpu_ticks": ticks, "server_processes": 1, "scope": "server only; native driver/probe excluded", "python_http_baseline": "not measured; no legacy HTTP assembly supplied to this native qualifier"}
	tickRaw, err := exec.Command("getconf", "CLK_TCK").Output()
	if err == nil {
		hz, _ := strconv.ParseFloat(strings.TrimSpace(string(tickRaw)), 64)
		if hz > 0 {
			result["server_cpu_ms_per_request"] = float64(ticks) * 1000 / hz / n
			result["cpu_tick_hz"] = hz
		}
	}
	return result, nil
}
func run(binary, repo, python string) (map[string]any, error) {
	f, err := launch(binary, repo, "")
	if err != nil {
		return nil, err
	}
	defer f.close()
	if err = verify(f, binary, repo, false); err != nil {
		return nil, err
	}
	metrics, err := measure(f)
	if err != nil {
		return nil, err
	}
	f.close() // Settle the ordinary fixture before admitting the separate pipe assembly.
	result := map[string]any{"ordinary_go_http": "passed", "jwt_signature_ttl_grants": "passed", "static_bytes_confinement": "passed", "resources": metrics, "audio_model_provider_invocations": 0, "fixture_go_processes_max": 1, "native_driver_processes": 1, "short_lived_probe_processes_max": 1, "explicit_python_children_per_turn_max": 0, "listeners": "synthetic loopback only", "synthetic_python_pipe": "not requested"}
	if python != "" {
		pipe, err := launch(binary, repo, python)
		if err != nil {
			return nil, err
		}
		defer pipe.close()
		if err = verify(pipe, binary, repo, true); err != nil {
			return nil, err
		}
		result["synthetic_python_pipe"] = "passed (one explicit Python child per turn; model costs unmeasured)"
		result["explicit_python_children_per_turn_max"] = 1
	}
	return result, nil
}
func main() {
	binary := flag.String("binary", "", "already built speaker-web binary")
	repo := flag.String("repo", "../..", "repository directory")
	python := flag.String("chat-python", "", "optional explicit interpreter for synthetic echo pipe qualification")
	flag.Parse()
	b, err := filepath.Abs(*binary)
	if err != nil || *binary == "" {
		fmt.Fprintln(os.Stderr, "--binary required")
		os.Exit(2)
	}
	r, err := filepath.Abs(*repo)
	if err != nil {
		os.Exit(2)
	}
	p := *python
	if p != "" {
		p, err = filepath.Abs(p)
		if err != nil {
			os.Exit(2)
		}
	}
	result, err := run(b, r, p)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	_ = json.NewEncoder(os.Stdout).Encode(result)
}

// Exercise the built listener, not just the handler: net/http must not drain a
// body withheld after unauthorized headers before sending the final response.
func verifyStalledUnauthorized(f *fixture) error {
	conn, err := net.DialTimeout("tcp", strings.TrimPrefix(f.base, "http://"), time.Second)
	if err != nil {
		return errors.New("stalled-body qualification connect failed")
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(400 * time.Millisecond))
	if _, err = io.WriteString(conn, "POST /chat HTTP/1.1\r\nHost: local.test\r\nContent-Length: 1000\r\n\r\n"); err != nil {
		return errors.New("stalled-body qualification write failed")
	}
	response, err := http.ReadResponse(bufio.NewReader(conn), nil)
	if err != nil {
		return errors.New("unauthorized response waited for absent body")
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil || response.StatusCode != 401 || !response.Close || !bytes.Contains(body, []byte("missing or invalid bearer token")) {
		return errors.New("stalled-body refusal contract failed")
	}
	return nil
}
