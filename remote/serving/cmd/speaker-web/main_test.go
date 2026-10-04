package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestConfigPortAndCLIAdmission(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	if port, err := readPort(path); err != nil || port != 8080 {
		t.Fatal("missing default port")
	}
	for _, tc := range []struct {
		raw  string
		port int
		fail bool
	}{
		{`{}`, 8080, false}, {`{"remote":{"token_server_port":9090}}`, 9090, false}, {`{"remote":{"token_server_port":"bad"}}`, 0, true}, {`malformed`, 0, true},
	} {
		if err := os.WriteFile(path, []byte(tc.raw), 0600); err != nil {
			t.Fatal(err)
		}
		port, err := readPort(path)
		if (err != nil) != tc.fail || !tc.fail && port != tc.port {
			t.Fatal("config port mismatch")
		}
	}
	t.Setenv("SPEAKER_REMOTE_BIND_ALL", "1")
	t.Setenv("SPEAKER_REMOTE_TOKEN", "")
	t.Setenv("SPEAKER_REMOTE_ALLOW_NOAUTH", "1")
	if err := run(options{port: 8080}); err == nil {
		t.Fatal("unauthenticated bind-all admitted")
	}
	if err := run(options{port: -1}); err == nil {
		t.Fatal("invalid port admitted")
	}
	if err := run(options{port: 8080, echo: true}); err == nil {
		t.Fatal("implicit Python echo admitted")
	}
}
func TestLoopbackProbeNoRedirectOrProxy(t *testing.T) {
	for _, u := range []string{"https://127.0.0.1/healthz", "http://localhost/healthz", "http://192.168.1.1/healthz", "http://127.0.0.1/chat", "http://127.0.0.1/healthz?x=1", "http://user@127.0.0.1/healthz"} {
		if probe(u) == nil {
			t.Fatal("unsafe probe admitted")
		}
	}
	t.Setenv("HTTP_PROXY", "http://127.0.0.1:1")
	good := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte(`{"ok":true}`)) }))
	defer good.Close()
	if err := probe(good.URL + "/healthz"); err != nil {
		t.Fatal(err)
	}
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, good.URL+"/healthz", 302) }))
	defer bad.Close()
	if err := probe(bad.URL + "/healthz"); err == nil {
		t.Fatal("probe followed redirect")
	}
}

func TestBindAllCannotTreatPythonWhitespaceAsConfiguredBearer(t *testing.T) {
	t.Setenv("SPEAKER_REMOTE_BIND_ALL", "1")
	t.Setenv("SPEAKER_REMOTE_ALLOW_NOAUTH", "1")
	for _, token := range []string{"", "\x1c", "\x1f\r\n", "\u2003"} {
		t.Setenv("SPEAKER_REMOTE_TOKEN", token)
		if err := run(options{port: 8080}); err == nil || err.Error() != "bind-all requires SPEAKER_REMOTE_TOKEN" {
			t.Fatal("no-auth bind-all admission policy mismatch", err)
		}
	}
}
