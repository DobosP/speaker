// speaker-web is an optional, dormant rollback HTTP facade, never an audio entry.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"speaker/remote/serving"
)

type options struct {
	config, web, python, repo, probe string
	port                             int
	rollback, echo                   bool
}

func readPort(path string) (int, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return 8080, nil
	}
	if err != nil {
		return 0, errors.New("configuration unavailable")
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, 1024*1024+1))
	if err != nil || len(raw) > 1024*1024 {
		return 0, errors.New("configuration rejected")
	}
	var cfg struct {
		Remote struct {
			Port *int `json:"token_server_port"`
		} `json:"remote"`
	}
	if json.Unmarshal(raw, &cfg) != nil {
		return 0, errors.New("configuration rejected")
	}
	if cfg.Remote.Port == nil {
		return 8080, nil
	}
	return *cfg.Remote.Port, nil
}
func probe(raw string) error {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "http" || u.User != nil || u.Path != "/healthz" || u.RawQuery != "" || u.Fragment != "" {
		return errors.New("probe requires loopback HTTP /healthz")
	}
	ip := net.ParseIP(u.Hostname())
	if ip == nil || !ip.IsLoopback() {
		return errors.New("probe requires loopback HTTP /healthz")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", raw, nil)
	if err != nil {
		return errors.New("probe failed")
	}
	transport := &http.Transport{Proxy: nil}
	defer transport.CloseIdleConnections()
	client := &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	res, err := client.Do(req)
	if err != nil {
		return errors.New("probe failed")
	}
	defer res.Body.Close()
	body, err := io.ReadAll(io.LimitReader(res.Body, 1025))
	if err != nil || len(body) > 1024 || res.StatusCode != 200 {
		return errors.New("probe failed")
	}
	var health map[string]bool
	if json.Unmarshal(body, &health) != nil || len(health) != 1 || !health["ok"] {
		return errors.New("probe failed")
	}
	return nil
}
func run(o options) error {
	if o.probe != "" {
		return probe(o.probe)
	}
	if o.port == 0 {
		p, err := readPort(o.config)
		if err != nil {
			return err
		}
		o.port = p
	}
	if o.port < 1 || o.port > 65535 {
		return errors.New("port must be 1-65535")
	}
	cfg := serving.Config{RemoteToken: os.Getenv("SPEAKER_REMOTE_TOKEN"), AllowNoAuth: os.Getenv("SPEAKER_REMOTE_ALLOW_NOAUTH") == "1", LiveKitURL: os.Getenv("LIVEKIT_URL"), APIKey: os.Getenv("LIVEKIT_API_KEY"), APISecret: os.Getenv("LIVEKIT_API_SECRET"), RollbackVoice: o.rollback, WebDir: o.web}
	if o.python != "" {
		python, err := filepath.Abs(o.python)
		if err != nil {
			return errors.New("Python adapter unavailable")
		}
		info, err := os.Stat(python)
		if err != nil || !info.Mode().IsRegular() {
			return errors.New("Python adapter unavailable")
		}
		config, err := filepath.Abs(o.config)
		if err != nil {
			return errors.New("configuration unavailable")
		}
		repo, err := filepath.Abs(o.repo)
		if err != nil {
			return errors.New("repo directory unavailable")
		}
		cfg.ChatBackend = &serving.ProcessBackend{Python: python, ConfigPath: config, RepoDir: repo, Echo: o.echo}
	} else if o.echo {
		return errors.New("--chat-echo requires --chat-python")
	}
	host := "127.0.0.1"
	if os.Getenv("SPEAKER_REMOTE_BIND_ALL") == "1" {
		// Explicit bind-all cannot accidentally combine with unauthenticated dev mode.
		if strings.TrimSpace(cfg.RemoteToken) == "" {
			return errors.New("bind-all requires SPEAKER_REMOTE_TOKEN")
		}
		host = "0.0.0.0"
		log.Print("WARNING: explicit rollback facade bind-all requested")
	}
	if strings.TrimSpace(cfg.RemoteToken) == "" {
		if cfg.AllowNoAuth {
			log.Print("WARNING: dev no-auth enabled for loopback facade")
		} else {
			log.Print("remote auth unset; protected routes deny requests")
		}
	}
	handler, err := serving.New(cfg)
	if err != nil {
		return err
	}
	defer handler.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	server := &http.Server{Addr: net.JoinHostPort(host, strconv.Itoa(o.port)), Handler: handler, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 10 * time.Second, WriteTimeout: 35 * time.Second, IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16 * 1024, BaseContext: func(net.Listener) context.Context { return ctx }}
	listener, err := net.Listen("tcp", server.Addr)
	if err != nil {
		return errors.New("listener unavailable")
	}
	done := make(chan error, 1)
	go func() { done <- server.Serve(listener) }()
	log.Printf("optional rollback facade listening on %s", server.Addr)
	select {
	case <-ctx.Done():
		shutdown, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		if server.Shutdown(shutdown) != nil {
			_ = server.Close()
		}
		err = <-done
	case err = <-done:
	}
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	if err != nil {
		return errors.New("HTTP server failed")
	}
	return nil
}
func main() {
	var o options
	flag.StringVar(&o.config, "config", "config.json", "local configuration path (port and optional Python adapter)")
	flag.StringVar(&o.web, "web-dir", "web", "directory containing the three static web assets")
	flag.StringVar(&o.repo, "repo-dir", ".", "repo directory for explicit Python pipe adapter")
	flag.IntVar(&o.port, "port", 0, "loopback HTTP port; 0 reads remote.token_server_port (default 8080)")
	flag.StringVar(&o.python, "chat-python", "", "explicit Python interpreter path; absent means /chat unavailable")
	flag.BoolVar(&o.echo, "chat-echo", false, "synthetic pipe qualification only (requires --chat-python)")
	flag.BoolVar(&o.rollback, "rollback-voice", false, "explicit legacy loopback-only token minting; no trusted-LAN activation")
	flag.StringVar(&o.probe, "probe-url", "", "probe a loopback HTTP /healthz then exit")
	flag.Parse()
	if flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "unexpected arguments")
		os.Exit(2)
	}
	if err := run(o); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
