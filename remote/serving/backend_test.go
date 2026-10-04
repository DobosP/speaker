package serving

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// The test binary itself is a synthetic, model-free private-pipe worker.
func TestProcessHelper(t *testing.T) {
	if len(os.Args) < 3 || os.Args[len(os.Args)-2] != "worker" {
		return
	}
	mode := os.Args[len(os.Args)-1]
	_, _ = io.ReadAll(os.Stdin)
	switch mode {
	case "echo":
		fmt.Print(`{"reply":" synthetic "}`)
	case "error":
		fmt.Fprintln(os.Stderr, "private backend detail")
		os.Exit(1)
	case "null":
		fmt.Print(`{"reply":null}`)
	case "bad":
		fmt.Print(`not json`)
	case "extra":
		fmt.Print(`{"reply":"a","authority":"owner"}`)
	case "duplicate":
		fmt.Print(`{"reply":"a","reply":"b"}`)
	case "trailing":
		fmt.Print(`{"reply":"a"}{"reply":"b"}`)
	case "oversize":
		fmt.Print(strings.Repeat("x", MaxReplyBytes+1))
	case "utf8":
		_, _ = os.Stdout.Write([]byte{0xff})
	case "slow":
		time.Sleep(time.Hour)
	case "env":
		for _, name := range []string{"LIVEKIT_API_SECRET", "SPEAKER_REMOTE_TOKEN", "OPENROUTER_API_KEY", "HTTP_PROXY", "PYTHONPATH"} {
			if _, ok := os.LookupEnv(name); ok {
				os.Exit(2)
			}
		}
		fmt.Print(`{"reply":"isolated"}`)
	}
	os.Exit(0)
}
func fakeProcess(mode string) *ProcessBackend {
	return &ProcessBackend{command: func(ctx context.Context) *exec.Cmd {
		return exec.CommandContext(ctx, os.Args[0], "-test.run=^TestProcessHelper$", "--", "worker", mode)
	}}
}
func TestProcessBackendProtocolAndTerminal(t *testing.T) {
	for _, mode := range []string{"echo", "error", "null", "bad", "extra", "duplicate", "trailing", "oversize", "utf8"} {
		t.Run(mode, func(t *testing.T) {
			reply, err := fakeProcess(mode).Generate(context.Background(), "synthetic")
			if mode == "echo" {
				if err != nil || reply != " synthetic " {
					t.Fatalf("reply=%q err=%v", reply, err)
				}
			} else if err == nil || reply != "" {
				t.Fatal("malformed/failed source admitted")
			}
		})
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	reply, err := fakeProcess("slow").Generate(ctx, "synthetic")
	if err == nil || reply != "" || time.Since(start) > 2*time.Second {
		t.Fatal("cancelled process did not terminate promptly")
	}
}
func TestProcessBackendCredentialIsolationAndInputBound(t *testing.T) {
	for _, name := range []string{"LIVEKIT_API_SECRET", "SPEAKER_REMOTE_TOKEN", "OPENROUTER_API_KEY", "HTTP_PROXY", "PYTHONPATH"} {
		t.Setenv(name, "synthetic-test-value")
	}
	reply, err := fakeProcess("env").Generate(context.Background(), "synthetic")
	if err != nil || reply != "isolated" {
		t.Fatal("child environment was not isolated")
	}
	if _, err := fakeProcess("echo").Generate(context.Background(), strings.Repeat("x", MaxChatBytes)); err == nil {
		t.Fatal("oversized serialized IPC request accepted")
	}
	if _, err := (&ProcessBackend{}).Generate(context.Background(), "synthetic"); err == nil {
		t.Fatal("missing backend config admitted")
	}
}

func TestProcessBackendExactInputBoundaryHasNoFramingOverhead(t *testing.T) {
	reply, err := fakeProcess("echo").Generate(context.Background(), strings.Repeat("x", MaxChatBytes-len(`{"message":""}`)))
	if err != nil || reply != " synthetic " {
		t.Fatal("exact cap input rejected", err)
	}
}

func TestProcessBackendPreservesExactUTF8AndLiteralEscapes(t *testing.T) {
	for _, message := range []string{strings.Repeat("a\u2028", 4092), strings.Repeat("a\u2029", 4092), `literal \u2028 and \u2029`, "\x00\b\f\n\r\t\"\\"} {
		raw, err := textRequest(message)
		if err != nil {
			t.Fatal(err)
		}
		var got struct{ Message string }
		if json.Unmarshal(raw, &got) != nil || got.Message != message {
			t.Fatal("private pipe text rewrite")
		}
		if _, err := fakeProcess("echo").Generate(context.Background(), message); err != nil {
			t.Fatal(err)
		}
	}
}
