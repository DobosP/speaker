package serving

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync/atomic"
	"testing"
)

func TestJSONUnicodeAcceptsScalarValuesAndEscapedBackslashes(t *testing.T) {
	for _, raw := range []string{
		`{"message":"ASCII"}`,
		`{"message":"\ud800\udc00"}`,
		`{"message":"\uDBFF\uDFFF"}`,
		`{"message":"\uD83d\uDe00"}`,
		`{"message":"\ud7ff\ue000\u0000"}`,
		`{"message":"\uFFFD"}`,
		`{"message":"�😀"}`,
		`{"message":"\\ud800"}`,
		`{"message":"\\\\ud800"}`,
		`{"message":"\\\uD83D\uDE00"}`,
		`{"message":"escaped quote: \" and backslash: \\"}`,
		`{"message":"a\\","ignored":"\uFFFD"}`,
		`{"\ud83d\ude00":["text",{"ignored":"\ufffd"}],"message":"ok"}`,
		`["\ud83d\ude00",{"nested":["😀","\\udfff"]}]`,
		`"\ud83d\ude00"`,
	} {
		t.Run(raw, func(t *testing.T) {
			if !validJSONUnicode([]byte(raw)) {
				t.Fatalf("valid scalar sequence rejected: %s", raw)
			}
			var value any
			if json.Unmarshal([]byte(raw), &value) != nil {
				t.Fatalf("test fixture is not valid JSON: %s", raw)
			}
		})
	}
}

func TestJSONUnicodeRejectsUnpairedSurrogatesAndMalformedStrings(t *testing.T) {
	for _, raw := range []string{
		`{"message":"\ud800"}`,
		`{"message":"\uDBFF"}`,
		`{"message":"\udc00"}`,
		`{"message":"\uDFFF"}`,
		`{"message":"\ud800\ud801"}`,
		`{"message":"\ud800\ud7ff"}`,
		`{"message":"\ud800\ue000"}`,
		`{"message":"\ud800x\udc00"}`,
		`{"message":"\ud800 \udc00"}`,
		`{"message":"\ud800\\udc00"}`,
		`{"message":"\\\ud800"}`,
		`{"message":"\ud83d\ude00\udfff"}`,
		`{"message":"escaped quote: \" \ud800"}`,
		`{"message":"a\\","ignored":"\ud800"}`,
		`{"\ud800":"value","message":"ok"}`,
		`{"message":"ok","ignored":[{"nested":"\udfff"}]}`,
		`{"message":"\uD80g"}`,
		`{"message":"\u123"}`,
		`{"message":"\ud800\uDC0"}`,
		`{"message":"\ud800\uDC0g"}`,
		`{"message":"\u`,
		`{"message":"\ud800`,
		`{"message":"\ud800\`,
		`{"message":"\ud800\u`,
		`{"message":"trailing\`,
		`{"message":"\q"}`,
		"{\"message\":\"raw\x00control\"}",
		"{\"message\":\"raw\ncontrol\"}",
	} {
		t.Run(raw, func(t *testing.T) {
			if validJSONUnicode([]byte(raw)) {
				t.Fatalf("malformed string admitted: %s", raw)
			}
		})
	}
	for _, raw := range [][]byte{
		{0xff},
		[]byte("{\"message\":\"\xff\"}"),
		[]byte("{\"message\":\"\xed\xa0\x80\"}"),     // UTF-8 encoding of a surrogate.
		[]byte("{\"message\":\"\xf4\x90\x80\x80\"}"), // Beyond U+10FFFF.
		[]byte("{\"message\":\"\xe2\x82\"}"),         // Truncated UTF-8.
	} {
		if validJSONUnicode(raw) {
			t.Fatalf("invalid raw UTF-8 admitted: %x", raw)
		}
	}
}

func TestJSONUnicodePrecheckLeavesStructureToJSONDecoder(t *testing.T) {
	for _, raw := range []string{"", "not JSON", `{"message":"ok"`, `{"message":"ok"} trailing`} {
		if !validJSONUnicode([]byte(raw)) {
			t.Fatalf("scalar-only precheck rejected string-safe structure: %q", raw)
		}
		var value any
		if json.Unmarshal([]byte(raw), &value) == nil {
			t.Fatalf("invalid structure fixture decoded: %q", raw)
		}
	}
}

// This compiled Go test process emits adversarial private-pipe JSON. It needs
// no Python, listener, model, provider, or assistant runtime.
func TestUnicodeProcessHelper(t *testing.T) {
	if len(os.Args) < 3 || os.Args[len(os.Args)-2] != "unicode-worker" {
		return
	}
	_, _ = io.Copy(io.Discard, os.Stdin)
	mode := os.Args[len(os.Args)-1]
	switch mode {
	case "high":
		fmt.Print(`{"reply":"private-prefix\ud800"}`)
	case "low":
		fmt.Print(`{"reply":"private-prefix\udfff"}`)
	case "adjacent-high":
		fmt.Print(`{"reply":"private-prefix\ud800\ud801"}`)
	case "pair":
		fmt.Print(`{"reply":"\ud83d\ude00"}`)
	case "replacement":
		fmt.Print(`{"reply":"�"}`)
	case "escaped-replacement":
		fmt.Print(`{"reply":"\ufffd"}`)
	case "backslash":
		fmt.Print(`{"reply":"\\ud800"}`)
	case "quote-boundary":
		fmt.Print(`{"reply":"escaped quote: \" and literal \\ud800"}`)
	default:
		os.Exit(2)
	}
	os.Exit(0)
}

func unicodeProcess(mode string) *ProcessBackend {
	return &ProcessBackend{command: func(ctx context.Context) *exec.Cmd {
		return exec.CommandContext(ctx, os.Args[0], "-test.run=^TestUnicodeProcessHelper$", "--", "unicode-worker", mode)
	}}
}

func TestUnicodePrivatePipeRejectsUnpairedSurrogatesWithoutPublishingText(t *testing.T) {
	for _, mode := range []string{"high", "low", "adjacent-high"} {
		t.Run(mode, func(t *testing.T) {
			backend := unicodeProcess(mode)
			reply, err := backend.Generate(context.Background(), "synthetic")
			if err == nil || reply != "" || err.Error() != "backend output rejected" {
				t.Fatalf("invalid scalar reply escaped: reply=%q err=%v", reply, err)
			}
			s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: backend})
			w := serveContract(s, "POST", "/chat", `{"message":"synthetic"}`, "")
			assertDetail(t, w, 500, "chat backend error")
			if strings.Contains(w.Body.String(), "private-prefix") {
				t.Fatal("rejected provider text escaped to HTTP")
			}
		})
	}
}

func TestUnicodePrivatePipePreservesValidPairsAndLiteralText(t *testing.T) {
	for _, tc := range []struct{ mode, want string }{
		{"pair", "😀"},
		{"replacement", "�"},
		{"escaped-replacement", "�"},
		{"backslash", `\ud800`},
		{"quote-boundary", `escaped quote: " and literal \ud800`},
	} {
		t.Run(tc.mode, func(t *testing.T) {
			reply, err := unicodeProcess(tc.mode).Generate(context.Background(), "synthetic")
			if err != nil || reply != tc.want {
				t.Fatalf("valid scalar text changed: reply=%q want=%q err=%v", reply, tc.want, err)
			}
		})
	}
}

func TestUnicodeHTTPRejectsMalformedStringsAnywhereBeforeBackend(t *testing.T) {
	var calls atomic.Int32
	s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(context.Context, string) (string, error) {
		calls.Add(1)
		return "synthetic", nil
	})})
	for _, raw := range []string{
		`{"message":"\ud800"}`,
		`{"message":"\udfff"}`,
		`{"message":"ok","ignored":{"nested":"\ud800"}}`,
		`{"message":"ok","\udfff":"ignored"}`,
		`[{"ignored":"\ud800"}]`,
		`{"message":"","ignored":"\udfff"}`,
	} {
		assertDetail(t, serveContract(s, "POST", "/chat", raw, ""), 400, "invalid JSON body")
	}
	if calls.Load() != 0 {
		t.Fatalf("malformed scalar JSON invoked backend %d times", calls.Load())
	}
}

func TestUnicodeHTTPPreservesValidPairsReplacementAndEscapedBackslashes(t *testing.T) {
	var received string
	s := newContractServer(t, Config{AllowNoAuth: true, ChatBackend: BackendFunc(func(_ context.Context, message string) (string, error) {
		received = message
		return message, nil
	})})
	for _, tc := range []struct{ raw, want string }{
		{`{"message":"\ud83d\ude00"}`, "😀"},
		{`{"message":"�"}`, "�"},
		{`{"message":"\\ud800"}`, `\ud800`},
		{`{"message":"\u0000"}`, "\x00"},
	} {
		w := serveContract(s, "POST", "/chat", tc.raw, "")
		assertJSON(t, w, 200, map[string]any{"reply": tc.want})
		if received != tc.want {
			t.Fatalf("backend text changed: got=%q want=%q", received, tc.want)
		}
	}
}
