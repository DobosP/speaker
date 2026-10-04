package serving

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
	"unicode/utf8"
)

// ProcessBackend runs one stateless Python inference turn over private pipes.
// It never starts or proxies a Python HTTP server. Source terminal is cmd.Wait;
// a cancelled source retains its Server BUSY slot until that wait returns.
// It cannot attest termination of a separate Ollama/native server operation.
type ProcessBackend struct {
	Python     string
	ConfigPath string
	RepoDir    string
	Echo       bool                            // synthetic qualification only; explicit CLI opt-in
	command    func(context.Context) *exec.Cmd // test-only fake process injection
}

func (p *ProcessBackend) Generate(parent context.Context, message string) (string, error) {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	request, err := textRequest(message)
	if err != nil {
		return "", err
	}
	var cmd *exec.Cmd
	if p.command != nil {
		cmd = p.command(ctx)
	} else {
		if !filepath.IsAbs(p.Python) || !filepath.IsAbs(p.ConfigPath) || !filepath.IsAbs(p.RepoDir) {
			return "", errors.New("backend configuration unavailable")
		}
		args := []string{"-B", "-m", "remote.text_backend", "--config", p.ConfigPath}
		if p.Echo {
			args = append(args, "--echo")
		}
		cmd = exec.CommandContext(ctx, p.Python, args...)
		cmd.Dir = p.RepoDir
	}
	// Never copy provider/LiveKit/remote credentials, proxy variables, PYTHONPATH,
	// or arbitrary ambient configuration into the inference child.
	cmd.Env = []string{"PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1"}
	for _, name := range []string{"PATH", "SYSTEMROOT", "OLLAMA_HOST", "OLLAMA_KEEP_ALIVE"} {
		if v, ok := os.LookupEnv(name); ok {
			cmd.Env = append(cmd.Env, name+"="+v)
		}
	}
	cmd.Stdin = bytes.NewReader(request)
	output := &cappedOutput{cancel: cancel}
	cmd.Stdout = output
	cmd.Stderr = io.Discard
	cmd.WaitDelay = time.Second
	if err := cmd.Run(); err != nil {
		return "", errors.New("backend process failed")
	}
	raw := output.Bytes()
	if parent.Err() != nil || output.overflow || !validJSONUnicode(raw) {
		return "", errors.New("backend output rejected")
	}
	// Require exactly one object member and one complete response document.
	d := json.NewDecoder(bytes.NewReader(raw))
	tok, err := d.Token()
	if err != nil || tok != json.Delim('{') || !d.More() {
		return "", errors.New("backend output rejected")
	}
	key, err := d.Token()
	if err != nil || key != "reply" {
		return "", errors.New("backend output rejected")
	}
	value, err := d.Token()
	reply, isString := value.(string)
	if err != nil || !isString || d.More() {
		return "", errors.New("backend output rejected")
	}
	tok, err = d.Token()
	if err != nil || tok != json.Delim('}') {
		return "", errors.New("backend output rejected")
	}
	if err := d.Decode(new(any)); err != io.EOF {
		return "", errors.New("backend output rejected")
	}
	return reply, nil
}

type cappedOutput struct {
	mu       sync.Mutex
	buf      bytes.Buffer
	overflow bool
	cancel   context.CancelFunc
}

func (b *cappedOutput) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(p) > MaxReplyBytes-b.buf.Len() {
		b.overflow = true
		b.cancel()
		return 0, errors.New("output limit")
	}
	return b.buf.Write(p)
}
func (b *cappedOutput) Bytes() []byte {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]byte(nil), b.buf.Bytes()...)
}

// textRequest writes only JSON-required escapes, retaining raw UTF-8 U+2028/29.
// encoding/json's JavaScript-safe escaping can otherwise expand an admitted
// exact-cap HTTP message beyond the private pipe cap.
func textRequest(message string) ([]byte, error) {
	if !utf8.ValidString(message) || len(message) > MaxChatBytes {
		return nil, errors.New("backend input rejected")
	}
	var out bytes.Buffer
	out.WriteString(`{"message":"`)
	const hex = "0123456789abcdef"
	for i := 0; i < len(message); i++ {
		c := message[i]
		switch c {
		case '"', '\\':
			out.WriteByte('\\')
			out.WriteByte(c)
		case '\b':
			out.WriteString(`\b`)
		case '\f':
			out.WriteString(`\f`)
		case '\n':
			out.WriteString(`\n`)
		case '\r':
			out.WriteString(`\r`)
		case '\t':
			out.WriteString(`\t`)
		default:
			if c < 32 {
				out.WriteString(`\u00`)
				out.WriteByte(hex[c>>4])
				out.WriteByte(hex[c&15])
			} else {
				out.WriteByte(c)
			}
		}
		if out.Len() > MaxChatBytes-2 {
			return nil, errors.New("backend input rejected")
		}
	}
	out.WriteString(`"}`)
	if out.Len() > MaxChatBytes {
		return nil, errors.New("backend input rejected")
	}
	return out.Bytes(), nil
}
