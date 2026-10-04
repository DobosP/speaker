package serving

import "unicode/utf8"

// validJSONUnicode checks the scalar integrity of every JSON string before
// encoding/json can replace invalid Unicode with U+FFFD. It does not validate
// JSON structure; the caller must still use encoding/json for that purpose.
func validJSONUnicode(raw []byte) bool {
	if !utf8.Valid(raw) {
		return false
	}
	inString := false
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if !inString {
			if c == '"' {
				inString = true
			}
			continue
		}
		switch c {
		case '"':
			inString = false
		case '\\':
			i++
			if i >= len(raw) {
				return false
			}
			switch raw[i] {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
				// Consume the whole escape, preserving escaped quote boundaries
				// and the parity of adjacent backslashes.
			case 'u':
				value, ok := jsonHexQuad(raw, i+1)
				if !ok {
					return false
				}
				i += 4
				if value >= 0xd800 && value <= 0xdbff {
					// A high surrogate must be immediately paired with an
					// escaped low surrogate, with no literal or escape between.
					next := i + 1
					if next+6 > len(raw) || raw[next] != '\\' || raw[next+1] != 'u' {
						return false
					}
					low, ok := jsonHexQuad(raw, next+2)
					if !ok || low < 0xdc00 || low > 0xdfff {
						return false
					}
					i = next + 5
				} else if value >= 0xdc00 && value <= 0xdfff {
					return false
				}
			default:
				return false
			}
		default:
			if c < 0x20 {
				return false
			}
		}
	}
	return !inString
}

func jsonHexQuad(raw []byte, start int) (uint16, bool) {
	if start < 0 || start+4 > len(raw) {
		return 0, false
	}
	var value uint16
	for _, c := range raw[start : start+4] {
		var digit byte
		switch {
		case c >= '0' && c <= '9':
			digit = c - '0'
		case c >= 'a' && c <= 'f':
			digit = c - 'a' + 10
		case c >= 'A' && c <= 'F':
			digit = c - 'A' + 10
		default:
			return 0, false
		}
		value = value<<4 | uint16(digit)
	}
	return value, true
}
