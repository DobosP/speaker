// Speaker remote web client.
// Extracted from an inline <script> so the page can run under a strict
// Content-Security-Policy (script-src 'self', no 'unsafe-inline').
"use strict";

const logEl = document.getElementById("log");
const msgEl = document.getElementById("msg");

// Optional bearer token for authenticated deployments (SPEAKER_REMOTE_TOKEN on
// the server). Supply it via ?token=... in the URL (stored for the session) or
// localStorage.setItem("speaker_token", "..."). When no token is set the client
// sends no Authorization header, which works against a dev server running with
// SPEAKER_REMOTE_ALLOW_NOAUTH=1.
function authToken() {
  const q = new URLSearchParams(location.search).get("token");
  if (q) { try { localStorage.setItem("speaker_token", q); } catch (_) {} return q; }
  try { return localStorage.getItem("speaker_token") || ""; } catch (_) { return ""; }
}
function authHeaders(base) {
  const h = Object.assign({}, base || {});
  const t = authToken();
  if (t) h["Authorization"] = "Bearer " + t;
  return h;
}

function line(text, cls) {
  const d = document.createElement("div");
  d.className = "line " + (cls || "");
  d.textContent = text;
  logEl.appendChild(d);
  logEl.scrollTop = logEl.scrollHeight;
}

// --- text chat -----------------------------------------------------------
async function sendText() {
  const text = msgEl.value.trim();
  if (!text) return;
  msgEl.value = "";
  line("You: " + text, "you");
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    line("Assistant: " + (data.reply || "(no reply)"), "bot");
  } catch (e) {
    line("chat error: " + e, "sys");
  }
}
document.getElementById("send").onclick = sendText;
msgEl.addEventListener("keydown", (e) => { if (e.key === "Enter") sendText(); });

// --- live voice via LiveKit ---------------------------------------------
document.getElementById("voice").onclick = async () => {
  if (!window.LivekitClient) { line("LiveKit SDK failed to load", "sys"); return; }
  try {
    const params = new URLSearchParams({ identity: "web-" + Math.floor(Math.random() * 9999), room: "assistant" });
    const tk = await (await fetch("/token?" + params.toString(), { headers: authHeaders() })).json();
    if (!tk.url) { line("LIVEKIT_URL not configured on the server", "sys"); return; }

    const { Room, RoomEvent } = window.LivekitClient;
    const room = new Room();

    room.on(RoomEvent.TrackSubscribed, (track) => {
      if (track.kind === "audio") {
        const el = track.attach();
        el.autoplay = true;
        document.body.appendChild(el);
      }
    });
    room.on(RoomEvent.DataReceived, (payload) => {
      try {
        const msg = JSON.parse(new TextDecoder().decode(payload));
        if (msg.event_type === "user_transcript") line("You (voice): " + msg.payload.text, "you");
        else if (msg.event_type === "assistant_sentence") line("Assistant: " + msg.payload.text, "bot");
      } catch (_) {}
    });

    await room.connect(tk.url, tk.token);
    await room.localParticipant.setMicrophoneEnabled(true);
    line("Voice connected to room '" + tk.room + "'. Speak now.", "sys");
  } catch (e) {
    line("voice error: " + e, "sys");
  }
};
