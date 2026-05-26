/*
 * Speaker mobile (React Native) - MVP scaffold.
 *
 * This is a STARTING POINT and cannot be built/verified in the cloud container.
 * Setup on your machine:
 *   1. Install a React Native toolchain (RN CLI or Expo bare workflow).
 *   2. From mobile/:  npm install     (adjust versions in package.json as needed)
 *   3. Set TOKEN_SERVER below to your token server (remote/token_server.py),
 *      reachable from the phone (same LAN, or a tunnel). Use HTTPS/WSS in prod;
 *      plain http/localhost is fine only in dev.
 *   4. Permissions: iOS NSMicrophoneUsageDescription; Android RECORD_AUDIO.
 *   5. Run:  npm run ios   (or)   npm run android
 *
 * It connects to the LiveKit room, publishes the mic, plays the assistant audio
 * track, and shows transcripts from the room data channel - the same server side
 * as the web client.
 */
import React, { useState, useCallback } from 'react';
import { SafeAreaView, Text, Button, ScrollView } from 'react-native';
import {
  AudioSession,
  LiveKitRoom,
  useDataChannel,
  registerGlobals,
} from '@livekit/react-native';

registerGlobals();

const TOKEN_SERVER = 'http://10.0.0.2:8080'; // <-- set to your token server

export default function App() {
  const [conn, setConn] = useState(null);

  const connect = useCallback(async () => {
    await AudioSession.startAudioSession();
    const id = 'mobile-' + Math.floor(Math.random() * 9999);
    const res = await fetch(`${TOKEN_SERVER}/token?identity=${id}&room=assistant`);
    const data = await res.json();
    setConn({ url: data.url, token: data.token });
  }, []);

  if (!conn) {
    return (
      <SafeAreaView style={{ flex: 1, justifyContent: 'center', padding: 24 }}>
        <Text style={{ fontSize: 22, marginBottom: 16 }}>Speaker</Text>
        <Button title="Connect voice" onPress={connect} />
      </SafeAreaView>
    );
  }

  return (
    <LiveKitRoom serverUrl={conn.url} token={conn.token} connect={true} audio={true}>
      <Transcripts />
    </LiveKitRoom>
  );
}

function Transcripts() {
  const [lines, setLines] = useState([]);
  useDataChannel((msg) => {
    try {
      const data = JSON.parse(new TextDecoder().decode(msg.payload));
      const who = data.event_type === 'user_transcript' ? 'You' : 'Assistant';
      setLines((prev) => [...prev, `${who}: ${data.payload.text}`]);
    } catch (_) {}
  });
  return (
    <SafeAreaView style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontSize: 18, marginBottom: 8 }}>Listening… speak now</Text>
      <ScrollView>
        {lines.map((l, i) => (
          <Text key={i} style={{ marginVertical: 4 }}>{l}</Text>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}
