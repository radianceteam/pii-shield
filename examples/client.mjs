// Minimal sidecar client for a non-Python consumer (ClawBot, a Node gateway, ...).
//
// The contract is deliberately small: anonymize, call your model, deanonymize.
// The only rule that matters is the error handling — a 422 or 503 must abort the
// outbound call, never fall through to sending the raw text.

const BASE = process.env.PII_SHIELD_URL ?? "http://127.0.0.1:8099";
const TOKEN = process.env.PII_SHIELD_TOKEN;

const headers = {
  "content-type": "application/json",
  ...(TOKEN ? { authorization: `Bearer ${TOKEN}` } : {}),
};

export async function anonymize(text, sessionId = null) {
  const res = await fetch(`${BASE}/v1/anonymize`, {
    method: "POST",
    headers,
    body: JSON.stringify({ text, session_id: sessionId }),
  });
  if (res.status === 422) {
    const { detail } = await res.json();
    throw new Error(`blocked: ${detail.entities.join(", ")}`);
  }
  if (!res.ok) throw new Error(`pii-shield unavailable (${res.status}) — refusing to send`);
  return res.json();
}

export async function deanonymize(text, sessionId, consume = true) {
  const res = await fetch(`${BASE}/v1/deanonymize`, {
    method: "POST",
    headers,
    body: JSON.stringify({ text, session_id: sessionId, consume }),
  });
  if (!res.ok) throw new Error(`deanonymize failed (${res.status})`);
  return (await res.json()).text;
}

// Usage:
//   const safe = await anonymize("Thomas Müller anrufen: +49 30 12345678");
//   const answer = await callYourModel(safe.text);
//   const real = await deanonymize(answer, safe.session_id);
