// server.js
// Simple Express proxy that forwards prompts to Hugging Face Inference API
// and returns either JSON (links/metadata) or binary video back to the browser.
//
// Important: set HUGGINGFACE_API_KEY in environment (or in a .env for local dev).
// Recommended: set HUGGINGFACE_API_KEY in Render environment variables (do NOT commit .env).

require('dotenv').config();
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');

const app = express();

// Port handling (Render provides PORT)
const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;

// Hugging Face key and default model
const HF_API_KEY = process.env.HUGGINGFACE_API_KEY || process.env.HF_API_KEY_FALLBACK || null;
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'stabilityai/stable-video-diffusion-img2vid'; // change if desired

// sanity checks
console.log('Starting server...');
console.log('PORT:', PORT);
console.log('DEFAULT_MODEL:', DEFAULT_MODEL);
console.log('HF_API_KEY set?', !!HF_API_KEY ? 'YES' : 'NO (will fail without key)');

app.use(cors());
app.use(bodyParser.json({ limit: '30mb' })); // allow larger payloads if needed
app.use(express.static(path.join(__dirname, 'public')));

// health route
app.get('/health', (req, res) => {
  res.json({ status: 'ok', time: new Date().toISOString() });
});

/**
 * POST /generate
 * Body: { prompt: string, model?: string, options?: object }
 *
 * Behavior:
 * - calls Hugging Face Inference API for the chosen model
 * - handles JSON responses (e.g., { error } or urls) and binary responses (video bytes)
 * - returns JSON when HF returns JSON; streams binary when HF returns binary
 */
app.post('/generate', async (req, res) => {
  try {
    const { prompt, model, options } = req.body;
    if (!prompt || typeof prompt !== 'string') {
      return res.status(400).json({ error: 'Missing required field: prompt (string).' });
    }
    if (!HF_API_KEY) {
      return res.status(500).json({ error: 'Server missing Hugging Face API key. Set HUGGINGFACE_API_KEY.' });
    }

    const modelId = (model && typeof model === 'string') ? model : DEFAULT_MODEL;
    const hfUrl = `https://api-inference.huggingface.co/models/${encodeURIComponent(modelId)}`;

    // Build payload widely accepted by HF inference endpoints
    const payload = { inputs: prompt };
    if (options && typeof options === 'object') {
      payload.parameters = options;
    }

    // Make request to Hugging Face Inference API
    const hfResp = await axios({
      method: 'post',
      url: hfUrl,
      headers: {
        Authorization: `Bearer ${HF_API_KEY}`,
        Accept: '*/*',
        'Content-Type': 'application/json'
      },
      data: payload,
      responseType: 'arraybuffer', // capture binary if returned
      timeout: 10 * 60 * 1000 // 10 min - adjust if you expect longer
    });

    const contentType = (hfResp.headers && hfResp.headers['content-type']) ? hfResp.headers['content-type'] : '';

    // If HF returned JSON text (application/json), parse it and return JSON to client
    if (contentType.includes('application/json') || contentType.includes('text/json')) {
      const text = Buffer.from(hfResp.data).toString('utf8');
      let json;
      try { json = JSON.parse(text); } catch (e) { json = { raw: text }; }

      // If HF returns an output URL inside JSON, forward it plainly
      return res.json({
        from: 'huggingface',
        model: modelId,
        result: json
      });
    }

    // If HF returned some other textual content (e.g. error as text), attempt to parse
    if (contentType.startsWith('text/')) {
      const text = Buffer.from(hfResp.data).toString('utf8');
      return res.json({ from: 'huggingface', model: modelId, text });
    }

    // Otherwise assume binary (video, gif, octet-stream). Forward bytes with preserved content-type.
    const ct = contentType || 'application/octet-stream';
    res.setHeader('Content-Type', ct);
    // inline so browser can play if video/mp4
    res.setHeader('Content-Disposition', 'inline; filename="output"');
    return res.send(Buffer.from(hfResp.data));
  } catch (err) {
    console.error('Error in /generate:', err?.response?.status, err?.message || err);

    // If HF responded with an error body, try to forward it
    if (err.response && err.response.data) {
      try {
        const text = Buffer.from(err.response.data).toString('utf8');
        // Try JSON parse
        try {
          const parsed = JSON.parse(text);
          return res.status(err.response.status || 500).json({ error: parsed });
        } catch (e) {
          return res.status(err.response.status || 500).json({ error: text });
        }
      } catch (e) {
        // fall through
      }
    }

    // Generic fallback
    return res.status(500).json({ error: err.message || 'Unknown server error' });
  }
});

// Fallback for SPA (serve index.html)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
