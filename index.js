// server.js
require('dotenv').config();
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const PORT = process.env.PORT || 3000;
const HF_API_KEY = process.env.HUGGINGFACE_API_KEY;
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || 'facebook/text-to-video-zero-shot'; // replace with your chosen model ID

if (!HF_API_KEY) {
  console.error('ERROR: HUGGINGFACE_API_KEY is not set in .env');
  process.exit(1);
}

app.use(cors());
app.use(bodyParser.json({ limit: '5mb' })); // prompt payloads
app.use(express.static('public')); // serve frontend from /public

// Endpoint: POST /generate
// Body: { prompt: string, model?: string, options?: {} }
app.post('/generate', async (req, res) => {
  try {
    const { prompt, model, options } = req.body;
    if (!prompt || typeof prompt !== 'string') {
      return res.status(400).json({ error: 'Missing prompt string in request body.' });
    }

    const modelId = model || DEFAULT_MODEL;
    const hfUrl = `https://api-inference.huggingface.co/models/${encodeURIComponent(modelId)}`;

    // Build payload based on common HF Inference expectations
    // Many text-to-video models accept { inputs: prompt, options: {...} } or plain prompt
    const payload = { inputs: prompt };
    if (options && typeof options === 'object') payload.parameters = options;

    // Make the POST to Hugging Face Inference API
    const hfResponse = await axios({
      method: 'post',
      url: hfUrl,
      headers: {
        Authorization: `Bearer ${HF_API_KEY}`,
        Accept: '*/*',
        // Content-Type will be application/json by default with axios
      },
      data: payload,
      responseType: 'arraybuffer', // try to capture binary if model returns video bytes
      timeout: 10 * 60 * 1000, // long timeout (10 min) — video generation can take time
    });

    const contentType = hfResponse.headers['content-type'] || '';

    // If Hugging Face returned JSON (application/json), parse and return JSON
    if (contentType.includes('application/json')) {
      // parse buffer to JSON
      const text = Buffer.from(hfResponse.data).toString('utf8');
      let json;
      try { json = JSON.parse(text); } catch (err) { json = { raw: text }; }
      return res.json({ from: 'huggingface', model: modelId, result: json });
    }

    // If response is binary (video), forward bytes to client as stream:
    // Set content-type and send buffer
    // We try to preserve content-type if provided (e.g., video/mp4)
    const ct = contentType || 'application/octet-stream';
    res.setHeader('Content-Type', ct);
    // Suggest filename (frontend can download or play)
    res.setHeader('Content-Disposition', 'inline; filename="output"');

    return res.send(Buffer.from(hfResponse.data));
  } catch (err) {
    console.error('Generate error:', err.message || err);
    // Attempt to extract JSON error body if present
    if (err.response && err.response.data) {
      try {
        const bodyText = Buffer.from(err.response.data).toString('utf8');
        return res.status(err.response.status || 500).json({ error: bodyText });
      } catch (e) {
        // continue to generic
      }
    }
    return res.status(500).json({ error: 'Server error while talking to Hugging Face.' });
  }
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
  console.log(`Default model: ${DEFAULT_MODEL}`);
});
