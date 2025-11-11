/**
 * index.js - Hugging Face Video Backend
 *
 * Usage:
 * - Set environment variables on Render:
 *     PROVIDER=huggingface
 *     HUGGINGFACE_API_TOKEN=hf_...
 *     PORT (optional)
 *
 * - Deploy to Render (no local run required).
 *
 * Endpoint:
 * POST /api/generate
 * Body: { prompt: "text prompt", size: "256x256" }
 * Response: { videos: [ "data:video/mp4;base64,...." ] }
 *
 * NOTE: This uses Hugging Face router endpoint:
 *   https://router.huggingface.co/hf-inference/models/{model}
 *
 * Model used: ali-vilab/text-to-video-ms-1.7b  (short clips ~2-3s)
 */

require('dotenv').config();
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

const app = express();
app.use(helmet());
app.use(express.json({ limit: '20mb' }));

const FRONTEND_ORIGIN = process.env.FRONTEND_ORIGIN || '*';
app.use(cors({ origin: FRONTEND_ORIGIN }));

const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 20,
  message: { error: 'Too many requests, please slow down.' }
});
app.use(limiter);

const PORT = process.env.PORT || 4000;
const PROVIDER = (process.env.PROVIDER || 'huggingface').toLowerCase();
const HF_TOKEN = process.env.HUGGINGFACE_API_TOKEN;

// Choose video model (text -> short video)
const HF_MODEL = 'ali-vilab/text-to-video-ms-1.7b';
const HF_ROUTER_BASE = 'https://router.huggingface.co/hf-inference/models';

if (PROVIDER !== 'huggingface') {
  console.warn('Configured provider is not huggingface. Set PROVIDER=huggingface in env.');
}
if (!HF_TOKEN) {
  console.warn('HUGGINGFACE_API_TOKEN not set in environment.');
}

// Helper: call HF router for the model
async function callHFRouter(model, prompt, options = {}) {
  const url = `${HF_ROUTER_BASE}/${encodeURIComponent(model)}`;
  // Many video models accept inputs: { inputs: prompt, options: { wait_for_model: true }, parameters: {...} }
  const payload = {
    inputs: prompt,
    options: { wait_for_model: true },
    parameters: {}
  };

  // You can pass additional parameters if model supports (e.g., num_inference_steps, width/height)
  if (options.size) {
    const [w, h] = String(options.size).split('x').map(x => parseInt(x, 10));
    if (!isNaN(w) && !isNaN(h)) {
      // Some models accept width/height; include if supported
      payload.parameters.width = w;
      payload.parameters.height = h;
    }
  }

  const resp = await axios.post(url, payload, {
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      Accept: 'application/json'
    },
    responseType: 'arraybuffer',
    timeout: 120000
  });

  const contentType = resp.headers['content-type'] || '';

  // If response is video bytes
  if (contentType.startsWith('video/')) {
    const base64 = Buffer.from(resp.data).toString('base64');
    const dataUrl = `data:${contentType};base64,${base64}`;
    return { videos: [dataUrl] };
  }

  // If response is JSON (some models return JSON with keys)
  if (contentType.includes('application/json') || contentType.includes('text/json')) {
    const text = Buffer.from(resp.data).toString('utf8');
    let parsed = null;
    try { parsed = JSON.parse(text); } catch (e) { parsed = text; }

    // Try to extract video bytes or data url inside parsed object
    // e.g., some HF responses contain base64 strings or urls
    const videos = [];
    function extract(obj) {
      if (!obj) return;
      if (typeof obj === 'string') {
        if (obj.startsWith('data:video')) videos.push(obj);
        // if it's a long base64-looking string, assume mp4
        else if (/^[A-Za-z0-9+/=]{100,}$/.test(obj)) videos.push('data:video/mp4;base64,' + obj);
      } else if (Array.isArray(obj)) {
        obj.forEach(extract);
      } else if (typeof obj === 'object') {
        Object.values(obj).forEach(extract);
      }
    }
    extract(parsed);
    if (videos.length) return { videos };
    // maybe model returned a URL to the video
    if (typeof parsed === 'object') {
      // try to find strings that look like .mp4 urls
      const urls = [];
      function findUrls(o) {
        if (!o) return;
        if (typeof o === 'string' && /\.mp4(\?|$)/.test(o)) urls.push(o);
        else if (Array.isArray(o)) o.forEach(findUrls);
        else if (typeof o === 'object') Object.values(o).forEach(findUrls);
      }
      findUrls(parsed);
      if (urls.length) return { videos: urls };
    }
    // fallback: return raw parsed object for debugging
    return { raw: parsed };
  }

  throw new Error('Unknown response format from Hugging Face router');
}

async function generateVideo({ prompt, size = '256x256' }) {
  if (!HF_TOKEN) throw new Error('HUGGINGFACE_API_TOKEN not configured');
  // The video models are usually slow; call once per request (n=1)
  const out = await callHFRouter(HF_MODEL, prompt, { size });
  // out should contain { videos: [...] } or { raw: ... }
  if (out.videos && out.videos.length) return out.videos;
  // If out.raw contains a URL or something, try returning that as single item
  if (out.raw) {
    // if parsed raw contains url to video, return that
    if (typeof out.raw === 'string') return [out.raw];
    // fallback to returning JSON in data URL so client can show text
    return ['data:application/json;base64,' + Buffer.from(JSON.stringify(out.raw)).toString('base64')];
  }
  throw new Error('No video produced by model');
}

// API endpoint
app.post('/api/generate', async (req, res) => {
  try {
    const { prompt, size } = req.body || {};
    if (!prompt || !String(prompt).trim()) return res.status(400).json({ error: 'Missing prompt' });

    if (PROVIDER !== 'huggingface') return res.status(400).json({ error: `Unsupported provider: ${PROVIDER}` });

    // safety: ensure small size for free models
    const allowedSizes = ['256x256', '512x512'];
    const imgSize = String(size || '256x256');
    const finalSize = allowedSizes.includes(imgSize) ? imgSize : '256x256';

    const videos = await generateVideo({ prompt: String(prompt), size: finalSize });
    return res.json({ videos });
  } catch (err) {
    let message = 'Generation failed';
    if (err.response && err.response.data) {
      try {
        // try to parse error body
        const text = Buffer.from(err.response.data).toString('utf8');
        const parsed = JSON.parse(text);
        message = parsed.error || JSON.stringify(parsed);
      } catch (e) {
        message = String(err.response.data).slice(0, 500);
      }
    } else if (err.message) {
      message = err.message;
    }
    console.error('Generate error:', message);
    return res.status(500).json({ error: message });
  }
});

app.get('/', (req, res) => res.send('AI Video Backend (Hugging Face) — running'));

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT} (provider=${PROVIDER})`);
});
