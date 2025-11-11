/**
 * index.js - Hugging Face Video Backend (robust multi-model fallback)
 *
 * Set env:
 * PROVIDER=huggingface
 * HUGGINGFACE_API_TOKEN=hf_...
 * PORT=4000
 *
 * This file will attempt multiple video models in order until one succeeds.
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
app.use(cors({ origin: process.env.FRONTEND_ORIGIN || '*' }));

app.use(rateLimit({
  windowMs: 60 * 1000,
  max: 20,
  message: { error: 'Too many requests, slow down.' }
}));

const PORT = process.env.PORT || 4000;
const PROVIDER = (process.env.PROVIDER || 'huggingface').toLowerCase();
const HF_TOKEN = process.env.HUGGINGFACE_API_TOKEN;
const HF_ROUTER_BASE = 'https://router.huggingface.co/hf-inference/models';

// list of candidate video models to try (order matters)
const CANDIDATE_MODELS = [
  'ali-vilab/text-to-video-ms-1.7b',
  'damo-vilab/text-to-video',
  'stabilityai/stable-video-diffusion-img2vid-xt',
  'pesser/stable-video-diffusion'
];

if (PROVIDER !== 'huggingface') {
  console.warn('Provider not huggingface. Set PROVIDER=huggingface');
}
if (!HF_TOKEN) {
  console.warn('HUGGINGFACE_API_TOKEN missing in env.');
}

// call the HF router for a single model; returns { videos } or throws
async function callModelOnce(model, prompt, options = {}) {
  const url = `${HF_ROUTER_BASE}/${encodeURIComponent(model)}`;
  const payload = {
    inputs: prompt,
    options: { wait_for_model: true },
    parameters: {}
  };
  if (options.size) {
    const [w,h] = String(options.size).split('x').map(x => parseInt(x,10));
    if (!isNaN(w) && !isNaN(h)) {
      payload.parameters.width = w;
      payload.parameters.height = h;
    }
  }

  try {
    const resp = await axios.post(url, payload, {
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        Accept: 'application/json'
      },
      responseType: 'arraybuffer',
      timeout: 120000
    });

    const contentType = resp.headers['content-type'] || '';
    // video bytes
    if (contentType.startsWith('video/')) {
      const base64 = Buffer.from(resp.data).toString('base64');
      return { videos: [`data:${contentType};base64,${base64}`] };
    }

    // json-like response
    const text = Buffer.from(resp.data).toString('utf8');
    let parsed;
    try { parsed = JSON.parse(text); } catch (e) { parsed = text; }

    // extract data:video or mp4 urls
    const videos = [];
    function extract(obj) {
      if (!obj) return;
      if (typeof obj === 'string') {
        if (obj.startsWith('data:video')) videos.push(obj);
        else if (/\.mp4(\?|$)/.test(obj)) videos.push(obj);
        else if (/^[A-Za-z0-9+/=]{100,}$/.test(obj)) videos.push('data:video/mp4;base64,'+obj);
      } else if (Array.isArray(obj)) obj.forEach(extract);
      else if (typeof obj === 'object') Object.values(obj).forEach(extract);
    }
    extract(parsed);
    if (videos.length) return { videos };

    // if parsed contains something useful (fallback)
    return { raw: parsed };
  } catch (err) {
    // bubble up status and message for caller to interpret
    if (err.response) {
      const status = err.response.status;
      const body = err.response.data ? Buffer.from(err.response.data).toString('utf8') : '';
      const message = `Model ${model} error: status=${status} body=${body}`;
      const e = new Error(message);
      e.status = status;
      e.body = body;
      throw e;
    }
    throw err;
  }
}

// Try models in order until one succeeds
async function generateVideoWithFallback(models, prompt, options) {
  const errors = [];
  for (const model of models) {
    console.log(`Trying model: ${model}`);
    try {
      const out = await callModelOnce(model, prompt, options);
      console.log(`Model succeeded: ${model}`);
      // If out.videos exist return them
      if (out.videos && out.videos.length) return { model, videos: out.videos };
      // If raw JSON with mp4 urls, return them
      if (out.raw) {
        // try to find mp4 links inside raw
        const urls = [];
        (function findUrls(o){
          if (!o) return;
          if (typeof o === 'string' && /\.mp4(\?|$)/.test(o)) urls.push(o);
          else if (Array.isArray(o)) o.forEach(findUrls);
          else if (typeof o === 'object') Object.values(o).forEach(findUrls);
        })(out.raw);
        if (urls.length) return { model, videos: urls };
        // else return raw as debug
        return { model, raw: out.raw };
      }
      errors.push({ model, note: 'no videos in response' });
    } catch (err) {
      // record the error and continue on 404 / 403 etc.
      console.warn(`Model ${model} failed:`, err.message?.slice?.(0,300) || String(err));
      errors.push({ model, status: err.status || null, msg: err.message || String(err) });
      // If non-recoverable (like 401 unauthorized) stop early
      if (err.status === 401 || err.status === 403) {
        throw new Error(`Fatal auth error when calling model ${model}: ${err.message}`);
      }
      // otherwise continue to next model
    }
  }
  // all failed
  const e = new Error('All candidate models failed: ' + JSON.stringify(errors));
  e.details = errors;
  throw e;
}

app.post('/api/generate', async (req, res) => {
  try {
    if (PROVIDER !== 'huggingface') return res.status(400).json({ error: 'Unsupported provider' });
    if (!HF_TOKEN) return res.status(500).json({ error: 'HUGGINGFACE_API_TOKEN not configured' });

    const { prompt, size } = req.body || {};
    if (!prompt || !String(prompt).trim()) return res.status(400).json({ error: 'Missing prompt' });

    const allowed = ['256x256','512x512'];
    const finalSize = allowed.includes(String(size||'256x256')) ? size : '256x256';

    const result = await generateVideoWithFallback(CANDIDATE_MODELS, String(prompt), { size: finalSize });

    // success result may contain .videos or .raw
    if (result.videos && result.videos.length) {
      return res.json({ model: result.model, videos: result.videos });
    } else {
      return res.json({ model: result.model || null, raw: result.raw || null });
    }
  } catch (err) {
    console.error('Generate error:', err.message?.slice?.(0,1000) || err);
    // If we have structured details, include them for debugging (safe)
    const debug = err.details ? err.details : undefined;
    const msg = err.message || 'Generation failed';
    return res.status(500).json({ error: msg, debug });
  }
});

app.get('/', (req, res) => res.send('AI Video Backend (HF fallback) running'));
app.listen(PORT, () => console.log(`Server listening on port ${PORT} (provider=${PROVIDER})`));
