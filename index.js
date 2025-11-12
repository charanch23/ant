/**
 * index.js - Hugging Face continuous video backend
 *
 * - Tries router video models (configured list or defaults).
 * - If router fails, optionally calls Hugging Face Spaces listed in HF_VIDEO_SPACES env var.
 * - Returns JSON: { model: <model-or-space>, videos: [url-or-dataUrl], note: ... }
 *
 * IMPORTANT:
 * - Some HF video models are not exposed via the router and only run as Spaces.
 * - For Spaces, the "run" API endpoint is: https://hf.space/run/{space}/api/predict
 *   But each Space expects different input shapes. You may need to inspect the Space's page
 *   to see the expected payload.
 *
 * ENV:
 *   HUGGINGFACE_API_TOKEN (required for router calls)
 *   HF_VIDEO_MODELS (optional comma-separated router model slugs)
 *   HF_VIDEO_SPACES (optional comma-separated space slugs to try)
 *   PORT (optional)
 *
 * Example request:
 * POST /api/generate-video
 * { "prompt": "a slow cinematic drone flight over a tropical island, golden hour", "duration": 5 }
 */

require('dotenv').config();
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

const app = express();
app.use(helmet());
app.use(express.json({ limit: '25mb' }));
app.use(cors({ origin: process.env.FRONTEND_ORIGIN || '*' }));
app.use(rateLimit({ windowMs: 60*1000, max: 30 }));

const PORT = process.env.PORT || 4000;
const HF_TOKEN = process.env.HUGGINGFACE_API_TOKEN;
const HF_ROUTER_BASE = 'https://router.huggingface.co/hf-inference/models';

// Defaults: router video models to try (order matters). You can override with HF_VIDEO_MODELS env var.
const DEFAULT_ROUTER_MODELS = [
  'ali-vilab/text-to-video-ms-1.7b',           // common
  'damo-vilab/text-to-video',                 // community
  'stabilityai/stable-video-diffusion-img2vid-xt', // try stability video version
  'runwayml/stable-video-v1'                  // example (may not be available via router)
];

// Helper: get router model list from env or defaults
function getRouterModelList() {
  const envList = (process.env.HF_VIDEO_MODELS || '').trim();
  if (!envList) return DEFAULT_ROUTER_MODELS;
  return envList.split(',').map(s => s.trim()).filter(Boolean);
}

// Helper: get spaces list from env (space slugs like owner/space-name)
function getSpacesList() {
  const envList = (process.env.HF_VIDEO_SPACES || '').trim();
  if (!envList) return [];
  return envList.split(',').map(s => s.trim()).filter(Boolean);
}

// POST to HF router model
async function callRouterModel(modelSlug, payload, timeout = 180000) {
  if (!HF_TOKEN) throw new Error('HUGGINGFACE_API_TOKEN not configured');
  const url = `${HF_ROUTER_BASE}/${encodeURIComponent(modelSlug)}`;
  const resp = await axios.post(url, payload, {
    headers: { Authorization: `Bearer ${HF_TOKEN}`, Accept: 'application/json' },
    responseType: 'arraybuffer',
    timeout
  });
  return resp;
}

// Try router video models in order. Return { model, videos: [...] } on success or null if none.
async function tryRouterVideos(prompt, durationSec, fps = 12) {
  const models = getRouterModelList();
  for (const m of models) {
    try {
      // payload shape is generic — many video models accept prompt + duration + fps; models differ
      const payload = {
        inputs: prompt,
        options: { wait_for_model: true },
        parameters: { duration: durationSec, fps }
      };
      console.log(`Trying router model: ${m}`);
      const resp = await callRouterModel(m, payload, 240000);
      const ct = (resp.headers['content-type'] || '').toLowerCase();

      // If raw video bytes returned:
      if (ct.startsWith('video/')) {
        const b64 = Buffer.from(resp.data).toString('base64');
        const dataUrl = `data:${ct};base64,${b64}`;
        return { model: m, videos: [dataUrl], source: 'router' };
      }

      // If JSON/text (common), parse & try to extract mp4 url or data
      const text = Buffer.from(resp.data).toString('utf8');
      let parsed = null;
      try { parsed = JSON.parse(text); } catch (e) { parsed = text; }

      // Search parsed for mp4 urls or data:video strings
      const found = [];
      function findUrls(o) {
        if (!o) return;
        if (typeof o === 'string') {
          if (o.startsWith('data:video')) found.push(o);
          else if (/\.mp4(\?|$)/.test(o)) found.push(o);
          else if (/^[A-Za-z0-9+/=]{200,}$/.test(o)) found.push('data:video/mp4;base64,' + o);
        } else if (Array.isArray(o)) o.forEach(findUrls);
        else if (typeof o === 'object') Object.values(o).forEach(findUrls);
      }
      findUrls(parsed);

      if (found.length) {
        return { model: m, videos: found, source: 'router' };
      }

      // No obvious video output -> treat as failure for this model
      console.warn(`Router model ${m} returned no video-like data (content-type=${ct})`);
    } catch (err) {
      if (err.response) {
        console.warn(`Router model ${m} responded ${err.response.status}: ${String(err.response.data).slice(0,300)}`);
        // 404 means model not available via router — continue to next
      } else {
        console.warn(`Router call to ${m} failed:`, err.message || err);
      }
      continue;
    }
  }
  return null;
}

/**
 * Try calling a Hugging Face Space's run API:
 * POST https://hf.space/run/{space}/api/predict (or /api/predict/ or /api/predict/)
 * Many Spaces implement /api/predict and accept JSON like { data: [ ... ] }.
 *
 * Because Spaces vary wildly, we allow the caller to configure expected input shape via env.
 * By default we try sending { "data": [ prompt ] } and { "data": { "prompt": prompt } }.
 */
async function trySpaces(spaces, prompt, extra = {}) {
  for (const space of spaces) {
    try {
      const base = `https://hf.space/run/${space}`;
      const endpoints = ['/api/predict', '/api/predict/', '/api/predict']; // try canonical one
      console.log(`Trying Space: ${space}`);
      // Try common payload shapes
      const triedPayloads = [
        { data: [ prompt ] },
        { data: { prompt } },
        { inputs: prompt },
        { prompt },
        { data: [ prompt, extra ] }
      ];
      let lastErr = null;
      for (const payload of triedPayloads) {
        try {
          const resp = await axios.post(base + '/api/predict', payload, { timeout: 180000 });
          // sometimes response is JSON with `data` or `output`
          const out = resp.data;
          // look for URLs / base64
          const found = [];
          function find(o) {
            if (!o) return;
            if (typeof o === 'string') {
              if (o.startsWith('data:video') || o.startsWith('data:video')) found.push(o);
              else if (/\.mp4(\?|$)/.test(o)) found.push(o);
              else if (/^[A-Za-z0-9+/=]{200,}$/.test(o)) found.push('data:video/mp4;base64,' + o);
            } else if (Array.isArray(o)) o.forEach(find);
            else if (typeof o === 'object') Object.values(o).forEach(find);
          }
          find(out);
          if (found.length) return { model: space, videos: found, source: 'space' };
          // if out.data contains nested
          if (out && (out.data || out.output)) find(out.data || out.output);
          if (found.length) return { model: space, videos: found, source: 'space' };
          // otherwise continue trying other payloads
        } catch (e) {
          lastErr = e;
          // try next payload
        }
      }
      console.warn(`Space ${space} tried payloads but yielded no video-like output. Last err: ${lastErr?.message?.slice?.(0,200) || lastErr}`);
    } catch (err) {
      console.warn(`Space call failed for ${space}: ${err.message || err}`);
      continue;
    }
  }
  return null;
}

// Public endpoint
app.post('/api/generate-video', async (req, res) => {
  try {
    const { prompt } = req.body || {};
    if (!prompt || !String(prompt).trim()) return res.status(400).json({ error: 'Missing prompt' });

    const duration = Math.min(Math.max(parseFloat(req.body.duration || 3), 0.5), 12); // clamp 0.5..12s
    const fps = Math.min(Math.max(parseInt(req.body.fps || 12, 10), 6), 30);

    // 1) Try router models
    const routerResult = await tryRouterVideos(prompt, duration, fps);
    if (routerResult) {
      return res.json({ model: routerResult.model, source: routerResult.source, videos: routerResult.videos });
    }

    // 2) Try Spaces if configured
    const spaces = getSpacesList();
    if (spaces.length > 0) {
      const spaceResult = await trySpaces(spaces, prompt, { duration, fps });
      if (spaceResult) {
        return res.json({ model: spaceResult.model, source: spaceResult.source, videos: spaceResult.videos });
      }
    }

    // 3) If we reach here, no HF router/video or Spaces worked
    const msg = {
      error: 'No continuous video model available via Hugging Face router or configured Spaces.',
      note: 'Either add a working Space to HF_VIDEO_SPACES env var, or use Replicate/Fal.ai for text-to-video.',
      triedRouterModels: getRouterModelList(),
      triedSpaces: getSpacesList()
    };
    console.warn('No HF video model succeeded:', JSON.stringify(msg, null, 2));
    return res.status(404).json(msg);
  } catch (err) {
    console.error('generate-video error:', err?.response?.data || err.message || err);
    let message = err.message || 'Generation failed';
    if (err.response && err.response.data) {
      try { message = JSON.stringify(err.response.data).slice(0,2000); } catch(e){}
    }
    return res.status(500).json({ error: message });
  }
});

app.get('/', (req, res) => res.send('Hugging Face continuous video backend — running'));
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
