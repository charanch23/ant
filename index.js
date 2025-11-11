/**
 * index.js
 * Robust video backend:
 * - tries HF video models via router
 * - fallback: generates N image frames (HF image model) and assembles a GIF server-side
 *
 * Env:
 * PROVIDER=huggingface
 * HUGGINGFACE_API_TOKEN=hf_...
 * PORT=4000
 */

require('dotenv').config();
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const GIFEncoder = require('gifencoder');
const Jimp = require('jimp');

const app = express();
app.use(helmet());
app.use(express.json({ limit: '30mb' }));

const FRONTEND_ORIGIN = process.env.FRONTEND_ORIGIN || '*';
app.use(cors({ origin: FRONTEND_ORIGIN }));

app.use(rateLimit({
  windowMs: 60 * 1000,
  max: 20,
  message: { error: 'Too many requests, please slow down.' }
}));

const PORT = process.env.PORT || 4000;
const PROVIDER = (process.env.PROVIDER || 'huggingface').toLowerCase();
const HF_TOKEN = process.env.HUGGINGFACE_API_TOKEN;
const HF_ROUTER = 'https://router.huggingface.co/hf-inference/models';

// Candidate video models (router) to try first (order matters)
const VIDEO_MODELS = [
  'ali-vilab/text-to-video-ms-1.7b',
  'damo-vilab/text-to-video',
  'stabilityai/stable-video-diffusion-img2vid-xt',
  'pesser/stable-video-diffusion'
];

// Fallback image model (known working on router)
const IMAGE_MODEL = 'black-forest-labs/FLUX.1-schnell';

// Utility: call HF router for a model, return arraybuffer and content-type
async function hfPostModel(model, payload, responseType = 'arraybuffer', timeout = 120000) {
  const url = `${HF_ROUTER}/${encodeURIComponent(model)}`;
  const resp = await axios.post(url, payload, {
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      Accept: 'application/json'
    },
    responseType,
    timeout
  });
  return resp;
}

// Try video models in order; if one returns video bytes or mp4 URL, return { model, videos }
async function tryVideoModels(prompt, size) {
  for (const model of VIDEO_MODELS) {
    try {
      const payload = {
        inputs: prompt,
        options: { wait_for_model: true },
        parameters: {}
      };
      // attempt size mapping (some models accept width/height)
      if (size && size.includes('x')) {
        const [w,h] = size.split('x').map(s => parseInt(s,10));
        if (!isNaN(w) && !isNaN(h)) {
          payload.parameters.width = w;
          payload.parameters.height = h;
        }
      }

      const resp = await hfPostModel(model, payload, 'arraybuffer', 180000);
      const contentType = (resp.headers['content-type'] || '').toLowerCase();

      // If response is raw video bytes (e.g., video/mp4), return data URL
      if (contentType.startsWith('video/')) {
        const base64 = Buffer.from(resp.data).toString('base64');
        return { model, videos: [`data:${contentType};base64,${base64}`] };
      }

      // If JSON or other, attempt to parse and extract mp4 URLs or base64 strings
      const text = Buffer.from(resp.data).toString('utf8');
      let parsed = null;
      try { parsed = JSON.parse(text); } catch (e) { parsed = text; }

      // search for mp4 urls or data:video entries
      const found = [];
      (function find(o){
        if (!o) return;
        if (typeof o === 'string') {
          if (o.startsWith('data:video')) found.push(o);
          else if (/\.mp4(\?|$)/.test(o)) found.push(o);
          else if (/^[A-Za-z0-9+/=]{200,}$/.test(o)) found.push('data:video/mp4;base64,' + o);
        } else if (Array.isArray(o)) o.forEach(find);
        else if (typeof o === 'object') Object.values(o).forEach(find);
      })(parsed);

      if (found.length) return { model, videos: found };
      // if nothing found, treat as failure and continue to next model
    } catch (err) {
      // If 401/403, return fatal auth error
      if (err.response && (err.response.status === 401 || err.response.status === 403)) {
        const body = err.response.data ? Buffer.from(err.response.data).toString('utf8') : '';
        throw new Error(`Auth error calling model ${model}: status=${err.response.status} body=${body}`);
      }
      // otherwise log and continue
      console.warn(`Model ${model} failed:`, err.response ? `${err.response.status} ${String(err.response.data).slice(0,200)}` : err.message);
      continue;
    }
  }
  // none succeeded
  return null;
}

// Fallback: call image model to produce a single image (data:image...)
async function callImageModel(prompt, size = '512x512') {
  const payload = {
    inputs: prompt,
    options: { wait_for_model: true },
    parameters: {}
  };
  if (size && size.includes('x')) {
    const [w,h] = size.split('x').map(s => parseInt(s,10));
    if (!isNaN(w) && !isNaN(h)) {
      payload.parameters.width = w;
      payload.parameters.height = h;
    }
  }
  const resp = await hfPostModel(IMAGE_MODEL, payload, 'arraybuffer', 120000);
  const ct = (resp.headers['content-type'] || '').toLowerCase();
  if (ct.startsWith('image/')) {
    const dataUrl = `data:${ct};base64,${Buffer.from(resp.data).toString('base64')}`;
    return dataUrl;
  }
  const text = Buffer.from(resp.data).toString('utf8');
  try {
    const parsed = JSON.parse(text);
    // search parsed for base64 or data:image
    let found = null;
    (function find(o){
      if (found) return;
      if (!o) return;
      if (typeof o === 'string') {
        if (o.startsWith('data:image')) found = o;
        else if (/^[A-Za-z0-9+/=]{100,}$/.test(o)) found = 'data:image/png;base64,' + o;
      } else if (Array.isArray(o)) for (const v of o) find(v);
      else if (typeof o === 'object') for (const v of Object.values(o)) find(v);
    })(parsed);
    if (found) return found;
  } catch (e) {
    // ignore
  }
  throw new Error('No image returned from image model');
}

// Assemble GIF from array of Data URLs (data:image/...) using Jimp + GIFEncoder
async function assembleGifFromDataUrls(dataUrls, delayMs = 120, quality = 10) {
  if (!Array.isArray(dataUrls) || dataUrls.length === 0) throw new Error('No frames to assemble');

  // Load Jimp images and ensure consistent size (use first as canonical)
  const jimpImgs = [];
  for (const d of dataUrls) {
    // support raw url (http) or data URL
    let buffer;
    if (d.startsWith('data:')) {
      const comma = d.indexOf(',');
      buffer = Buffer.from(d.slice(comma + 1), 'base64');
    } else {
      // fetch remote image
      const r = await axios.get(d, { responseType: 'arraybuffer', timeout: 120000 });
      buffer = Buffer.from(r.data);
    }
    const img = await Jimp.read(buffer);
    jimpImgs.push(img);
  }

  // Resize all frames to match first frame's size
  const width = jimpImgs[0].bitmap.width;
  const height = jimpImgs[0].bitmap.height;
  for (let i = 0; i < jimpImgs.length; i++) {
    if (jimpImgs[i].bitmap.width !== width || jimpImgs[i].bitmap.height !== height) {
      jimpImgs[i].resize(width, height);
    }
  }

  // Create GIF encoder and pipe to buffer
  const encoder = new GIFEncoder(width, height);
  encoder.setRepeat(0);   // 0 = loop forever
  encoder.setDelay(delayMs);
  encoder.setQuality(quality);

  // We'll collect encoder output via stream
  const stream = encoder.createReadStream();
  encoder.start();

  for (const img of jimpImgs) {
    // Jimp bitmap is in RGBA format as Buffer
    // GIFEncoder expects raw pixel data in RGBA order
    encoder.addFrame(img.bitmap.data);
  }
  encoder.finish();

  // collect stream into buffer
  const chunks = [];
  await new Promise((resolve, reject) => {
    stream.on('data', (c) => chunks.push(c));
    stream.on('end', resolve);
    stream.on('error', reject);
  });
  const outBuffer = Buffer.concat(chunks);
  const dataUrl = 'data:image/gif;base64,' + outBuffer.toString('base64');
  return dataUrl;
}

// Public endpoint: /api/generate-video
// Body: { prompt: string, size: '256x256'|'512x512', frames: number (fallback) }
// Response:
// - If HF video model used: { model, videos: [ dataUrlOrUrl, ... ] }
// - Else fallback GIF: { model: 'frames-gif-fallback', gif: 'data:image/gif;base64,...', frames: N }
app.post('/api/generate-video', async (req, res) => {
  try {
    if (PROVIDER !== 'huggingface') return res.status(400).json({ error: 'Unsupported provider' });
    if (!HF_TOKEN) return res.status(500).json({ error: 'HUGGINGFACE_API_TOKEN not configured' });

    const { prompt } = req.body || {};
    if (!prompt || !String(prompt).trim()) return res.status(400).json({ error: 'Missing prompt' });
    const size = String(req.body.size || '256x256');
    const framesRequested = Math.min(Math.max(parseInt(req.body.frames || 8, 10), 4), 16); // 4..16

    // 1) Try video models
    let videoResult = null;
    try {
      videoResult = await tryVideoModels(prompt, size);
    } catch (err) {
      // fatal auth etc -> return error
      console.error('Video model fatal error:', err.message || err);
      return res.status(500).json({ error: 'Video model error', detail: String(err.message || err) });
    }

    if (videoResult && videoResult.videos && videoResult.videos.length) {
      return res.json({ model: videoResult.model, videos: videoResult.videos });
    }

    // 2) Fallback: generate frames via image model
    const frames = [];
    for (let i = 0; i < framesRequested; i++) {
      // minor prompt variation so HF doesn't cache identical prompt (ensures variety)
      const variantPrompt = `${prompt} --frame ${i}`;
      const dataUrl = await callImageModel(variantPrompt, size);
      frames.push(dataUrl);
      // polite short delay not strictly required
      await new Promise(r => setTimeout(r, 300));
    }

    // 3) Assemble GIF on server
    const gifDataUrl = await assembleGifFromDataUrls(frames, 120, 10);

    return res.json({ model: 'frames-gif-fallback', gif: gifDataUrl, frames: frames.length });
  } catch (err) {
    console.error('Generate-video error:', err?.response?.data || err.message || err);
    let message = err.message || 'Generation failed';
    if (err.response && err.response.data) {
      try {
        const txt = Buffer.from(err.response.data).toString('utf8');
        message = txt;
      } catch (e) {}
    }
    return res.status(500).json({ error: message });
  }
});

app.get('/', (req, res) => res.send('AI Video Backend (robust) — running'));
app.listen(PORT, () => console.log(`Server listening on port ${PORT} (provider=${PROVIDER})`));
