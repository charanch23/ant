/**
 * index.js - Complete Video Backend
 * Routes:
 *  - POST /api/generate-video     => tries HF video models, fallback -> image frames -> server-side GIF
 *  - POST /api/generate-frames    => returns array of data:image frames (calls HF image model)
 *  - POST /api/generate           => compatibility forwarder to /api/generate-video
 *  - GET  /                       => health
 *
 * Env:
 * PROVIDER=huggingface
 * HUGGINGFACE_API_TOKEN=hf_...
 * PORT=4000
 *
 * NOTE: Keep HUGGINGFACE_API_TOKEN secret. Do not commit .env to git.
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
  max: 30,
  message: { error: 'Too many requests, please slow down.' }
}));

const PORT = process.env.PORT || 4000;
const PROVIDER = (process.env.PROVIDER || 'huggingface').toLowerCase();
const HF_TOKEN = process.env.HUGGINGFACE_API_TOKEN;
const HF_ROUTER = 'https://router.huggingface.co/hf-inference/models';

// VIDEO models to try (order)
const VIDEO_MODELS = [
  'ali-vilab/text-to-video-ms-1.7b',
  'damo-vilab/text-to-video',
  'stabilityai/stable-video-diffusion-img2vid-xt',
  'pesser/stable-video-diffusion'
];

// Fallback image model
const IMAGE_MODEL = 'black-forest-labs/FLUX.1-schnell';

// ------------------ Helpers ------------------

async function hfPostModel(model, payload, responseType = 'arraybuffer', timeout = 120000) {
  const url = `${HF_ROUTER}/${encodeURIComponent(model)}`;
  const resp = await axios.post(url, payload, {
    headers: { Authorization: `Bearer ${HF_TOKEN}`, Accept: 'application/json' },
    responseType,
    timeout
  });
  return resp;
}

async function tryVideoModels(prompt, size) {
  for (const model of VIDEO_MODELS) {
    try {
      const payload = {
        inputs: prompt,
        options: { wait_for_model: true },
        parameters: {}
      };
      if (size && size.includes('x')) {
        const [w,h] = size.split('x').map(s=>parseInt(s,10));
        if (!isNaN(w) && !isNaN(h)) {
          payload.parameters.width = w;
          payload.parameters.height = h;
        }
      }
      const resp = await hfPostModel(model, payload, 'arraybuffer', 180000);
      const ct = (resp.headers['content-type'] || '').toLowerCase();

      if (ct.startsWith('video/')) {
        const b64 = Buffer.from(resp.data).toString('base64');
        return { model, videos: [`data:${ct};base64,${b64}`] };
      }

      // parse JSON/text and extract mp4 urls or data:video
      const text = Buffer.from(resp.data).toString('utf8');
      let parsed;
      try { parsed = JSON.parse(text); } catch (e) { parsed = text; }
      const found = [];
      (function find(o){
        if (!o) return;
        if (typeof o === 'string') {
          if (o.startsWith('data:video')) found.push(o);
          else if (/\.mp4(\?|$)/.test(o)) found.push(o);
          else if (/^[A-Za-z0-9+/=]{200,}$/.test(o)) found.push('data:video/mp4;base64,'+o);
        } else if (Array.isArray(o)) o.forEach(find);
        else if (typeof o === 'object') Object.values(o).forEach(find);
      })(parsed);
      if (found.length) return { model, videos: found };
    } catch (err) {
      if (err.response && (err.response.status === 401 || err.response.status === 403)) {
        const body = err.response.data ? Buffer.from(err.response.data).toString('utf8') : '';
        throw new Error(`Auth error calling model ${model}: status=${err.response.status} body=${body}`);
      }
      console.warn(`Model ${model} failed:`, err.response ? `${err.response.status} ${String(err.response.data).slice(0,200)}` : err.message);
      continue;
    }
  }
  return null;
}

async function callImageModel(prompt, size='512x512') {
  const payload = { inputs: prompt, options: { wait_for_model: true }, parameters: {} };
  if (size && size.includes('x')) {
    const [w,h] = size.split('x').map(s => parseInt(s,10));
    if (!isNaN(w) && !isNaN(h)) { payload.parameters.width = w; payload.parameters.height = h; }
  }
  const resp = await hfPostModel(IMAGE_MODEL, payload, 'arraybuffer', 120000);
  const ct = (resp.headers['content-type'] || '').toLowerCase();
  if (ct.startsWith('image/')) {
    return `data:${ct};base64,${Buffer.from(resp.data).toString('base64')}`;
  }
  const text = Buffer.from(resp.data).toString('utf8');
  try {
    const parsed = JSON.parse(text);
    let found = null;
    (function find(o){
      if (found) return;
      if (!o) return;
      if (typeof o === 'string') {
        if (o.startsWith('data:image')) found = o;
        else if (/^[A-Za-z0-9+/=]{100,}$/.test(o)) found = 'data:image/png;base64,'+o;
      } else if (Array.isArray(o)) for (const v of o) find(v);
      else if (typeof o === 'object') for (const v of Object.values(o)) find(v);
    })(parsed);
    if (found) return found;
  } catch (e) {}
  throw new Error('No image returned from image model');
}

async function assembleGifFromDataUrls(dataUrls, delayMs = 120, quality = 10) {
  if (!Array.isArray(dataUrls) || dataUrls.length === 0) throw new Error('No frames to assemble');
  const jimpImgs = [];
  for (const d of dataUrls) {
    let buffer;
    if (d.startsWith('data:')) {
      const comma = d.indexOf(',');
      buffer = Buffer.from(d.slice(comma+1), 'base64');
    } else {
      const r = await axios.get(d, { responseType: 'arraybuffer', timeout: 120000 });
      buffer = Buffer.from(r.data);
    }
    const img = await Jimp.read(buffer);
    jimpImgs.push(img);
  }
  const width = jimpImgs[0].bitmap.width;
  const height = jimpImgs[0].bitmap.height;
  for (let i=0;i<jimpImgs.length;i++) {
    if (jimpImgs[i].bitmap.width !== width || jimpImgs[i].bitmap.height !== height) {
      jimpImgs[i].resize(width, height);
    }
  }
  const encoder = new GIFEncoder(width, height);
  encoder.setRepeat(0);
  encoder.setDelay(delayMs);
  encoder.setQuality(quality);
  const stream = encoder.createReadStream();
  encoder.start();
  for (const img of jimpImgs) {
    encoder.addFrame(img.bitmap.data);
  }
  encoder.finish();
  const chunks = [];
  await new Promise((resolve, reject) => {
    stream.on('data', c => chunks.push(c));
    stream.on('end', resolve);
    stream.on('error', reject);
  });
  const outBuffer = Buffer.concat(chunks);
  return 'data:image/gif;base64,' + outBuffer.toString('base64');
}

// ------------------ Routes ------------------

// Health
app.get('/', (req, res) => res.send('AI Video Backend (complete) — running'));

// Primary: generate-video
app.post('/api/generate-video', async (req, res) => {
  try {
    if (PROVIDER !== 'huggingface') return res.status(400).json({ error: 'Unsupported provider' });
    if (!HF_TOKEN) return res.status(500).json({ error: 'HUGGINGFACE_API_TOKEN not configured' });

    const { prompt } = req.body || {};
    if (!prompt || !String(prompt).trim()) return res.status(400).json({ error: 'Missing prompt' });

    const size = String(req.body.size || '256x256');
    const framesRequested = Math.min(Math.max(parseInt(req.body.frames || 8, 10), 4), 16);

    // Try video models first
    let v = null;
    try { v = await tryVideoModels(prompt, size); } catch (err) {
      console.error('Video model fatal error:', err.message || err);
      return res.status(500).json({ error: 'Video model error', detail: String(err.message || err) });
    }
    if (v && v.videos && v.videos.length) {
      return res.json({ model: v.model, videos: v.videos });
    }

    // Fallback: generate frames via image model
    const frames = [];
    for (let i=0;i<framesRequested;i++) {
      const variantPrompt = `${prompt} --frame ${i}`;
      const dataUrl = await callImageModel(variantPrompt, size);
      frames.push(dataUrl);
      await new Promise(r => setTimeout(r, 250));
    }

    // Assemble GIF
    const gifDataUrl = await assembleGifFromDataUrls(frames, 120, 10);
    return res.json({ model: 'frames-gif-fallback', gif: gifDataUrl, frames: frames.length });
  } catch (err) {
    console.error('Generate-video error:', err?.response?.data || err.message || err);
    let message = err.message || 'Generation failed';
    if (err.response && err.response.data) {
      try { message = Buffer.from(err.response.data).toString('utf8'); } catch(e) {}
    }
    return res.status(500).json({ error: message });
  }
});

// Frames: generate multiple frames and return base64 images
app.post('/api/generate-frames', async (req, res) => {
  try {
    if (PROVIDER !== 'huggingface') return res.status(400).json({ error: 'Unsupported provider' });
    if (!HF_TOKEN) return res.status(500).json({ error: 'HUGGINGFACE_API_TOKEN not configured' });

    const { prompt } = req.body || {};
    if (!prompt || !String(prompt).trim()) return res.status(400).json({ error: 'Missing prompt' });

    const framesCount = Math.min(Math.max(parseInt(req.body.frames || 8, 10), 4), 24);
    const size = String(req.body.size || '512x512');

    const frames = [];
    for (let i=0;i<framesCount;i++) {
      const variant = `${prompt} --frame ${i}`;
      const image = await callImageModel(variant, size);
      frames.push(image);
      await new Promise(r => setTimeout(r, 200));
    }
    return res.json({ frames });
  } catch (err) {
    console.error('Generate-frames error:', err?.response?.data || err.message || err);
    let msg = err.message || 'Frames generation failed';
    if (err.response && err.response.data) {
      try { msg = Buffer.from(err.response.data).toString('utf8'); } catch(e){}
    }
    return res.status(500).json({ error: msg });
  }
});

// Compatibility forwarder: /api/generate -> /api/generate-video
const axiosLocal = axios.create({ timeout: 300000 });
app.post('/api/generate', async (req, res) => {
  try {
    // Forward to local generate-video route
    const resp = await axiosLocal.post(`http://127.0.0.1:${PORT}/api/generate-video`, req.body, {
      headers: { 'Content-Type': 'application/json' }
    });
    return res.status(resp.status).json(resp.data);
  } catch (err) {
    console.error('Forward /api/generate failed:', err?.response?.data || err.message || err);
    const status = err.response?.status || 500;
    const body = err.response?.data || err.message || 'Forward failed';
    return res.status(status).json({ error: body });
  }
});

// Start
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT} (provider=${PROVIDER})`);
});
