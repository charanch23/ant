/**
 * index.frames.js
 * Simple endpoint to generate multiple image frames (free) and return base64 images.
 *
 * Env:
 * PROVIDER=huggingface
 * HUGGINGFACE_API_TOKEN=hf_...
 * PORT=4000
 *
 * Endpoint:
 * POST /api/generate-frames
 * Body: { prompt: "text", size: "512x512", frames: 8 }
 * Response: { frames: [ "data:image/png;base64,..." ] }
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
  windowMs: 60*1000,
  max: 40,
  message: { error: 'Too many requests' }
}));

const PORT = process.env.PORT || 4000;
const HF_TOKEN = process.env.HUGGINGFACE_API_TOKEN;
const HF_ROUTER = 'https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell';

// Helper: call HF image model (returns data:image... or throws)
async function callHFImage(prompt, size='512x512') {
  if (!HF_TOKEN) throw new Error('HUGGINGFACE_API_TOKEN not configured');

  const payload = {
    inputs: prompt,
    options: { wait_for_model: true },
    parameters: {}
  };

  if (size && size.includes('x')) {
    const [w,h] = size.split('x').map(s=>parseInt(s.trim(),10));
    if (!isNaN(w) && !isNaN(h)) {
      payload.parameters.width = w;
      payload.parameters.height = h;
    }
  }

  const resp = await axios.post(HF_ROUTER, payload, {
    headers: {
      Authorization: `Bearer ${HF_TOKEN}`,
      Accept: 'application/json'
    },
    responseType: 'arraybuffer',
    timeout: 120000
  });

  const ct = (resp.headers['content-type']||'');
  if (ct.startsWith('image/')) {
    const b64 = Buffer.from(resp.data).toString('base64');
    return `data:${ct};base64,${b64}`;
  }

  // Some responses come as JSON with base64 or data urls
  const text = Buffer.from(resp.data).toString('utf8');
  try {
    const parsed = JSON.parse(text);
    // extract first base64/image-like string
    function findImgs(obj) {
      if (!obj) return null;
      if (typeof obj === 'string') {
        if (obj.startsWith('data:image')) return obj;
        if (/^[A-Za-z0-9+/=]{100,}$/.test(obj)) return 'data:image/png;base64,' + obj;
      } else if (Array.isArray(obj)) {
        for (const v of obj) {
          const r = findImgs(v);
          if (r) return r;
        }
      } else if (typeof obj === 'object') {
        for (const v of Object.values(obj)) {
          const r = findImgs(v);
          if (r) return r;
        }
      }
      return null;
    }
    const found = findImgs(parsed);
    if (found) return found;
  } catch (e) { /* ignore */ }

  throw new Error('Unknown response format from HF image model');
}

// POST /api/generate-frames
app.post('/api/generate-frames', async (req, res) => {
  try {
    const { prompt, size } = req.body || {};
    if (!prompt || !prompt.toString().trim()) return res.status(400).json({ error: 'Missing prompt' });

    const frameCount = Math.min(Math.max(parseInt(req.body.frames || 8, 10), 4), 24); // 4..24 frames
    const imgSize = size || '512x512';

    // We'll generate frames by appending a tiny variation to the prompt (so HF treats them as different)
    const frames = [];
    for (let i=0;i<frameCount;i++) {
      // Variation: add small non-intrusive token (seed-like)
      const variantPrompt = `${prompt} --frame=${i}`;
      const img = await callHFImage(variantPrompt, imgSize);
      frames.push(img);
      // small delay to be polite (optional)
      await new Promise(r => setTimeout(r, 300));
    }

    return res.json({ frames });
  } catch (err) {
    console.error('Frames error:', err.message || err);
    let msg = err.message || 'Generation failed';
    if (err.response && err.response.data) {
      try { msg = Buffer.from(err.response.data).toString('utf8'); } catch(e){}
    }
    return res.status(500).json({ error: msg });
  }
});

app.get('/', (req,res)=>res.send('HF frames backend running'));
app.listen(PORT, ()=>console.log(`Frames backend running on ${PORT}`));
