#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";

function parseArgs(argv) {
  const opts = {
    api: "http://127.0.0.1:8080",
    payload: "sample/nodejs/payload.local.json",
    output: "sample/nodejs/out/result.jpg",
    format: "jpg",
    width: 0,
    quality: 95,
    pollIntervalMs: 2000,
    timeoutMs: 10 * 60 * 1000,
    embedLocalImages: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    const val = argv[i + 1];
    if (!key.startsWith("--")) {
      continue;
    }
    if (val === undefined || val.startsWith("--")) {
      throw new Error(`Missing value for ${key}`);
    }
    i += 1;
    switch (key) {
      case "--api":
        opts.api = val;
        break;
      case "--payload":
        opts.payload = val;
        break;
      case "--output":
        opts.output = val;
        break;
      case "--format":
        opts.format = val.toLowerCase();
        break;
      case "--width":
        opts.width = Number.parseInt(val, 10);
        break;
      case "--quality":
        opts.quality = Number.parseInt(val, 10);
        break;
      case "--poll-interval-ms":
        opts.pollIntervalMs = Number.parseInt(val, 10);
        break;
      case "--timeout-ms":
        opts.timeoutMs = Number.parseInt(val, 10);
        break;
      case "--embed-local-images":
        opts.embedLocalImages = parseBool(val, "--embed-local-images");
        break;
      default:
        throw new Error(`Unknown option: ${key}`);
    }
  }

  if (!["jpg", "jpeg", "avif"].includes(opts.format)) {
    throw new Error("format must be one of: jpg, jpeg, avif");
  }
  if (!Number.isFinite(opts.width) || opts.width < 0) {
    throw new Error("width must be >= 0");
  }
  if (!Number.isFinite(opts.quality) || opts.quality < 0 || opts.quality > 100) {
    throw new Error("quality must be between 0 and 100");
  }
  if (!Number.isFinite(opts.pollIntervalMs) || opts.pollIntervalMs < 200) {
    throw new Error("poll-interval-ms must be >= 200");
  }
  if (!Number.isFinite(opts.timeoutMs) || opts.timeoutMs < 1000) {
    throw new Error("timeout-ms must be >= 1000");
  }

  return opts;
}

function parseBool(value, keyName) {
  const v = String(value).trim().toLowerCase();
  if (["1", "true", "yes", "y", "on"].includes(v)) {
    return true;
  }
  if (["0", "false", "no", "n", "off"].includes(v)) {
    return false;
  }
  throw new Error(`${keyName} must be true/false`);
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function readJson(filePath) {
  const raw = await fs.readFile(filePath, "utf8");
  return JSON.parse(raw);
}

async function requestJson(url, init) {
  const resp = await fetch(url, init);
  const text = await resp.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  return { ok: resp.ok, status: resp.status, body };
}

async function downloadBinary(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    const errBody = await resp.text();
    throw new Error(`Download failed (${resp.status}): ${errBody}`);
  }
  const buffer = Buffer.from(await resp.arrayBuffer());
  return buffer;
}

function buildImageURL(base, opts) {
  const url = new URL(base);
  const format = opts.format === "jpeg" ? "jpg" : opts.format;
  url.searchParams.set("format", format);
  url.searchParams.set("quality", String(opts.quality));
  if (opts.width > 0) {
    url.searchParams.set("width", String(opts.width));
  }
  return url.toString();
}

function collectLocalImagePaths(payload) {
  const paths = [];
  if (Array.isArray(payload.image_paths)) {
    for (const item of payload.image_paths) {
      const text = String(item || "").trim();
      if (text) {
        paths.push(text);
      }
    }
  }
  if (typeof payload.image_path === "string" && payload.image_path.trim()) {
    paths.push(payload.image_path.trim());
  }
  return paths;
}

async function buildSubmitPayload(payload, opts) {
  const out = JSON.parse(JSON.stringify(payload ?? {}));
  if (!opts.embedLocalImages) {
    return out;
  }

  const localPaths = collectLocalImagePaths(out);
  if (localPaths.length === 0) {
    return out;
  }

  const imageFiles = [];
  for (const imagePath of localPaths) {
    const fileBuf = await fs.readFile(imagePath);
    imageFiles.push({
      filename: path.basename(imagePath),
      content_base64: fileBuf.toString("base64"),
    });
  }

  const existing = Array.isArray(out.image_files) ? out.image_files : [];
  out.image_files = [...existing, ...imageFiles];
  out.image_paths = [];
  delete out.image_path;
  return out;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const payload = await readJson(opts.payload);
  const submitPayload = await buildSubmitPayload(payload, opts);

  const submitURL = `${opts.api.replace(/\/+$/, "")}/thumbnail`;
  const enqueue = await requestJson(submitURL, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(submitPayload),
  });

  if (!enqueue.ok) {
    throw new Error(`Submit failed (${enqueue.status}): ${JSON.stringify(enqueue.body)}`);
  }
  const enqueueBody = enqueue.body || {};
  const jobID = enqueueBody.job_id;
  const jobURL = enqueueBody.job_url || `${opts.api.replace(/\/+$/, "")}/job/${jobID}`;
  if (!jobID) {
    throw new Error(`Invalid enqueue response: ${JSON.stringify(enqueueBody)}`);
  }

  console.log(`[ENQUEUE] job_id=${jobID} queue_position=${enqueueBody.queue_position}`);
  console.log(`[ENQUEUE] job_url=${jobURL}`);
  if (opts.embedLocalImages) {
    console.log("[ENQUEUE] local image paths embedded as image_files");
  }

  const started = Date.now();
  let lastStatus = "";
  let statusBody = null;
  while (Date.now() - started <= opts.timeoutMs) {
    const statusResp = await requestJson(jobURL, { method: "GET" });
    if (!statusResp.ok) {
      throw new Error(`Status failed (${statusResp.status}): ${JSON.stringify(statusResp.body)}`);
    }
    statusBody = statusResp.body || {};
    const status = statusBody.status || "unknown";
    if (status !== lastStatus) {
      console.log(`[STATUS] ${status}`);
      lastStatus = status;
    }

    if (status === "done") {
      break;
    }
    if (status === "failed") {
      throw new Error(`Job failed: ${JSON.stringify(statusBody)}`);
    }
    await sleep(opts.pollIntervalMs);
  }

  if (!statusBody || statusBody.status !== "done") {
    throw new Error(`Timeout waiting for job completion after ${opts.timeoutMs}ms`);
  }

  const imageBaseURL = statusBody.image_url || `${opts.api.replace(/\/+$/, "")}/job/${jobID}/image`;
  const imageURL = buildImageURL(imageBaseURL, opts);
  const fileBuffer = await downloadBinary(imageURL);

  await fs.mkdir(path.dirname(opts.output), { recursive: true });
  await fs.writeFile(opts.output, fileBuffer);

  const statusOut = `${opts.output}.status.json`;
  await fs.writeFile(statusOut, JSON.stringify(statusBody, null, 2), "utf8");

  console.log(`[DONE] image=${opts.output}`);
  console.log(`[DONE] status_json=${statusOut}`);
  console.log(`[DONE] image_url=${imageURL}`);
}

main().catch((err) => {
  console.error(`[ERROR] ${err.message}`);
  process.exit(1);
});
