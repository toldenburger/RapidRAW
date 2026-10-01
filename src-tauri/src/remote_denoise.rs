// Proof-of-concept alternative to the local CPU-only AI denoise path in
// `ai_processing.rs`. Instead of running the NIND ONNX model locally one
// tile at a time, this dispatches small batches of tiles as concurrent jobs
// to a RunPod Serverless endpoint (GPU), and blends the results back exactly
// like the local tiling loop does.
//
// RunPod Serverless's job-queue API caps input payloads (10MB for /run), far
// below a full image's tile data (hundreds of MB), so tiles are chunked into
// small batches small enough to fit, sent as separate concurrent jobs, and
// reassembled by original tile order once all complete. This also gives
// real incremental progress, unlike sending everything in one request.
use anyhow::{Result, anyhow};
use base64::{Engine as _, engine::general_purpose};
use futures::stream::{FuturesUnordered, StreamExt};
use image::{DynamicImage, Rgb32FImage};
use reqwest::Client;
use serde_json::{Value, json};
use std::time::Duration;
use tauri::{AppHandle, Emitter};

use crate::ai_processing::{
    SeamlessBlend, TileParams, accumulator_to_rgb32f, apply_seamless, extract_tile_mirror,
    select_tile_params,
};

const CHANNELS: usize = 3;
// Each tile is CHANNELS*504*504*4 bytes raw (~2.9MB); base64 inflates that by
// ~1.33x. 2 tiles/job keeps a chunk's JSON body well under RunPod's 10MB
// /run limit.
const TILES_PER_JOB: usize = 2;
const POLL_INTERVAL: Duration = Duration::from_millis(800);

struct TilePlacement {
    x0: i32,
    y0: i32,
}

fn le_bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

async fn submit_and_wait(
    client: &Client,
    endpoint_url: &str,
    api_key: &str,
    tiles_b64: String,
    num_tiles: usize,
) -> Result<String> {
    let run_res = client
        .post(format!("{}/run", endpoint_url))
        .bearer_auth(api_key)
        .json(&json!({"input": {"tiles_b64": tiles_b64, "num_tiles": num_tiles}}))
        .send()
        .await?;
    if !run_res.status().is_success() {
        return Err(anyhow!("RunPod /run failed: {}", run_res.text().await?));
    }
    let run_body: Value = run_res.json().await?;
    let job_id = run_body["id"]
        .as_str()
        .ok_or_else(|| anyhow!("No job id in RunPod response: {}", run_body))?
        .to_string();

    loop {
        let status_res = client
            .get(format!("{}/status/{}", endpoint_url, job_id))
            .bearer_auth(api_key)
            .send()
            .await?;
        let status_body: Value = status_res.json().await?;
        let status = status_body["status"].as_str().unwrap_or("");
        match status {
            "COMPLETED" => {
                let tiles_b64_out = status_body["output"]["tiles_b64"]
                    .as_str()
                    .ok_or_else(|| {
                        anyhow!("No tiles_b64 in completed job output: {}", status_body)
                    })?
                    .to_string();
                return Ok(tiles_b64_out);
            }
            "FAILED" | "CANCELLED" | "TIMED_OUT" => {
                return Err(anyhow!(
                    "RunPod job {} ended with status {}: {}",
                    job_id,
                    status,
                    status_body
                ));
            }
            _ => {
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }
    }
}

pub async fn run_remote_denoise(
    rgb_img: &Rgb32FImage,
    intensity: f32,
    endpoint_url: &str,
    api_key: Option<&str>,
    app_handle: &AppHandle,
) -> Result<DynamicImage> {
    let api_key = api_key.ok_or_else(|| anyhow!("RunPod API key not configured"))?;
    let (width, height) = rgb_img.dimensions();
    let (width, height) = (width as usize, height as usize);
    let params: TileParams = select_tile_params(intensity);

    let step = params.ucs.saturating_sub(params.overlap).max(1);
    let iperhl = (width.saturating_sub(params.ucs) as f64 / step as f64).ceil() as usize;
    let ipervl = (height.saturating_sub(params.ucs) as f64 / step as f64).ceil() as usize;
    let total = (iperhl + 1) * (ipervl + 1);
    let tile_floats = CHANNELS * params.cs * params.cs;

    let mut placements = Vec::with_capacity(total);
    let mut tiles_raw: Vec<Vec<u8>> = Vec::with_capacity(total);

    for i in 0..total {
        let yi = i / (iperhl + 1);
        let xi = i % (iperhl + 1);
        let x0 =
            params.ucs as i32 * xi as i32 - params.overlap as i32 * xi as i32 - params.pad as i32;
        let y0 =
            params.ucs as i32 * yi as i32 - params.overlap as i32 * yi as i32 - params.pad as i32;

        let crop = extract_tile_mirror(rgb_img, x0, y0, params.cs);
        let standard = crop.as_standard_layout();
        let slice = standard
            .as_slice()
            .ok_or_else(|| anyhow!("Tile data was not contiguous"))?;
        let mut bytes = Vec::with_capacity(slice.len() * 4);
        for v in slice {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        tiles_raw.push(bytes);
        placements.push(TilePlacement { x0, y0 });
    }

    let client = Client::new();
    let num_chunks = total.div_ceil(TILES_PER_JOB);

    let _ = app_handle.emit(
        "denoise-progress",
        format!(
            "Dispatching {} tiles across {} remote jobs...",
            total, num_chunks
        ),
    );

    let mut futures_set = FuturesUnordered::new();
    for (chunk_idx, chunk) in tiles_raw.chunks(TILES_PER_JOB).enumerate() {
        let mut combined = Vec::with_capacity(chunk.iter().map(|c| c.len()).sum());
        for c in chunk {
            combined.extend_from_slice(c);
        }
        let tiles_b64 = general_purpose::STANDARD.encode(&combined);
        let num_tiles_in_chunk = chunk.len();
        let client = client.clone();
        let endpoint_url = endpoint_url.to_string();
        let api_key = api_key.to_string();
        futures_set.push(async move {
            let r = submit_and_wait(&client, &endpoint_url, &api_key, tiles_b64, num_tiles_in_chunk)
                .await;
            (chunk_idx, r)
        });
    }

    let mut chunk_results: Vec<Option<Vec<f32>>> = vec![None; num_chunks];
    let mut completed = 0usize;
    while let Some((chunk_idx, result)) = futures_set.next().await {
        let tiles_b64_out = result?;
        let decoded = general_purpose::STANDARD.decode(&tiles_b64_out)?;
        chunk_results[chunk_idx] = Some(le_bytes_to_f32_vec(&decoded));
        completed += 1;
        let pct = (completed as f32 / num_chunks as f32) * 100.0;
        let _ = app_handle.emit("denoise-progress", format!("Remote denoising… {:.0}%", pct));
    }

    let mut response_floats: Vec<f32> = Vec::with_capacity(total * tile_floats);
    for cr in chunk_results {
        response_floats.extend_from_slice(&cr.ok_or_else(|| anyhow!("Missing chunk result"))?);
    }

    if response_floats.len() != total * tile_floats {
        return Err(anyhow!(
            "Remote denoise returned {} floats, expected {} for {} tiles",
            response_floats.len(),
            total * tile_floats,
            total
        ));
    }

    let mut accumulator = vec![0.0f32; width * height * CHANNELS];

    for (i, placement) in placements.iter().enumerate() {
        let TilePlacement { x0, y0 } = *placement;
        let start = i * tile_floats;
        let tile_data = &response_floats[start..start + tile_floats];

        let x1pad = (0i32).max(x0 + params.cs as i32 - width as i32) as usize;
        let y1pad = (0i32).max(y0 + params.cs as i32 - height as i32) as usize;
        let ud0 = params.pad;
        let ud1 = params.pad;
        let ud2 = params.cs - params.pad.max(x1pad);
        let ud3 = params.cs - params.pad.max(y1pad);
        let absx0 = (x0 + params.pad as i32).max(0) as usize;
        let absy0 = (y0 + params.pad as i32).max(0) as usize;

        let mut tile = ndarray::Array4::from_shape_vec(
            (1, CHANNELS, params.cs, params.cs),
            tile_data.to_vec(),
        )?;

        apply_seamless(
            &mut tile,
            &SeamlessBlend {
                ud0,
                ud1,
                ud2,
                ud3,
                absx0,
                absy0,
                fswidth: width,
                fsheight: height,
                overlap: params.overlap,
            },
        );

        for cy in 0..(ud3 - ud1) {
            for cx in 0..(ud2 - ud0) {
                let gx = absx0 + cx;
                let gy = absy0 + cy;
                if gx < width && gy < height {
                    let base = (gy * width + gx) * CHANNELS;
                    accumulator[base] += tile[[0, 0, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                    accumulator[base + 1] += tile[[0, 1, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                    accumulator[base + 2] += tile[[0, 2, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                }
            }
        }
    }

    let out_img_buffer = accumulator_to_rgb32f(&accumulator, width as u32, height as u32);
    Ok(DynamicImage::ImageRgb32F(out_img_buffer))
}
