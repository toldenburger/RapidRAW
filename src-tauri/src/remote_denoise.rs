// Proof-of-concept alternative to the local CPU-only AI denoise path in
// `ai_processing.rs`. Instead of running the NIND ONNX model locally one
// tile at a time, this sends every tile for the image to a remote server in
// a single request (so an image with 100 tiles costs one network round
// trip, not 100), and the server runs them through the same model. The
// server-side model itself is exported with a fixed batch size of 1, so it
// still infers tiles one at a time there — the win this PoC tests is
// avoiding per-tile network latency and, eventually, running on a real GPU.
use anyhow::{Result, anyhow};
use image::{DynamicImage, Rgb32FImage};
use reqwest::{Client, multipart};
use tauri::{AppHandle, Emitter};

use crate::ai_processing::{
    SeamlessBlend, TileParams, accumulator_to_rgb32f, apply_seamless, extract_tile_mirror,
    select_tile_params,
};

const CHANNELS: usize = 3;

#[allow(dead_code)] // not wired into a connection-indicator UI yet (out of scope for this PoC)
pub async fn check_status(base_url: &str) -> bool {
    let client = Client::new();
    client
        .get(format!("{}/health", base_url))
        .send()
        .await
        .is_ok()
}

fn f32_slice_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn le_bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct TilePlacement {
    x0: i32,
    y0: i32,
}

pub async fn run_remote_denoise(
    rgb_img: &Rgb32FImage,
    intensity: f32,
    base_url: &str,
    token: Option<&str>,
    app_handle: &AppHandle,
) -> Result<DynamicImage> {
    let (width, height) = rgb_img.dimensions();
    let (width, height) = (width as usize, height as usize);
    let params: TileParams = select_tile_params(intensity);

    let step = params.ucs.saturating_sub(params.overlap).max(1);
    let iperhl = (width.saturating_sub(params.ucs) as f64 / step as f64).ceil() as usize;
    let ipervl = (height.saturating_sub(params.ucs) as f64 / step as f64).ceil() as usize;
    let total = (iperhl + 1) * (ipervl + 1);

    let _ = app_handle.emit(
        "denoise-progress",
        format!("Uploading {} tiles to remote server...", total),
    );

    let mut placements = Vec::with_capacity(total);
    let mut request_bytes = Vec::with_capacity(total * CHANNELS * params.cs * params.cs * 4);

    for i in 0..total {
        let yi = i / (iperhl + 1);
        let xi = i % (iperhl + 1);
        let x0 =
            params.ucs as i32 * xi as i32 - params.overlap as i32 * xi as i32 - params.pad as i32;
        let y0 =
            params.ucs as i32 * yi as i32 - params.overlap as i32 * yi as i32 - params.pad as i32;

        let crop = extract_tile_mirror(rgb_img, x0, y0, params.cs);
        let standard = crop.as_standard_layout();
        request_bytes.extend_from_slice(&f32_slice_to_le_bytes(
            standard.as_slice().ok_or_else(|| anyhow!("Tile data was not contiguous"))?,
        ));
        placements.push(TilePlacement { x0, y0 });
    }

    let client = Client::new();
    let part = multipart::Part::bytes(request_bytes)
        .file_name("tiles.bin")
        .mime_str("application/octet-stream")?;
    let form = multipart::Form::new()
        .text("num_tiles", total.to_string())
        .part("tiles", part);

    let mut req = client
        .post(format!("{}/denoise_batch", base_url))
        .multipart(form);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let res = req.send().await?;

    if !res.status().is_success() {
        return Err(anyhow!(
            "Remote denoise server returned an error: {}",
            res.text().await?
        ));
    }

    let _ = app_handle.emit(
        "denoise-progress",
        "Received denoised tiles, blending...".to_string(),
    );

    let response_bytes = res.bytes().await?;
    let response_floats = le_bytes_to_f32_vec(&response_bytes);

    let tile_floats = CHANNELS * params.cs * params.cs;
    if response_floats.len() != total * tile_floats {
        return Err(anyhow!(
            "Remote denoise server returned {} floats, expected {} for {} tiles",
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
