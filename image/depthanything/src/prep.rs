//! The host's part: the picture made the network's input as transformers' DPTImageProcessor makes it (Depth Anything
//! V2's preprocessor_config: the side nearest 518 that keeps the aspect, both sides multiples of 14, Pillow's bicubic,
//! / 255, ImageNet's mean and deviation), the position table resized to the patches as Dinov2Embeddings does it, and
//! the depth map back at the picture's size.

use nextcuda_image::Picture;

/// The side the processor aims for (size 518 x 518, keep_aspect_ratio)
pub const SIDE: usize = 518;
/// DINOv2's patch, and what both sides are a multiple of (ensure_multiple_of)
pub const PATCH: usize = 14;
pub const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// DPT's constrain_to_multiple_of: Python's round (half to even) to a multiple, at least one patch here (the
/// processor's own answer for a sliver of a picture is 0, which no network takes)
fn to_multiple(v: f64) -> usize {
    let x = ((v / PATCH as f64).round_ties_even() as usize) * PATCH;
    x.max(PATCH)
}

/// The network's input size (width, height) for a picture of `w` x `h`: DPT's get_resize_output_image_size with
/// keep_aspect_ratio - the scale that changes the picture least (nearer 1) for both sides
pub fn input_size(w: u32, h: u32) -> (usize, usize) {
    let (sh, sw) = (SIDE as f64 / h as f64, SIDE as f64 / w as f64);
    let s = if (1.0 - sw).abs() < (1.0 - sh).abs() { sw } else { sh };
    (to_multiple(s * w as f64), to_multiple(s * h as f64))
}

/// The picture as the network takes it: resized (Pillow's bicubic on 8-bit RGB, as the processor's PIL resize), in
/// [0, 1], normalized - float32 [h, w, 3], channels last; and its size (width, height)
pub fn prepare(pic: &Picture) -> (Vec<f32>, usize, usize) {
    let (w, h) = input_size(pic.width, pic.height);
    let r = pic.rgb().resize_bicubic(w as u32, h as u32);
    let k = [1.0 / (255.0 * STD[0]), 1.0 / (255.0 * STD[1]), 1.0 / (255.0 * STD[2])];
    let b = [MEAN[0] / STD[0], MEAN[1] / STD[1], MEAN[2] / STD[2]];
    let px = r.data.as_chunks::<3>().0.iter().flat_map(|p| (0..3).map(move |c| p[c] as f32 * k[c] - b[c])).collect();
    (px, w, h)
}

/// PyTorch's bicubic resize (align_corners false, A = -0.75, indices clamped) of planes [c, n, n] to [c, gh, gw]:
/// Dinov2Embeddings.interpolate_pos_encoding's, which transformers runs with size=(gh, gw)
pub fn bicubic(src: &[f32], c: usize, n: usize, gh: usize, gw: usize) -> Vec<f32> {
    let taps = |m: usize| -> Vec<(i64, [f32; 4])> {
        let a = -0.75f32;
        let c1 = |x: f32| ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        let c2 = |x: f32| ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
        let scale = n as f32 / m as f32;
        (0..m).map(|o| {
            let s = (o as f32 + 0.5) * scale - 0.5;
            let f = s.floor();
            let t = s - f;
            (f as i64, [c2(t + 1.0), c1(t), c1(1.0 - t), c2(2.0 - t)])
        }).collect()
    };
    let (ty, tx) = (taps(gh), taps(gw));
    let at = |i: i64| i.clamp(0, n as i64 - 1) as usize;
    let mut out = vec![0f32; c * gh * gw];
    for ch in 0..c {
        let plane = &src[ch * n * n..(ch + 1) * n * n];
        for (oy, (fy, wy)) in ty.iter().enumerate() {
            for (ox, (fx, wx)) in tx.iter().enumerate() {
                let mut acc = 0f32;
                for (j, wyj) in wy.iter().enumerate() {
                    let row = &plane[at(fy - 1 + j as i64) * n..];
                    let mut r = 0f32;
                    for (i, wxi) in wx.iter().enumerate() {
                        r += wxi * row[at(fx - 1 + i as i64)];
                    }
                    acc += wyj * r;
                }
                out[ch * gh * gw + oy * gw + ox] = acc;
            }
        }
    }
    out
}

/// The position table for gh x gw patches: rows [class token's | the patches' (row-major)] of `width`, from the
/// stored table [1 + n n, width] (transformers keeps it as it is when the patches are its own n x n)
pub fn positions(table: &[f32], width: usize, gh: usize, gw: usize) -> Result<Vec<f32>, String> {
    let rows = table.len() / width;
    let n = ((rows - 1) as f64).sqrt().round() as usize;
    if n * n + 1 != rows {
        return Err(format!("a position table of {rows} rows"));
    }
    let mut pos = table[..width].to_vec();
    if (gh, gw) == (n, n) {
        pos.extend_from_slice(&table[width..]);
        return Ok(pos);
    }
    // [n n, c] -> [c, n, n] -> bicubic -> [gh gw, c]
    let planes: Vec<f32> = (0..width).flat_map(|c| (0..n * n).map(move |i| (c, i))).map(|(c, i)| table[(1 + i) * width + c]).collect();
    let r = bicubic(&planes, width, n, gh, gw);
    for i in 0..gh * gw {
        pos.extend((0..width).map(|c| r[c * gh * gw + i]));
    }
    Ok(pos)
}

/// A map [h, w] resized to [ho, wo] bilinearly (half-pixel centres, align_corners false: what PyTorch's interpolate
/// does with mode bilinear)
pub fn resize_bilinear(d: &[f32], w: usize, h: usize, wo: usize, ho: usize) -> Vec<f32> {
    if (w, h) == (wo, ho) {
        return d.to_vec();
    }
    let axis = |n: usize, m: usize| -> Vec<(usize, usize, f32)> {
        let s = n as f32 / m as f32;
        (0..m).map(|o| {
            let x = ((o as f32 + 0.5) * s - 0.5).max(0.0);
            let i0 = (x.floor() as usize).min(n - 1);
            let i1 = (i0 + 1).min(n - 1);
            (i0, i1, x - i0 as f32)
        }).collect()
    };
    let (ax, ay) = (axis(w, wo), axis(h, ho));
    let mut out = vec![0f32; wo * ho];
    for (y, &(y0, y1, fy)) in ay.iter().enumerate() {
        for (x, &(x0, x1, fx)) in ax.iter().enumerate() {
            let top = d[y0 * w + x0] * (1.0 - fx) + d[y0 * w + x1] * fx;
            let bot = d[y1 * w + x0] * (1.0 - fx) + d[y1 * w + x1] * fx;
            out[y * wo + x] = top * (1.0 - fy) + bot * fy;
        }
    }
    out
}

/// A convolution's weight [co, ci, k, k] as the im2col's rows take it: [co, k, k, ci]
pub fn conv_weight(w: &[f32], co: usize, ci: usize, k: usize) -> Vec<f32> {
    let mut r = vec![0f32; w.len()];
    for o in 0..co {
        for c in 0..ci {
            for t in 0..k * k {
                r[(o * k * k + t) * ci + c] = w[(o * ci + c) * k * k + t];
            }
        }
    }
    r
}

/// A transposed convolution with its stride its kernel (k x k, no overlap) as one product: its weight [ci, co, k, k] as
/// a linear layer's [co k k, ci], output column co k k + i k + j - the order of PixelShuffle(k)'s input channels
pub fn deconv_weight(w: &[f32], ci: usize, co: usize, k: usize) -> Vec<f32> {
    let mut r = vec![0f32; w.len()];
    for c in 0..ci {
        for o in 0..co {
            for t in 0..k * k {
                r[(o * k * k + t) * ci + c] = w[(c * co + o) * k * k + t];
            }
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_sizes_are_the_processors() {
        // landscape: the height to 518, the width its aspect on the 14 grid (Python's round)
        assert_eq!(input_size(768, 576), (686, 518));
        assert_eq!(input_size(1280, 720), (924, 518));
        assert_eq!(input_size(576, 768), (518, 686));
        assert_eq!(input_size(518, 518), (518, 518));
        assert_eq!(input_size(256, 256), (518, 518));
        // a small picture made bigger: still the scale nearer 1 (here the width's)
        assert_eq!(input_size(300, 200), (518, 350));
        // a long strip: the width's scale changes it less, so the height follows it
        assert_eq!(input_size(1344, 256), (518, 98));
        // half to even: 7.5 patches -> 8, 6.5 -> 6
        assert_eq!(to_multiple(105.0), 112);
        assert_eq!(to_multiple(91.0), 84);
        assert_eq!(to_multiple(3.0), 14, "never no patch");
        for (w, h) in [(640, 480), (1920, 1080), (333, 777), (4000, 3000)] {
            let (a, b) = input_size(w, h);
            assert!(a % 14 == 0 && b % 14 == 0 && a.min(b) <= 518 + 14 && a.max(b) >= 518 - 14, "{w}x{h} -> {a}x{b}");
        }
    }

    #[test]
    fn preparation_normalizes_channels_last() {
        let pic = Picture { width: 518, height: 518, channels: 3, data: [255u8, 0, 128].repeat(518 * 518) };
        let (px, w, h) = prepare(&pic);
        assert_eq!((w, h, px.len()), (518, 518, 518 * 518 * 3));
        let want = [(1.0 - MEAN[0]) / STD[0], (0.0 - MEAN[1]) / STD[1], (128.0 / 255.0 - MEAN[2]) / STD[2]];
        for c in 0..3 {
            assert!((px[3 * 1000 + c] - want[c]).abs() < 1e-5, "channel {c}: {} vs {}", px[3 * 1000 + c], want[c]);
        }
    }

    #[test]
    fn positions_keep_their_own_grid_and_resize_others() {
        let (width, n) = (2, 3);
        let table: Vec<f32> = (0..(1 + n * n) * width).map(|i| i as f32).collect();
        assert_eq!(positions(&table, width, 3, 3).unwrap(), table);
        // the height kept (3 -> 3 is the identity in bicubic), the width resized; a constant plane stays constant
        let flat: Vec<f32> = (0..(1 + n * n) * width).map(|i| if i < width { 9.0 } else { (i % width) as f32 }).collect();
        let p = positions(&flat, width, 3, 5).unwrap();
        assert_eq!(p.len(), (1 + 15) * width);
        assert_eq!(&p[..2], &[9.0, 9.0]);
        assert!(p[2..].chunks(2).all(|r| r[0].abs() < 1e-6 && (r[1] - 1.0).abs() < 1e-6));
        let r = bicubic(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1, 3, 3, 3);
        assert_eq!(r, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(positions(&[0.0; 6], 2, 1, 1).is_err(), "2 rows is no square table");
    }

    #[test]
    fn bilinear_resize_keeps_ramps_and_size() {
        let d: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let r = resize_bilinear(&d, 4, 1, 8, 2);
        assert_eq!(r.len(), 16);
        assert_eq!(r[0], 0.0);
        assert_eq!(r[7], 3.0);
        assert!((r[3] - 1.25).abs() < 1e-6, "{r:?}");
        assert_eq!(&r[..8], &r[8..]);
    }

    #[test]
    fn weights_are_reordered_for_the_products() {
        // conv [co 1, ci 2, 1 x 1] is already [co, k, k, ci]
        assert_eq!(conv_weight(&[1.0, 2.0], 1, 2, 1), vec![1.0, 2.0]);
        // [co 1, ci 2, 2 x 2] -> taps outer, channels inner
        let w: Vec<f32> = (0..8).map(|i| i as f32).collect();
        assert_eq!(conv_weight(&w, 1, 2, 2), vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]);
        // deconv [ci 2, co 1, 2 x 2]: row (o, i, j) holds W[:, o, i, j]
        assert_eq!(deconv_weight(&w, 2, 1, 2), vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]);
        // deconv [ci 1, co 2, 1 x 1]: one row an output channel
        assert_eq!(deconv_weight(&[3.0, 4.0], 1, 2, 1), vec![3.0, 4.0]);
    }
}
