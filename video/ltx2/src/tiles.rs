//! Tiled work for the VAE (diffusers' `enable_tiling`): the pieces a big canvas is cut into, blended where they
//! overlap and put side by side again. The same for the decoder's pixels and the encoder's latents (only the channel
//! count and the units differ), apart from any GPU so that a seam moved by a refactor shows in a test.
//!
//! A tile is `(data [frames, h, w, c], h, w)`; the tiles start every `stride` and are `tile` big (the last ones
//! smaller); each tile's top rows and left columns are cross-faded with the neighbour above and to the left over
//! `blend = tile - stride`; the output takes each tile's first `stride` x `stride` cells.

pub type Tile = (Vec<f32>, usize, usize);

/// The tile origins along an extent
pub fn starts(extent: usize, stride: usize) -> Vec<usize> {
    (0..extent).step_by(stride.max(1)).collect()
}

/// The window at (i, j), th x tw big, of `src` [frames, h, w, c]
#[allow(clippy::too_many_arguments)]
pub fn crop(src: &[f32], frames: usize, h: usize, w: usize, c: usize, i: usize, j: usize, th: usize, tw: usize) -> Vec<f32> {
    let mut sub = vec![0f32; frames * th * tw * c];
    for t in 0..frames {
        for y in 0..th {
            let s = ((t * h + i + y) * w + j) * c;
            let d = (t * th + y) * tw * c;
            sub[d..d + tw * c].copy_from_slice(&src[s..s + tw * c]);
        }
    }
    sub
}

/// Cross-fade each tile's top `blend` rows with the bottom rows of the tile above, then its left columns with the
/// right columns of the tile to its left (the linear ramp k / e from the neighbour to the tile)
pub fn blend(tiles: &mut [Vec<Tile>], frames: usize, c: usize, blend: usize) {
    for ri in 0..tiles.len() {
        for ci in 0..tiles[ri].len() {
            if ri > 0 {
                let (a, b) = tiles.split_at_mut(ri);
                let (above, cur) = (&a[ri - 1][ci], &mut b[0][ci]);
                let e = blend.min(above.1).min(cur.1);
                let (ah, aw, cw) = (above.1, above.2, cur.2);
                for t in 0..frames {
                    for y in 0..e {
                        let k = y as f32 / e as f32;
                        for x in 0..cw.min(aw) {
                            for col in 0..c {
                                let av = above.0[((t * ah + ah - e + y) * aw + x) * c + col];
                                let bi = ((t * cur.1 + y) * cw + x) * c + col;
                                cur.0[bi] = av * (1.0 - k) + cur.0[bi] * k;
                            }
                        }
                    }
                }
            }
            if ci > 0 {
                let (a, b) = tiles[ri].split_at_mut(ci);
                let (left, cur) = (&a[ci - 1], &mut b[0]);
                let e = blend.min(left.2).min(cur.2);
                let (lh, lw) = (left.1, left.2);
                for t in 0..frames {
                    for y in 0..cur.1.min(lh) {
                        for x in 0..e {
                            let k = x as f32 / e as f32;
                            for col in 0..c {
                                let av = left.0[((t * lh + y) * lw + lw - e + x) * c + col];
                                let bi = ((t * cur.1 + y) * cur.2 + x) * c + col;
                                cur.0[bi] = av * (1.0 - k) + cur.0[bi] * k;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The blended tiles' first `stride` x `stride` cells side by side: [frames, out_h, out_w, c], each value through `f`
pub fn place(tiles: &[Vec<Tile>], frames: usize, c: usize, stride: usize, out_h: usize, out_w: usize, f: impl Fn(f32) -> f32) -> Vec<f32> {
    let mut out = vec![0f32; frames * out_h * out_w * c];
    for (ri, row) in tiles.iter().enumerate() {
        for (ci, (px, th, tw)) in row.iter().enumerate() {
            let (y0, x0) = (ri * stride, ci * stride);
            for t in 0..frames {
                for y in 0..stride.min(*th).min(out_h.saturating_sub(y0)) {
                    for x in 0..stride.min(*tw).min(out_w.saturating_sub(x0)) {
                        for col in 0..c {
                            out[((t * out_h + y0 + y) * out_w + x0 + x) * c + col] = f(px[((t * th + y) * tw + x) * c + col]);
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a [frames, h, w, c] volume whose value says where it is
    fn vol(frames: usize, h: usize, w: usize, c: usize) -> Vec<f32> {
        (0..frames * h * w * c).map(|i| i as f32).collect()
    }

    /// cut `src` into tiles of `tile` every `stride`, as the decoder and encoder do
    fn cut(src: &[f32], frames: usize, h: usize, w: usize, c: usize, tile: usize, stride: usize) -> Vec<Vec<Tile>> {
        starts(h, stride).iter().map(|&i| starts(w, stride).iter().map(|&j| {
            let (th, tw) = (tile.min(h - i), tile.min(w - j));
            (crop(src, frames, h, w, c, i, j, th, tw), th, tw)
        }).collect()).collect()
    }

    #[test]
    fn tiles_start_every_stride_and_cover_the_extent() {
        assert_eq!(starts(16, 6), vec![0, 6, 12]);
        assert_eq!(starts(8, 6), vec![0, 6]);
        assert_eq!(starts(6, 6), vec![0]);
        for (extent, stride) in [(7, 6), (12, 6), (13, 6), (40, 14), (24, 14)] {
            let s = starts(extent, stride);
            assert_eq!(s[0], 0);
            assert!(*s.last().unwrap() + stride >= extent && *s.last().unwrap() < extent, "{extent} {stride}");
        }
    }

    #[test]
    fn a_crop_is_the_window() {
        let v = vol(2, 4, 5, 3);
        let c = crop(&v, 2, 4, 5, 3, 1, 2, 2, 3);
        assert_eq!(c.len(), 2 * 2 * 3 * 3);
        // frame 0, row 1, col 2, channel 0 is element ((0 * 4 + 1) * 5 + 2) * 3
        assert_eq!(c[0], ((0 * 4 + 1) * 5 + 2) as f32 * 3.0);
        // frame 1, row 2, col 4, channel 2 is the last element of the window
        assert_eq!(*c.last().unwrap(), (((4 + 2) * 5 + 4) * 3 + 2) as f32);
    }

    #[test]
    fn untouched_tiles_stitch_back_to_the_volume() {
        // identical overlaps blend to themselves: cutting then stitching gives the original
        for (h, w, tile, stride) in [(16usize, 16usize, 8usize, 6usize), (20, 12, 8, 6), (9, 9, 8, 6), (14, 28, 14, 8)] {
            let src = vol(2, h, w, 3);
            let mut tiles = cut(&src, 2, h, w, 3, tile, stride);
            blend(&mut tiles, 2, 3, tile - stride);
            let out = place(&tiles, 2, 3, stride, h, w, |v| v);
            assert_eq!(out.len(), src.len());
            let worst = out.iter().zip(&src).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst < 1e-3 * src.len() as f32, "{h}x{w} tile {tile} stride {stride}: {worst}");
            assert!(out.iter().zip(&src).filter(|(a, b)| (*a - *b).abs() > 0.5).count() == 0, "{h}x{w}: a seam moved");
        }
    }

    #[test]
    fn the_overlap_fades_from_the_neighbour_to_the_tile() {
        // two tiles side by side, 1 row, 1 channel, 4 wide each, overlapping 2 columns: left all 10, right all 20
        let mut tiles = vec![vec![(vec![10.0; 4], 1, 4), (vec![20.0; 4], 1, 4)]];
        blend(&mut tiles, 1, 1, 2);
        // the right tile's first two columns: k = 0 -> the left's value, k = 0.5 -> halfway
        assert_eq!(tiles[0][1].0, vec![10.0, 15.0, 20.0, 20.0]);
        assert_eq!(tiles[0][0].0, vec![10.0; 4], "the first tile is left alone");
        // a vertical pair does the same down the rows
        let mut tiles = vec![vec![(vec![10.0; 4], 4, 1)], vec![(vec![20.0; 4], 4, 1)]];
        blend(&mut tiles, 1, 1, 2);
        assert_eq!(tiles[1][0].0, vec![10.0, 15.0, 20.0, 20.0]);
    }

    #[test]
    fn the_output_takes_each_tiles_first_stride_cells_through_the_map() {
        let tiles = vec![vec![(vec![1.0, 2.0, 3.0, 4.0], 1, 4), (vec![5.0, 6.0, 7.0, 8.0], 1, 4)]];
        // stride 2: tile 0's cells 1, 2; tile 1's cells 5, 6 (the decoder maps [-1, 1] to [0, 1])
        assert_eq!(place(&tiles, 1, 1, 2, 1, 4, |v| v), vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(place(&tiles, 1, 1, 2, 1, 4, |v| (v / 2.0 + 0.5).clamp(0.0, 1.0)), vec![1.0, 1.0, 1.0, 1.0]);
        // a canvas smaller than the tiles' reach is cut at its edge
        assert_eq!(place(&tiles, 1, 1, 2, 1, 3, |v| v), vec![1.0, 2.0, 5.0]);
    }
}
