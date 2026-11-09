//! A scene of cubic Bezier paths as the kernels take it (svg.h's layout: one point array, a shape record a path, its
//! fill, stroke and stroke half-width), and the SVG it is written as.

use std::fmt::Write;

/// A path's record (`ns_svg_path`): where its points start, its segments, its flags
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PathShape {
    pub off: i32,
    pub nseg: i32,
    pub flags: i32,
    pub pad: i32,
}

pub const CLOSED: i32 = 1;
pub const FILL: i32 = 2;
pub const STROKE: i32 = 4;
pub const EVENODD: i32 = 8;
/// the host's alone (the kernels ignore it): a polygon - its segments straight, written as lines
pub const POLYGON: i32 = 16;
/// cull at the box as DiffVG's BVH does, without the pixel of filter margin (the check against DiffVG)
pub const DIFFVG_BOX: i32 = 32;

/// Paths drawn in order on a canvas of `width` x `height` pixels (points in pixels)
#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub width: u32,
    pub height: u32,
    pub paths: Vec<PathShape>,
    /// x, y of every point, path after path
    pub points: Vec<f32>,
    pub fill: Vec<[f32; 4]>,
    pub stroke: Vec<[f32; 4]>,
    /// stroke half-widths (DiffVG's stroke_width; the SVG's stroke-width is twice it)
    pub hw: Vec<f32>,
}

/// The points a path of `nseg` cubic segments has: 3 a segment, and its end point when it is open
pub fn npoints(nseg: i32, closed: bool) -> usize {
    (3 * nseg + if closed { 0 } else { 1 }) as usize
}

impl Scene {
    pub fn new(width: u32, height: u32) -> Scene {
        Scene { width, height, ..Default::default() }
    }

    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Adds a path of cubic segments (`pts`: 3 a segment, plus the end when open); its index
    pub fn add(&mut self, pts: &[[f32; 2]], closed: bool, fill: Option<[f32; 4]>, stroke: Option<([f32; 4], f32)>, flags: i32) -> usize {
        let nseg = if closed { pts.len() / 3 } else { (pts.len().saturating_sub(1)) / 3 } as i32;
        assert_eq!(npoints(nseg, closed), pts.len(), "a path's points: 3 a segment (+1 open)");
        let mut f = flags;
        if closed {
            f |= CLOSED;
        }
        if fill.is_some() {
            f |= FILL;
        }
        if stroke.is_some() {
            f |= STROKE;
        }
        self.paths.push(PathShape { off: (self.points.len() / 2) as i32, nseg, flags: f, pad: 0 });
        for p in pts {
            self.points.extend_from_slice(p);
        }
        self.fill.push(fill.unwrap_or([0.0; 4]));
        let (sc, sw) = stroke.unwrap_or(([0.0; 4], 0.0));
        self.stroke.push(sc);
        self.hw.push(sw);
        self.paths.len() - 1
    }

    /// Path i's points
    pub fn path_points(&self, i: usize) -> &[f32] {
        let p = self.paths[i];
        let n = npoints(p.nseg, p.flags & CLOSED != 0);
        &self.points[2 * p.off as usize..2 * (p.off as usize + n)]
    }

    /// The SVG: the canvas as its viewBox, shown at `out_w` x `out_h`; `bg` a rectangle under the paths
    pub fn to_svg(&self, out_w: u32, out_h: u32, bg: Option<[f32; 3]>) -> String {
        let mut s = String::new();
        let _ = write!(s, r#"<svg xmlns="http://www.w3.org/2000/svg" width="{out_w}" height="{out_h}" viewBox="0 0 {} {}">"#, self.width, self.height);
        s.push('\n');
        if let Some(c) = bg {
            let _ = writeln!(s, r#"<rect width="{}" height="{}" fill="{}"/>"#, self.width, self.height, hex(&[c[0], c[1], c[2]]));
        }
        for i in 0..self.paths.len() {
            let p = self.paths[i];
            let pts = self.path_points(i);
            let closed = p.flags & CLOSED != 0;
            let at = |k: usize| {
                let k = if closed { k % (pts.len() / 2) } else { k };
                (pts[2 * k], pts[2 * k + 1])
            };
            let mut d = String::new();
            let (x, y) = at(0);
            let _ = write!(d, "M{} {}", num(x), num(y));
            for j in 0..p.nseg as usize {
                if p.flags & POLYGON != 0 {
                    let (x, y) = at(3 * j + 3);
                    if !(closed && j + 1 == p.nseg as usize) {
                        let _ = write!(d, "L{} {}", num(x), num(y));
                    }
                } else {
                    let (a, b) = at(3 * j + 1);
                    let (c, e) = at(3 * j + 2);
                    let (f, g) = at(3 * j + 3);
                    let _ = write!(d, "C{} {} {} {} {} {}", num(a), num(b), num(c), num(e), num(f), num(g));
                }
            }
            if closed {
                d.push('Z');
            }
            let mut attrs = String::new();
            if p.flags & FILL != 0 {
                let c = self.fill[i];
                let _ = write!(attrs, r#" fill="{}""#, hex(&c));
                if c[3] < 0.999 {
                    let _ = write!(attrs, r#" fill-opacity="{}""#, num(c[3]));
                }
                if p.flags & EVENODD != 0 {
                    attrs.push_str(r#" fill-rule="evenodd""#);
                }
            } else {
                attrs.push_str(r#" fill="none""#);
            }
            if p.flags & STROKE != 0 && self.hw[i] > 0.0 {
                let c = self.stroke[i];
                let _ = write!(attrs, r#" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round""#, hex(&c), num(2.0 * self.hw[i]));
                if c[3] < 0.999 {
                    let _ = write!(attrs, r#" stroke-opacity="{}""#, num(c[3]));
                }
            }
            let _ = writeln!(s, r#"<path d="{d}"{attrs}/>"#);
        }
        s.push_str("</svg>\n");
        s
    }
}

/// A coordinate in a few characters: two decimals, trailing zeros dropped
fn num(v: f32) -> String {
    let v = if v.is_finite() { v } else { 0.0 };
    let mut t = format!("{v:.2}");
    if t.contains('.') {
        while t.ends_with('0') {
            t.pop();
        }
        if t.ends_with('.') {
            t.pop();
        }
    }
    if t == "-0" {
        t = "0".into();
    }
    t
}

fn hex(c: &[f32]) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8;
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_closed_path_and_a_stroke_are_written_as_svg() {
        let mut s = Scene::new(100, 50);
        s.add(&[[10.0, 10.0], [20.0, 0.0], [30.0, 0.0], [40.0, 10.0], [30.0, 30.0], [20.0, 30.0]], true, Some([1.0, 0.0, 0.0, 0.5]), None, EVENODD);
        s.add(&[[0.0, 0.0], [10.0, 10.0], [20.0, 10.0], [30.5, 0.25]], false, None, Some(([0.0, 0.0, 0.0, 1.0], 1.5)), 0);
        let svg = s.to_svg(200, 100, Some([1.0, 1.0, 1.0]));
        assert!(svg.starts_with(r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100" viewBox="0 0 100 50">"#));
        assert!(svg.contains(r##"<rect width="100" height="50" fill="#ffffff"/>"##));
        assert!(svg.contains(r##"<path d="M10 10C20 0 30 0 40 10C30 30 20 30 10 10Z" fill="#ff0000" fill-opacity="0.5" fill-rule="evenodd"/>"##));
        assert!(svg.contains(r##"<path d="M0 0C10 10 20 10 30.5 0.25" fill="none" stroke="#000000" stroke-width="3""##));
        assert!(svg.trim_end().ends_with("</svg>"));
        assert_eq!(s.path_points(1).len(), 8);
    }

    #[test]
    fn a_polygon_is_written_as_lines() {
        let mut s = Scene::new(8, 8);
        let sq = |x: f32, y: f32| [[x, y], [x + 4.0 / 3.0, y], [x + 8.0 / 3.0, y], [x + 4.0, y], [x + 4.0, y + 4.0 / 3.0], [x + 4.0, y + 8.0 / 3.0],
                                   [x + 4.0, y + 4.0], [x + 8.0 / 3.0, y + 4.0], [x + 4.0 / 3.0, y + 4.0], [x, y + 4.0], [x, y + 8.0 / 3.0], [x, y + 4.0 / 3.0]];
        s.add(&sq(0.0, 0.0), true, Some([0.0, 0.5, 1.0, 1.0]), None, POLYGON);
        let svg = s.to_svg(8, 8, None);
        assert!(svg.contains(r##"<path d="M0 0L4 0L4 4L0 4Z" fill="#0080ff"/>"##), "{svg}");
    }
}
