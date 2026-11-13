//! The engine's stage times as log lines: the text encode, the sampling and the VAE decode of each picture, kept (the
//! last 4,000 lines) so `nextcuda image bench` reads them after a picture.

use std::sync::Mutex;

static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// A line into the log
pub fn record(line: String) {
    let mut l = LOG.lock().unwrap_or_else(|p| p.into_inner());
    if l.len() >= 4000 {
        l.drain(..2000);
    }
    l.push(line);
}

/// The log lines since the last call
pub fn take_log() -> Vec<String> {
    std::mem::take(&mut *LOG.lock().unwrap_or_else(|p| p.into_inner()))
}

/// The stage times the log gives for one picture: (text encode, sampling, VAE decode), seconds
pub fn stage_times(lines: &[String]) -> (Option<f64>, Option<f64>, Option<f64>) {
    let taking = |needle: &str| -> Option<f64> {
        lines.iter().rev().find(|l| l.contains(needle)).and_then(|l| {
            let at = l.rfind("taking ")? + 7;
            l[at..].trim().trim_end_matches('s').parse::<f64>().ok()
        })
    };
    (taking("get_learned_condition completed"), taking("generating 1 latent images completed").or_else(|| taking("sampling completed")),
     taking("decode_first_stage completed"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stage_times_are_read_from_the_lines() {
        let l = ["get_learned_condition completed, taking 0.84s", "generating 1 latent images completed, taking 16.50s", "decode_first_stage completed, taking 2.60s"]
            .map(String::from);
        assert_eq!(stage_times(&l), (Some(0.84), Some(16.5), Some(2.6)));
        assert_eq!(stage_times(&[]), (None, None, None));
    }
}
