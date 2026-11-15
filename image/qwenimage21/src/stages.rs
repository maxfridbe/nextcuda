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

    #[test]
    fn the_last_picture_counts_and_an_older_sampling_line_is_read_too() {
        // two pictures in the lines: the bench wants the last one's times
        let l = ["generating 1 latent images completed, taking 9.00s", "generating 1 latent images completed, taking 7.25s"].map(String::from);
        assert_eq!(stage_times(&l).1, Some(7.25));
        // the wording some engines log for the steps
        assert_eq!(stage_times(&["sampling completed, taking 3.5s".to_string()]).1, Some(3.5));
        // a line without a time is no time
        assert_eq!(stage_times(&["decode_first_stage completed".to_string()]).2, None);
    }

    #[test]
    fn the_log_keeps_its_latest_lines_and_is_emptied_by_a_take() {
        // one test only touches the process's log (tests run in parallel); a full log drops its older half
        let _ = take_log();
        for i in 0..4500 {
            record(format!("line {i}"));
        }
        let l = take_log();
        assert!(l.len() <= 4000, "{}", l.len());
        assert_eq!(l.last().map(String::as_str), Some("line 4499"));
        assert_eq!(l.first().map(String::as_str), Some("line 2000"));
        assert!(take_log().is_empty());
    }
}
