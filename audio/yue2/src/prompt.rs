//! The prompts and the token ids, as upstream's protocol.py writes them (yue2-native-v1): the instruction of the cot
//! mode, the style under `[Tags]`, the lyrics under `[Lyrics]`, after an EOD; the score between ABC_START and ABC_END;
//! MUSIC_START before the semantic codes. The text is NFC-normalized, then tiktoken's `encode_ordinary`: a special
//! token's text in the lyrics is read as plain characters.

use nextcuda_tok::Tokenizer;
use unicode_normalization::UnicodeNormalization;

pub const EOD: u32 = 151_643;
pub const ABC_START: u32 = 151_847;
pub const ABC_END: u32 = 151_848;
pub const MUSIC_START: u32 = 151_851;
pub const MUSIC_END: u32 = 151_852;
pub const CODEC_OFFSET: u32 = 151_853;
pub const CODEC_SIZE: u32 = 32_768;

/// What the model plans before the music (protocol.py's INSTRUCTIONS)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cot {
    /// no score: the codes from the words alone
    Off,
    /// a melody-only score first (the covers' mode)
    Melody,
    /// a chord-annotated score first (the default)
    Full,
}

impl Cot {
    pub fn parse(s: &str) -> Option<Cot> {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "0" | "false" => Some(Cot::Off),
            "melody" => Some(Cot::Melody),
            "full" | "" => Some(Cot::Full),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Cot::Off => "off",
            Cot::Melody => "melody",
            Cot::Full => "full",
        }
    }

    pub fn instruction(self) -> &'static str {
        match self {
            Cot::Off => "Generate music with codec tokens from the given conditions.",
            Cot::Melody => "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
            Cot::Full => "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
        }
    }

    /// the semantic phase's guidance when the request names none (SongRequest.guidance)
    pub fn default_cfg(self) -> f32 {
        if self == Cot::Off { 1.01 } else { 1.0 }
    }
}

/// The text encoding upstream's tokenizer makes (NFC, no special tokens read out of the text)
pub fn encode(tok: &Tokenizer, text: &str) -> Vec<u32> {
    tok.encode_ordinary(&text.nfc().collect::<String>())
}

/// The request's text (SongRequest.text)
pub fn text(cot: Cot, style: &str, lyrics: &str) -> String {
    format!("{}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n", cot.instruction())
}

/// The first phase's prompt (token_prefixes with no score yet): cot off ends at MUSIC_START, else at ABC_START (the
/// score is drawn after it)
pub fn prefix(tok: &Tokenizer, cot: Cot, style: &str, lyrics: &str) -> Vec<u32> {
    let mut v = vec![EOD];
    v.extend(encode(tok, &text(cot, style, lyrics)));
    match cot {
        Cot::Off => v.extend([ABC_START, ABC_END, MUSIC_START]),
        _ => v.push(ABC_START),
    }
    v
}

/// The semantic phase's prompt: the first's with its score closed (token_prefixes with the score's ids)
pub fn semantic(first: &[u32], cot: Cot, abc: &[u32]) -> Vec<u32> {
    let mut v = first.to_vec();
    if cot != Cot::Off {
        v.extend_from_slice(abc);
        v.extend([ABC_END, MUSIC_START]);
    }
    v
}

/// The classifier-free twin (negative_prefix): the instruction alone, the same score
pub fn negative(tok: &Tokenizer, cot: Cot, abc: &[u32]) -> Vec<u32> {
    let mut v = vec![EOD];
    v.extend(encode(tok, cot.instruction()));
    if cot == Cot::Off {
        v.push(MUSIC_START);
    } else {
        v.push(ABC_START);
        v.extend_from_slice(abc);
        v.extend([ABC_END, MUSIC_START]);
    }
    v
}

/// chunk_ranges: the song's frames in chunks a context holds beside the prompt (whole songs are one in practice)
pub fn chunks(frames: usize, prompt: usize, context: usize) -> Vec<(usize, usize)> {
    let size = ((context.saturating_sub(prompt + 3)) / 2).min(context);
    if size == 0 {
        return Vec::new();
    }
    (0..frames).step_by(size).map(|a| (a, (a + size).min(frames))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chunks_cover_the_frames() {
        assert_eq!(chunks(9000, 3000, 24576), vec![(0, 9000)]);
        let c = chunks(9000, 10000, 24576);
        assert_eq!(c.first(), Some(&(0, 7286)));
        assert_eq!(c.last(), Some(&(7286, 9000)));
        assert!(chunks(10, 24575, 24576).is_empty());
    }

    #[test]
    fn the_modes_parse() {
        assert_eq!(Cot::parse("Melody"), Some(Cot::Melody));
        assert_eq!(Cot::parse(""), Some(Cot::Full));
        assert_eq!(Cot::parse("OFF"), Some(Cot::Off));
        assert_eq!(Cot::parse("chords"), None);
    }
}
