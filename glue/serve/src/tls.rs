//! https beside http: the front door's same routes over TLS on a second port (NS_TLS_PORT, 8443), with a
//! self-signed certificate it makes itself.
//!
//! Why: a browser gives a page the microphone only over https or from localhost, and the box's pages are opened over
//! plain http on the LAN and the tailnet. All Rust, in process, nothing of C: rustls with the graviola provider
//! (rustls-graviola: pure Rust, its assembly as Rust inline asm - no build script, no cc, unlike aws-lc-rs and ring),
//! the certificate made by rcgen with no crypto backend of its own (`GraviolaKey` signs for it), a kept one read by
//! x509-parser. The connections stay what they are on http - std threads, one request each (`Conn::Tls`); a WebSocket
//! passed on to a model's server (/v1/realtime) is pumped (`TlsConn::pump`) so it is read and written both ways at once.
//!
//! The certificate is made on the first start (`tls/cert.pem` and `tls/key.pem` beside serve.json, the key mode 600)
//! and kept, so the exception a browser was given - or the trust a device was told to place in it (`GET /cert.pem`) -
//! stays valid. A new one only when it is missing or within 30 days of its end (it is made for 10 years), or when a
//! name NS_TLS_SANS configures is not in it. One this server did not make (NS_TLS_CERT pointing at another) is used
//! as it is and never replaced. The names found - localhost, 127.0.0.1, ::1, the hostname, the tailnet name and
//! addresses, every stable address of the host - are read when a certificate is made: an address that changes later
//! does not make a new one (that would undo the browsers' trust at every lease); the log names what the kept one lacks.

use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use graviola::hashing::Sha256;
use graviola::signing::ecdsa::{SigningKey, P256};

/// How long a certificate made here is good for (days)
pub const VALID_DAYS: i64 = 3650;
/// One this close to its end (days) is made anew at start
pub const RENEW_DAYS: i64 = 30;
/// The subject's organisation on every certificate made here: how a kept one is known as this server's own
const ORG: &str = "nextcuda self-signed";

/// What `ensure_cert` did
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CertState {
    /// made now (why)
    Made(String),
    /// the kept one, still good
    Kept,
    /// a certificate made elsewhere: used as it is
    Foreign,
}

/// The names and addresses a new certificate is for: `configured` when it names any, else what the host has
pub fn wanted_sans(configured: &[String]) -> Vec<String> {
    let configured: Vec<String> = configured.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if !configured.is_empty() {
        return dedup(configured);
    }
    let mut out: Vec<String> = vec!["localhost".into(), "127.0.0.1".into(), "::1".into()];
    if let Some(h) = std::fs::read_to_string("/proc/sys/kernel/hostname").ok().map(|h| h.trim().to_string()).filter(|h| !h.is_empty() && h != "localhost") {
        if let Some((short, _)) = h.split_once('.') {
            out.push(short.to_string());
        }
        out.push(h);
    }
    let (ts_name, ts_ips) = tailscale_self();
    out.extend(ts_name);
    out.extend(host_addresses());
    out.extend(ts_ips);
    dedup(out)
}

fn dedup(v: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    v.into_iter().filter(|s| seen.insert(s.to_ascii_lowercase())).collect()
}

/// The host's addresses: every local IPv4 (/proc/net/fib_trie) and every global or unique-local IPv6 that is not a
/// temporary (privacy) one (/proc/net/if_inet6); loopback and link-local left out (127.0.0.1 and ::1 are named apart)
pub fn host_addresses() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(t) = std::fs::read_to_string("/proc/net/fib_trie") {
        out.extend(fib_trie_locals(&t));
    }
    if let Ok(t) = std::fs::read_to_string("/proc/net/if_inet6") {
        out.extend(if_inet6_stable(&t));
    }
    dedup(out)
}

fn fib_trie_locals(t: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut last: Option<&str> = None;
    for l in t.lines() {
        let l = l.trim();
        if let Some(ip) = l.strip_prefix("|-- ").or_else(|| l.strip_prefix("+-- ").map(|x| x.split('/').next().unwrap_or(x))) {
            last = Some(ip.trim());
        } else if l.starts_with("/32 host LOCAL") {
            if let Some(ip) = last.and_then(|x| x.parse::<std::net::Ipv4Addr>().ok()) {
                if !ip.is_loopback() && !ip.is_link_local() {
                    out.push(ip.to_string());
                }
            }
        }
    }
    out
}

fn if_inet6_stable(t: &str) -> Vec<String> {
    let mut out = Vec::new();
    for l in t.lines() {
        let f: Vec<&str> = l.split_whitespace().collect();
        if f.len() < 6 || f[0].len() != 32 {
            continue;
        }
        let (scope, flags) = (u32::from_str_radix(f[3], 16).unwrap_or(0xff), u32::from_str_radix(f[4], 16).unwrap_or(0));
        // scope 0: global (ULA included); flag 0x01 = IFA_F_TEMPORARY, the privacy address that rotates
        if scope != 0 || flags & 0x01 != 0 {
            continue;
        }
        let Ok(n) = u128::from_str_radix(f[0], 16) else { continue };
        out.push(std::net::Ipv6Addr::from(n).to_string());
    }
    out
}

/// This host's tailnet name and addresses (`tailscale status --json`; nothing without tailscale, 3 s at most)
fn tailscale_self() -> (Option<String>, Vec<String>) {
    use std::process::{Command, Stdio};
    let Ok(mut child) = Command::new("tailscale").args(["status", "--json", "--peers=false"]).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn() else {
        return (None, Vec::new());
    };
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if start.elapsed() < Duration::from_secs(3) => std::thread::sleep(Duration::from_millis(50)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return (None, Vec::new());
            }
        }
    }
    let mut buf = Vec::new();
    if let Some(mut o) = child.stdout.take() {
        let _ = o.read_to_end(&mut buf);
    }
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&buf) else { return (None, Vec::new()) };
    let name = v["Self"]["DNSName"].as_str().map(|s| s.trim_end_matches('.').to_string()).filter(|s| !s.is_empty());
    let ips = v["Self"]["TailscaleIPs"].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(String::from)).collect();
    (name, ips)
}

/// What a kept certificate says: whether it was made here, its names, its end (unix seconds)
pub struct CertInfo {
    pub ours: bool,
    pub sans: Vec<String>,
    pub not_after: i64,
}

pub fn read_cert(pem: &[u8]) -> Result<CertInfo, String> {
    use x509_parser::prelude::*;
    let (_, p) = x509_parser::pem::parse_x509_pem(pem).map_err(|e| format!("not a PEM certificate: {e}"))?;
    let c = p.parse_x509().map_err(|e| format!("not an X.509 certificate: {e}"))?;
    let ours = c.subject() == c.issuer() && c.subject().iter_organization().any(|o| o.as_str().ok() == Some(ORG));
    let mut sans = Vec::new();
    if let Ok(Some(ext)) = c.subject_alternative_name() {
        for n in &ext.value.general_names {
            match n {
                GeneralName::DNSName(d) => sans.push(d.to_string()),
                GeneralName::IPAddress(b) => match b.len() {
                    4 => sans.push(std::net::Ipv4Addr::from(<[u8; 4]>::try_from(*b).unwrap()).to_string()),
                    16 => sans.push(std::net::Ipv6Addr::from(<[u8; 16]>::try_from(*b).unwrap()).to_string()),
                    _ => {}
                },
                _ => {}
            }
        }
    }
    Ok(CertInfo { ours, sans, not_after: c.validity().not_after.timestamp() })
}

/// An ECDSA P-256 key of graviola's, signing for rcgen (which then needs no crypto backend: ring and aws-lc-rs are
/// both C)
struct GraviolaKey {
    key: SigningKey<P256>,
    /// the uncompressed public point (SubjectPublicKeyInfo's BIT STRING)
    public: Vec<u8>,
}

impl GraviolaKey {
    fn generate() -> Result<Self, String> {
        let key = SigningKey::<P256>::generate().map_err(|e| format!("a key: {e:?}"))?;
        let mut buf = [0u8; 128];
        let spki = key.to_spki_der(&mut buf).map_err(|e| format!("the public key: {e:?}"))?;
        // P-256's point is the last 65 bytes of its SPKI (0x04 || x || y)
        let public = spki[spki.len() - 65..].to_vec();
        Ok(GraviolaKey { key, public })
    }

    fn pkcs8_pem(&self) -> Result<String, String> {
        let mut buf = [0u8; 256];
        let der = self.key.to_pkcs8_der(&mut buf).map_err(|e| format!("the key's PKCS#8: {e:?}"))?;
        Ok(pem("PRIVATE KEY", der))
    }
}

impl rcgen::PublicKeyData for GraviolaKey {
    fn der_bytes(&self) -> &[u8] {
        &self.public
    }
    fn algorithm(&self) -> &'static rcgen::SignatureAlgorithm {
        &rcgen::PKCS_ECDSA_P256_SHA256
    }
}

impl rcgen::SigningKey for GraviolaKey {
    fn sign(&self, msg: &[u8]) -> Result<Vec<u8>, rcgen::Error> {
        let mut out = [0u8; 80];
        self.key.sign_asn1::<Sha256>(&[msg], &mut out).map(|s| s.to_vec()).map_err(|_| rcgen::Error::RemoteKeyError)
    }
}

/// A random positive 128-bit serial (rcgen derives one only with a crypto backend of its own)
fn serial() -> Result<rcgen::SerialNumber, String> {
    let mut b = [0u8; 16];
    graviola::random::fill(&mut b).map_err(|e| format!("random: {e:?}"))?;
    b[0] = (b[0] & 0x7f) | 0x01;
    Ok(rcgen::SerialNumber::from_slice(&b))
}

fn pem(label: &str, der: &[u8]) -> String {
    let b = crate::http::base64(der);
    let mut s = format!("-----BEGIN {label}-----\n");
    for line in b.as_bytes().chunks(64) {
        s.push_str(std::str::from_utf8(line).unwrap_or_default());
        s.push('\n');
    }
    s.push_str(&format!("-----END {label}-----\n"));
    s
}

/// A new self-signed certificate and its key (PEM both): ECDSA P-256, for `sans`, good from yesterday for `days`
pub fn make_cert(sans: &[String], days: i64) -> Result<(String, String), String> {
    use rcgen::{CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, IsCa, KeyUsagePurpose};
    let mut params = CertificateParams::new(sans.to_vec()).map_err(|e| format!("the certificate's names: {e}"))?;
    let mut dn = DistinguishedName::new();
    let cn = sans.iter().find(|s| s.parse::<std::net::IpAddr>().is_err() && *s != "localhost").or(sans.first()).cloned().unwrap_or_else(|| "localhost".into());
    dn.push(DnType::CommonName, cn);
    dn.push(DnType::OrganizationName, ORG);
    params.distinguished_name = dn;
    let now = time::OffsetDateTime::now_utc();
    params.not_before = now - time::Duration::days(1);
    params.not_after = now + time::Duration::days(days);
    // a leaf, not a CA: Firefox refuses a CA certificate used as a server's (CA_CERT_USED_AS_END_ENTITY)
    params.is_ca = IsCa::ExplicitNoCa;
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    params.serial_number = Some(serial()?);
    let key = GraviolaKey::generate()?;
    let cert = params.self_signed(&key).map_err(|e| format!("signing the certificate: {e}"))?;
    Ok((cert.pem(), key.pkcs8_pem()?))
}

/// The certificate at `cert` (its key at `key`): kept when it is good, made when it is missing, ending within
/// `RENEW_DAYS`, or lacks a name of `configured` (NS_TLS_SANS; empty = the names found)
pub fn ensure_cert(cert: &Path, key: &Path, configured: &[String]) -> Result<CertState, String> {
    ensure_cert_with(cert, key, configured, VALID_DAYS)
}

pub fn ensure_cert_with(cert: &Path, key: &Path, configured: &[String], days: i64) -> Result<CertState, String> {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let reason = match (std::fs::read(cert), key.exists()) {
        (Ok(pem), true) => match read_cert(&pem) {
            Ok(info) => {
                let left = info.not_after - now;
                let lacking: Vec<&str> = configured.iter().map(|s| s.trim()).filter(|s| !s.is_empty() && !info.sans.iter().any(|x| x.eq_ignore_ascii_case(s))).collect();
                if !info.ours {
                    if left < RENEW_DAYS * 86400 {
                        eprintln!("https: {} ends in {} days; it was not made here, so it is left as it is", cert.display(), left / 86400);
                    }
                    return Ok(CertState::Foreign);
                }
                if left < RENEW_DAYS * 86400 {
                    format!("the kept one ends in {} days", left / 86400)
                } else if !lacking.is_empty() {
                    format!("the kept one lacks {}", lacking.join(", "))
                } else {
                    if configured.iter().all(|s| s.trim().is_empty()) {
                        let missing: Vec<String> = wanted_sans(&[]).into_iter().filter(|w| !info.sans.iter().any(|x| x.eq_ignore_ascii_case(w))).collect();
                        if !missing.is_empty() {
                            eprintln!("https: the kept certificate does not name {} (delete {} and restart for a new one)", missing.join(", "), cert.display());
                        }
                    }
                    return Ok(CertState::Kept);
                }
            }
            Err(e) => format!("the kept one is unreadable ({e})"),
        },
        _ => "there was none".to_string(),
    };
    let sans = wanted_sans(configured);
    let (cert_pem, key_pem) = make_cert(&sans, days)?;
    write_file(key, key_pem.as_bytes(), 0o600)?;
    write_file(cert, cert_pem.as_bytes(), 0o644)?;
    eprintln!("https: a new certificate at {} for {} ({reason})", cert.display(), sans.join(", "));
    Ok(CertState::Made(reason))
}

/// Written beside and renamed over, so a reader never sees half a file; the key is 600 from its first byte
fn write_file(p: &Path, data: &[u8], mode: u32) -> Result<(), String> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let dir = p.parent().unwrap_or(Path::new("."));
    if !dir.exists() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
    }
    let tmp = dir.join(format!(".{}.tmp", p.file_name().and_then(|n| n.to_str()).unwrap_or("tls")));
    let _ = std::fs::remove_file(&tmp);
    let mut f = std::fs::OpenOptions::new().write(true).create_new(true).mode(mode).open(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    f.write_all(data).and_then(|_| f.sync_all()).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, p).map_err(|e| format!("{}: {e}", p.display()))
}

/// The rustls server side for the certificate and key at these paths (http/1.1: what the front door speaks)
pub fn server_config(cert: &Path, key: &Path) -> Result<Arc<rustls::ServerConfig>, String> {
    // graviola's x86_64 code needs aes, avx2, adx, bmi2, pclmulqdq (CPUs since about 2014): say so rather than fault
    if !graviola::util::is_cpu_supported() {
        return Err("this CPU lacks what the TLS code needs (graviola: aes, ssse3, avx, avx2, adx, bmi2, pclmulqdq)".into());
    }
    use rustls::pki_types::pem::PemObject;
    use rustls::pki_types::{CertificateDer, PrivateKeyDer};
    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(cert)
        .and_then(|i| i.collect::<Result<_, _>>())
        .map_err(|e| format!("{}: {e}", cert.display()))?;
    if certs.is_empty() {
        return Err(format!("{}: no certificate in it", cert.display()));
    }
    let key = PrivateKeyDer::from_pem_file(key).map_err(|e| format!("the key {}: {e}", key.display()))?;
    let mut cfg = rustls::ServerConfig::builder_with_provider(Arc::new(rustls_graviola::default_provider()))
        .with_safe_default_protocol_versions()
        .map_err(|e| e.to_string())?
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| format!("the certificate and its key: {e}"))?;
    cfg.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(Arc::new(cfg))
}

/// A TLS connection the server accepted. Read and write as the TCP one; dropped, it says close_notify, so a client
/// reading an answer to the connection's end (a stream) sees it end cleanly rather than cut.
///
/// One handle reads and writes the session (`Direct`) - what every request needs. A WebSocket passed on to a model's
/// server (`http::splice`: /v1/realtime) reads and writes at once from two threads, so `pump` hands the session to two
/// pump threads that move bytes between it and one end of an anonymous Unix socket pair, and the connection becomes
/// the other end (`Pumped`): an ordinary socket, cloned, shut down, read and written as TCP (after nextsycl's
/// glue/serve/src/tls.rs, where every TLS connection is pumped).
pub struct TlsConn {
    s: Side,
}

enum Side {
    Direct(rustls::StreamOwned<rustls::ServerConnection, TcpStream>),
    Pumped(UnixStream),
    /// between the two (never seen outside `pump`)
    Gone,
}

impl TlsConn {
    /// The handshake on an accepted connection (10 s at most)
    pub fn accept(sock: TcpStream, cfg: Arc<rustls::ServerConfig>) -> Result<TlsConn, String> {
        let mut conn = rustls::ServerConnection::new(cfg).map_err(|e| e.to_string())?;
        let mut sock = sock;
        let _ = sock.set_nodelay(true);
        sock.set_read_timeout(Some(Duration::from_secs(10))).map_err(|e| e.to_string())?;
        sock.set_write_timeout(Some(Duration::from_secs(10))).map_err(|e| e.to_string())?;
        while conn.is_handshaking() {
            conn.complete_io(&mut sock).map_err(|e| e.to_string())?;
        }
        // after it, the connection is as patient as a plain one
        sock.set_read_timeout(None).map_err(|e| e.to_string())?;
        sock.set_write_timeout(None).map_err(|e| e.to_string())?;
        Ok(TlsConn { s: Side::Direct(rustls::StreamOwned::new(conn, sock)) })
    }

    pub fn set_read_timeout(&self, d: Option<Duration>) -> io::Result<()> {
        match &self.s {
            Side::Direct(s) => s.sock.set_read_timeout(d),
            Side::Pumped(u) => u.set_read_timeout(d),
            Side::Gone => Ok(()),
        }
    }

    /// The session handed to two pump threads (see the type); the connection is then the plaintext end of a socket
    /// pair. What the client sent that the session has decrypted but nobody has read yet is passed on first.
    pub fn pump(&mut self) -> io::Result<()> {
        if !matches!(self.s, Side::Direct(_)) {
            return Ok(());
        }
        let Side::Direct(st) = std::mem::replace(&mut self.s, Side::Gone) else { unreachable!() };
        let rustls::StreamOwned { mut conn, sock } = st;
        // the pumps drain what they encrypt as they go: no limit needed on the session's own buffer
        conn.set_buffer_limit(None);
        let (app, pump) = UnixStream::pair()?;
        let shared = Arc::new(Session { conn: Mutex::new(conn), out: Mutex::new(sock.try_clone()?) });
        let (sh, tcp_in, pump_out) = (shared.clone(), sock.try_clone()?, pump.try_clone()?);
        std::thread::spawn(move || inbound(&sh, tcp_in, pump_out));
        std::thread::spawn(move || outbound(&shared, pump, sock));
        self.s = Side::Pumped(app);
        Ok(())
    }

    /// A second handle: a pumped connection only (one session cannot be written from two handles)
    pub fn try_clone(&self) -> io::Result<TlsConn> {
        match &self.s {
            Side::Pumped(u) => Ok(TlsConn { s: Side::Pumped(u.try_clone()?) }),
            _ => Err(io::Error::new(io::ErrorKind::Unsupported, "a TLS connection has one handle until it is pumped")),
        }
    }

    /// Both directions closed: a pumped one's pumps say close_notify and close the TCP connection
    pub fn shutdown(&self) {
        let _ = match &self.s {
            Side::Direct(s) => s.sock.shutdown(Shutdown::Both),
            Side::Pumped(u) => u.shutdown(Shutdown::Both),
            Side::Gone => Ok(()),
        };
    }
}

impl Read for TlsConn {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        match &mut self.s {
            Side::Direct(s) => match s.read(b) {
                // a client that closes without close_notify after its request: the end of what it sends
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(0),
                r => r,
            },
            Side::Pumped(u) => u.read(b),
            Side::Gone => Ok(0),
        }
    }
}

impl Write for TlsConn {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        match &mut self.s {
            Side::Direct(s) => s.write(b),
            Side::Pumped(u) => u.write(b),
            Side::Gone => Err(io::Error::from(io::ErrorKind::BrokenPipe)),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match &mut self.s {
            Side::Direct(s) => s.flush(),
            Side::Pumped(u) => u.flush(),
            Side::Gone => Ok(()),
        }
    }
}

impl Drop for TlsConn {
    fn drop(&mut self) {
        // a pumped one: its pumps end the session when the last handle of the pair's end is gone
        let Side::Direct(s) = &mut self.s else { return };
        s.conn.send_close_notify();
        let _ = s.sock.set_write_timeout(Some(Duration::from_secs(5)));
        while s.conn.wants_write() {
            if s.conn.write_tls(&mut s.sock).is_err() {
                break;
            }
        }
    }
}

/// One TLS session shared by its two pumps: the session itself, then the TCP connection's writing side (taken before
/// the session is let go, so records reach the wire in the order they were made)
struct Session {
    conn: Mutex<rustls::ServerConnection>,
    out: Mutex<TcpStream>,
}

impl Session {
    /// What the session has to send, sent: made under its lock, written under the writer's
    fn send(&self, mut c: std::sync::MutexGuard<'_, rustls::ServerConnection>) -> io::Result<()> {
        let mut bytes = Vec::new();
        while c.wants_write() {
            c.write_tls(&mut bytes)?;
        }
        if bytes.is_empty() {
            return Ok(());
        }
        let mut out = self.out.lock().unwrap_or_else(|e| e.into_inner());
        drop(c);
        out.write_all(&bytes)?;
        out.flush()
    }
}

/// The client's bytes: read from TCP, decrypted, written to the server's side; the end (EOF, close_notify) passed on
/// as the end of what the server reads
fn inbound(sh: &Session, mut tcp: TcpStream, mut app: UnixStream) {
    let mut buf = vec![0u8; 64 << 10];
    let (mut at, mut end) = (0usize, 0usize);
    let r: io::Result<()> = (|| loop {
        let mut plain = Vec::new();
        let mut closed = false;
        {
            let mut c = sh.conn.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                let mut fed = 0;
                if at < end {
                    fed = c.read_tls(&mut &buf[at..end])?;
                    at += fed;
                }
                let st = match c.process_new_packets() {
                    Ok(st) => st,
                    Err(e) => {
                        // the alert that says why, then the end
                        let _ = sh.send(c);
                        return Err(io::Error::new(io::ErrorKind::InvalidData, e));
                    }
                };
                let k = st.plaintext_bytes_to_read();
                if k > 0 {
                    let o = plain.len();
                    plain.resize(o + k, 0);
                    c.reader().read_exact(&mut plain[o..])?;
                }
                closed |= st.peer_has_closed();
                if at >= end || closed {
                    break;
                }
                if fed == 0 && k == 0 {
                    return Err(io::Error::other("the TLS session takes no more of what was read"));
                }
            }
            sh.send(c)?;
        }
        if !plain.is_empty() && app.write_all(&plain).is_err() {
            // the server's side has gone: the other pump ends the connection
            return Ok(());
        }
        if closed {
            return Ok(());
        }
        let n = tcp.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        (at, end) = (0, n);
    })();
    // what the server reads ends here (its answer can still go out); on an error the whole connection
    let _ = app.shutdown(Shutdown::Write);
    if r.is_err() {
        let _ = app.shutdown(Shutdown::Both);
        let _ = tcp.shutdown(Shutdown::Both);
    }
}

/// The server's bytes: read from its side, encrypted, sent; when it closes (or shuts down), close_notify and the TCP
/// connection closed
fn outbound(sh: &Session, mut app: UnixStream, tcp: TcpStream) {
    let mut buf = vec![0u8; 64 << 10];
    loop {
        let n = match app.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        let mut c = sh.conn.lock().unwrap_or_else(|e| e.into_inner());
        if c.writer().write_all(&buf[..n]).is_err() || sh.send(c).is_err() {
            // the client has gone: the server's next write fails as it would on TCP
            let _ = app.shutdown(Shutdown::Both);
            let _ = tcp.shutdown(Shutdown::Both);
            return;
        }
    }
    let mut c = sh.conn.lock().unwrap_or_else(|e| e.into_inner());
    c.send_close_notify();
    let _ = sh.send(c);
    let _ = tcp.shutdown(Shutdown::Both);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("nc-tls-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_certificate_is_made_with_its_names_then_kept() {
        let d = tmp();
        let (cert, key) = (d.join("tls/cert.pem"), d.join("tls/key.pem"));
        let sans: Vec<String> = ["localhost", "127.0.0.1", "brainiac-nvidia", "brainiac-nvidia.birman-prometheus.ts.net", "100.113.46.61", "::1"].map(String::from).to_vec();
        assert!(matches!(ensure_cert(&cert, &key, &sans).unwrap(), CertState::Made(_)));
        let info = read_cert(&std::fs::read(&cert).unwrap()).unwrap();
        assert!(info.ours);
        assert_eq!(info.sans, sans, "every name and address, as asked");
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
        assert!((3649..=3650).contains(&((info.not_after - now) / 86400)), "ten years");
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&key).unwrap().permissions().mode() & 0o777, 0o600, "the key is the owner's alone");
        // the key rustls loads is the one that signed: the server side builds from the pair
        server_config(&cert, &key).unwrap();
        // kept on the next start: the same bytes, so a browser's exception stays good
        let (c1, k1) = (std::fs::read(&cert).unwrap(), std::fs::read(&key).unwrap());
        assert_eq!(ensure_cert(&cert, &key, &sans).unwrap(), CertState::Kept);
        assert_eq!(ensure_cert(&cert, &key, &[]).unwrap(), CertState::Kept, "found names: an address that came later makes no new one");
        assert_eq!((std::fs::read(&cert).unwrap(), std::fs::read(&key).unwrap()), (c1, k1));
        // a configured name it lacks makes a new one
        let more: Vec<String> = sans.iter().cloned().chain(["studio.lan".to_string()]).collect();
        assert!(matches!(ensure_cert(&cert, &key, &more).unwrap(), CertState::Made(_)));
        assert!(read_cert(&std::fs::read(&cert).unwrap()).unwrap().sans.contains(&"studio.lan".to_string()));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_certificate_near_its_end_or_missing_is_made_anew() {
        let d = tmp();
        let (cert, key) = (d.join("cert.pem"), d.join("key.pem"));
        let sans = vec!["localhost".to_string()];
        ensure_cert_with(&cert, &key, &sans, 10).unwrap();
        let before = std::fs::read(&cert).unwrap();
        assert!(matches!(ensure_cert(&cert, &key, &sans).unwrap(), CertState::Made(r) if r.contains("ends in")));
        assert_ne!(std::fs::read(&cert).unwrap(), before);
        std::fs::remove_file(&key).unwrap();
        assert!(matches!(ensure_cert(&cert, &key, &sans).unwrap(), CertState::Made(_)));
        // a certificate of another organisation is used as it is
        let mut p = rcgen::CertificateParams::new(vec!["localhost".to_string()]).unwrap();
        p.not_after = time::OffsetDateTime::now_utc() + time::Duration::days(5);
        p.serial_number = Some(serial().unwrap());
        let k = GraviolaKey::generate().unwrap();
        let foreign = p.self_signed(&k).unwrap().pem();
        std::fs::write(&cert, &foreign).unwrap();
        assert_eq!(ensure_cert(&cert, &key, &sans).unwrap(), CertState::Foreign);
        assert_eq!(std::fs::read_to_string(&cert).unwrap(), foreign);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_found_names_start_with_the_loopback_ones() {
        assert_eq!(&wanted_sans(&[])[..3], ["localhost", "127.0.0.1", "::1"]);
        assert_eq!(wanted_sans(&[" a.lan ".into(), "".into(), "A.lan".into(), "10.0.0.2".into()]), ["a.lan", "10.0.0.2"]);
    }

    #[test]
    fn host_addresses_are_read_from_proc() {
        let trie = "Main:\n  +-- 0.0.0.0/0 3 0 5\n     |-- 0.0.0.0\n        /0 universe UNICAST\n     |-- 127.0.0.1\n        /32 host LOCAL\n     |-- 192.168.0.246\n        /32 host LOCAL\n     |-- 192.168.0.255\n        /32 link BROADCAST\n     |-- 100.113.46.61\n        /32 host LOCAL\n";
        assert_eq!(fib_trie_locals(trie), ["192.168.0.246", "100.113.46.61"]);
        let v6 = "fdd1e9ebb40370c8c20f5426e2a86645 02 40 00 01 enp1s0\nfdd1e9ebb40370c8505400fffe772f72 02 40 00 00 enp1s0\nfe80000000000000505400fffe772f72 02 40 20 80 enp1s0\n00000000000000000000000000000001 01 80 10 80 lo\n";
        assert_eq!(if_inet6_stable(v6), ["fdd1:e9eb:b403:70c8:5054:ff:fe77:2f72"]);
    }

    /// A WebSocket upgrade over https spliced on to a plain upstream (the front door to a model's /v1/realtime): the
    /// 101 back, then bytes both ways at once through the pumped session, and the end a clean close_notify (after
    /// nextsycl's https test's splice)
    #[test]
    fn an_upgrade_over_https_is_spliced_both_ways() {
        use crate::http::{self, Conn, Target};
        use rustls::pki_types::pem::PemObject;
        let d = tmp();
        let (cert, key) = (d.join("cert.pem"), d.join("key.pem"));
        ensure_cert(&cert, &key, &["localhost".to_string()]).unwrap();
        let cfg = server_config(&cert, &key).unwrap();
        // upstream: the 101, then each 5 bytes it reads upper-cased back, twice, then it closes
        let up_l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let up_addr = up_l.local_addr().unwrap().to_string();
        let upstream = std::thread::spawn(move || {
            let (mut c, _) = up_l.accept().unwrap();
            let got = http::read_request(&mut c).unwrap();
            c.write_all(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n").unwrap();
            for _ in 0..2 {
                let mut b = [0u8; 5];
                c.read_exact(&mut b).unwrap();
                c.write_all(&b.to_ascii_uppercase()).unwrap();
            }
            got
        });
        let door_l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let door_addr = door_l.local_addr().unwrap();
        let door = std::thread::spawn(move || {
            let (c, _) = door_l.accept().unwrap();
            let mut conn = Conn::Tls(Box::new(TlsConn::accept(c, cfg).unwrap()));
            let req = http::read_request(&mut conn).unwrap();
            assert!(http::is_upgrade(&req.head));
            http::splice(&mut conn, &Target::Tcp(up_addr), &req, "/v1/realtime?model=m").unwrap();
        });
        let mut roots = rustls::RootCertStore::empty();
        for c in rustls::pki_types::CertificateDer::pem_file_iter(&cert).unwrap() {
            roots.add(c.unwrap()).unwrap();
        }
        let ccfg = rustls::ClientConfig::builder_with_provider(Arc::new(rustls_graviola::default_provider()))
            .with_safe_default_protocol_versions().unwrap().with_root_certificates(roots).with_no_client_auth();
        let conn = rustls::ClientConnection::new(Arc::new(ccfg), "localhost".try_into().unwrap()).unwrap();
        let tcp = TcpStream::connect(door_addr).unwrap();
        tcp.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut s = rustls::StreamOwned::new(conn, tcp);
        s.write_all(b"GET /open/m/v1/realtime?model=m HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n").unwrap();
        let mut head = Vec::new();
        let mut b = [0u8; 1];
        while !head.ends_with(b"\r\n\r\n") {
            s.read_exact(&mut b).unwrap();
            head.push(b[0]);
        }
        assert!(head.starts_with(b"HTTP/1.1 101"), "{}", String::from_utf8_lossy(&head));
        s.write_all(b"hello").unwrap();
        let mut got = [0u8; 5];
        s.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"HELLO", "an answer while the socket stays open: both directions at once");
        s.write_all(b"again").unwrap();
        let mut rest = Vec::new();
        s.read_to_end(&mut rest).expect("the end, by close_notify");
        assert_eq!(rest, b"AGAIN");
        let req = upstream.join().unwrap();
        assert_eq!(req.target, "/v1/realtime?model=m");
        door.join().unwrap();
        let _ = std::fs::remove_dir_all(&d);
    }
}
