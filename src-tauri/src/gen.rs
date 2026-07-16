// Nova's NATIVE text/code generator
// ===================================
//
// This reuses the same feedforward network shape as nn.rs (hashed
// context -> tanh -> tanh -> softmax), but repurposed to predict the
// NEXT WORD given recent context. Trained over every fact Nova knows,
// this is a genuine (if small) neural language model.
//
// Upgrades in this version, aimed specifically at "noticeably better,
// longer text":
//   - 3-word context window instead of 2 — more context per prediction
//     means less repetition and more coherent word choices.
//   - Top-K sampling — instead of sampling from the ENTIRE vocabulary's
//     probabilities (which occasionally picks a bizarre low-probability
//     word), only the K most likely next words are considered at all.
//     This is a standard, well-understood technique for improving
//     generation quality without making the model more deterministic.
//   - Multi-sentence output — generation now runs long enough to
//     produce several sentences, with a heuristic sentence-break (our
//     vocabulary strips punctuation, so there's no literal "period"
//     token to predict; instead a period is inserted every ~10-16 words
//     and the next word is capitalized, faking sentence structure in an
//     honest, clearly-a-heuristic way rather than pretending the model
//     learned punctuation it was never shown).
//
// This remains deliberately separate from Nova's existing lightweight
// JS router and deterministic code builder — purely additive.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

fn clean(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| !"?!,;:'\"".contains(*c))
        .collect()
}

fn hash_word(word: &str, dim: usize) -> usize {
    let mut h: u32 = 0;
    for b in word.bytes() {
        h = h.wrapping_mul(131).wrapping_add(b as u32);
    }
    (h as usize) % dim
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 % 1_000_000) as f32) / 1_000_000.0
    }
    fn signed(&mut self) -> f32 {
        (self.next_f32() - 0.5) * 0.4
    }
}

const CONTEXT_WORDS: usize = 3;

fn featurize_context(words: &[&str], dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    for w in words {
        let idx = hash_word(w, dim);
        v[idx] += 1.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

pub struct GenNetwork {
    dim: usize,
    h1: usize,
    h2: usize,
    vocab: Vec<String>,
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    w3: Vec<Vec<f32>>,
    b3: Vec<f32>,
}

impl GenNetwork {
    fn new(dim: usize, h1: usize, h2: usize, vocab: Vec<String>, rng: &mut Rng) -> Self {
        let mat = |rows: usize, cols: usize, rng: &mut Rng| -> Vec<Vec<f32>> {
            (0..rows).map(|_| (0..cols).map(|_| rng.signed()).collect()).collect()
        };
        let n = vocab.len();
        GenNetwork {
            dim, h1, h2, vocab,
            w1: mat(h1, dim, rng), b1: vec![0f32; h1],
            w2: mat(h2, h1, rng), b2: vec![0f32; h2],
            w3: mat(n, h2, rng), b3: vec![0f32; n],
        }
    }

    fn param_count(&self) -> usize {
        self.h1 * self.dim + self.h1 + self.h2 * self.h1 + self.h2
            + self.vocab.len() * self.h2 + self.vocab.len()
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut a1 = vec![0f32; self.h1];
        for j in 0..self.h1 {
            let mut s = self.b1[j];
            for i in 0..self.dim {
                if x[i] != 0.0 {
                    s += self.w1[j][i] * x[i];
                }
            }
            a1[j] = s.tanh();
        }
        let mut a2 = vec![0f32; self.h2];
        for k in 0..self.h2 {
            let mut s = self.b2[k];
            for j in 0..self.h1 {
                s += self.w2[k][j] * a1[j];
            }
            a2[k] = s.tanh();
        }
        let n = self.vocab.len();
        let mut z = vec![0f32; n];
        let mut max_z = f32::MIN;
        for c in 0..n {
            let mut s = self.b3[c];
            for k in 0..self.h2 {
                s += self.w3[c][k] * a2[k];
            }
            z[c] = s;
            if s > max_z {
                max_z = s;
            }
        }
        let mut sum = 0f32;
        let mut p = vec![0f32; n];
        for c in 0..n {
            p[c] = (z[c] - max_z).exp();
            sum += p[c];
        }
        if sum > 0.0 {
            for c in 0..n {
                p[c] /= sum;
            }
        }
        (a1, a2, p)
    }

    fn train_step(&mut self, x: &[f32], target: usize, lr: f32) {
        let (a1, a2, p) = self.forward(x);
        let n = self.vocab.len();

        let mut dz = p.clone();
        dz[target] -= 1.0;

        let mut da2 = vec![0f32; self.h2];
        for c in 0..n {
            for k in 0..self.h2 {
                da2[k] += self.w3[c][k] * dz[c];
                self.w3[c][k] -= lr * dz[c] * a2[k];
            }
            self.b3[c] -= lr * dz[c];
        }

        let mut da1 = vec![0f32; self.h1];
        for k in 0..self.h2 {
            let g = da2[k] * (1.0 - a2[k] * a2[k]);
            for j in 0..self.h1 {
                da1[j] += self.w2[k][j] * g;
                self.w2[k][j] -= lr * g * a1[j];
            }
            self.b2[k] -= lr * g;
        }

        for j in 0..self.h1 {
            let g = da1[j] * (1.0 - a1[j] * a1[j]);
            for i in 0..self.dim {
                if x[i] != 0.0 {
                    self.w1[j][i] -= lr * g * x[i];
                }
            }
            self.b1[j] -= lr * g;
        }
    }
}

pub struct GenState(pub Mutex<Option<GenNetwork>>);

#[derive(Deserialize)]
#[allow(dead_code)] // topic isn't used by this trainer, only question+answer text
pub struct FactInput {
    pub topic: String,
    pub question: String,
    pub answer: String,
}

#[derive(Serialize)]
pub struct GenTrainResult {
    pub trained: bool,
    pub params: usize,
    pub vocab_size: usize,
    pub trained_on_words: usize,
    pub message: String,
    pub sample: String,
}

#[derive(Serialize)]
pub struct GenOutput {
    pub text: String,
}

fn tier_dims(size_mode: &str) -> (usize, usize, usize, usize) {
    match size_mode {
        "large" => (1024, 512, 256, 1200),
        "ultra" => (2048, 1024, 512, 2500),
        _ => (512, 256, 128, 600),
    }
}

fn tokenize_natural(text: &str) -> Vec<String> {
    clean(text)
        .split_whitespace()
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Top-K + temperature sampling: keep only the K most likely next words,
/// renormalize just those, then sample. This is what keeps generation
/// from occasionally blurting out a bizarre, near-zero-probability word
/// — a real, standard technique for improving output quality.
fn sample_top_k(p: &[f32], k: usize, temperature: f32, rng: &mut Rng) -> usize {
    let mut idxs: Vec<usize> = (0..p.len()).collect();
    idxs.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap_or(std::cmp::Ordering::Equal));
    idxs.truncate(k.max(1));

    let t = temperature.max(0.05);
    let mut adj: Vec<f32> = idxs.iter().map(|&i| p[i].max(1e-9).powf(1.0 / t)).collect();
    let sum: f32 = adj.iter().sum();
    if sum > 0.0 {
        for x in adj.iter_mut() {
            *x /= sum;
        }
    }
    let r = rng.next_f32();
    let mut acc = 0f32;
    for (pos, &prob) in adj.iter().enumerate() {
        acc += prob;
        if r <= acc {
            return idxs[pos];
        }
    }
    *idxs.last().unwrap_or(&0)
}

#[tauri::command]
pub fn train_native_generator(
    facts: Vec<FactInput>,
    size_mode: String,
    state: tauri::State<GenState>,
) -> GenTrainResult {
    let mut all_text = String::new();
    for f in &facts {
        all_text.push_str(&f.question);
        all_text.push_str(". ");
        all_text.push_str(&f.answer);
        all_text.push_str(". ");
    }
    let tokens = tokenize_natural(&all_text);

    if tokens.len() < 200 {
        return GenTrainResult {
            trained: false, params: 0, vocab_size: 0, trained_on_words: tokens.len(),
            message: "Not enough reading material yet — teach Nova more, or run a Deep Study, then try again.".into(),
            sample: String::new(),
        };
    }

    let (dim, h1, h2, vocab_cap) = tier_dims(&size_mode);
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for w in &tokens {
        *freq.entry(w.as_str()).or_insert(0) += 1;
    }
    let mut by_freq: Vec<(&str, usize)> = freq.into_iter().collect();
    by_freq.sort_by(|a, b| b.1.cmp(&a.1));
    let vocab: Vec<String> = by_freq.into_iter().take(vocab_cap).map(|(w, _)| w.to_string()).collect();
    let vocab_index: HashMap<&str, usize> = vocab.iter().enumerate().map(|(i, w)| (w.as_str(), i)).collect();

    let mut rng = Rng::new(0xBEEF ^ tokens.len() as u64);
    let mut net = GenNetwork::new(dim, h1, h2, vocab.clone(), &mut rng);

    // Training samples now use a 3-word sliding context window (up from
    // 2) — one extra word of history per prediction, meaningfully
    // reducing "it just repeats the same two words forever" behavior.
    let mut samples: Vec<(Vec<String>, usize)> = Vec::new();
    for i in CONTEXT_WORDS..tokens.len() {
        if let Some(&target) = vocab_index.get(tokens[i].as_str()) {
            let ctx = tokens[i - CONTEXT_WORDS..i].to_vec();
            samples.push((ctx, target));
        }
    }
    if samples.len() > 20_000 {
        samples.truncate(20_000);
    }
    if samples.is_empty() {
        return GenTrainResult {
            trained: false, params: 0, vocab_size: vocab.len(), trained_on_words: tokens.len(),
            message: "Couldn't build any training examples from what Nova knows yet.".into(),
            sample: String::new(),
        };
    }

    let epochs = ((40_000usize / samples.len().max(1)).clamp(6, 20)) as f32;
    let epochs_n = epochs as usize;
    for ep in 0..epochs_n {
        let lr = 0.12 * (1.0 - (ep as f32) / epochs) + 0.015;
        for i in (1..samples.len()).rev() {
            let j = (rng.next_f32() * ((i + 1) as f32)) as usize;
            samples.swap(i, j.min(i));
        }
        for (ctx, y) in samples.iter() {
            let words: Vec<&str> = ctx.iter().map(|s| s.as_str()).collect();
            let x = featurize_context(&words, dim);
            net.train_step(&x, *y, lr);
        }
    }

    let params = net.param_count();
    let vocab_size = net.vocab.len();
    let sample = generate_from_network(&net, None, 40, 0.85, &mut rng);

    *state.0.lock().unwrap() = Some(net);

    GenTrainResult {
        trained: true, params, vocab_size, trained_on_words: tokens.len(),
        message: format!(
            "Trained a native generator: {} parameters, {}-word vocabulary, {}-word context window, learned from {} words Nova has read.",
            params, vocab_size, CONTEXT_WORDS, tokens.len()
        ),
        sample,
    }
}

fn generate_from_network(
    net: &GenNetwork,
    seed: Option<&str>,
    max_words: usize,
    temperature: f32,
    rng: &mut Rng,
) -> String {
    let start_word = seed
        .map(|s| s.to_lowercase())
        .filter(|s| net.vocab.contains(s))
        .unwrap_or_else(|| net.vocab[(rng.next_f32() * net.vocab.len() as f32) as usize].clone());
    let mut out: Vec<String> = vec![start_word];
    while out.len() < CONTEXT_WORDS {
        out.push(net.vocab[(rng.next_f32() * net.vocab.len() as f32) as usize].clone());
    }

    // Every ~10-16 words, insert an artificial sentence break: a period
    // plus a capital letter on the next word. Our vocabulary has no
    // punctuation tokens at all (they're stripped before training), so
    // this is an honest, visible heuristic rather than the model having
    // "learned" where sentences end.
    let mut next_break = 10 + (rng.next_f32() * 6.0) as usize;
    let top_k = (net.vocab.len() / 12).clamp(5, 40);

    for i in 0..max_words {
        let ctx: Vec<&str> = out[out.len() - CONTEXT_WORDS..].iter().map(|s| s.as_str()).collect();
        let x = featurize_context(&ctx, net.dim);
        let (_, _, p) = net.forward(&x);
        let idx = sample_top_k(&p, top_k, temperature, rng);
        let mut word = net.vocab[idx].clone();
        if i + 1 == next_break && i + 1 < max_words {
            word.push('.');
            next_break = i + 1 + 10 + (rng.next_f32() * 6.0) as usize;
        }
        out.push(word);
    }

    // Capitalize the very first word and the first word after every
    // artificial period, then tidy up spacing around the inserted dots.
    let mut capitalize_next = true;
    let mut words_out: Vec<String> = Vec::with_capacity(out.len());
    for w in out {
        let mut w = w;
        if capitalize_next {
            if let Some(first) = w.get_mut(0..1) {
                first.make_ascii_uppercase();
            }
        }
        capitalize_next = w.ends_with('.');
        words_out.push(w);
    }
    let mut sentence = words_out.join(" ").replace(" .", ".");
    if !sentence.ends_with('.') {
        sentence.push('.');
    }
    sentence
}

#[tauri::command]
pub fn generate_native_text(
    seed: Option<String>,
    max_words: Option<u32>,
    state: tauri::State<GenState>,
) -> Option<GenOutput> {
    let guard = state.0.lock().unwrap();
    let net = guard.as_ref()?;
    let mut rng = Rng::new(42 ^ std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1));
    let text = generate_from_network(net, seed.as_deref(), max_words.unwrap_or(40) as usize, 0.85, &mut rng);
    Some(GenOutput { text })
}
