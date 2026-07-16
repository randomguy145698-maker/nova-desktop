// Nova's NATIVE neural network
// =============================
//
// This is a from-scratch feedforward network (bag-of-words input -> two
// hidden tanh layers -> softmax over topics), trained with plain gradient
// descent — the exact same algorithm Nova's browser version already runs
// in JavaScript, just written in compiled Rust instead of interpreted JS.
//
// Why this exists: the honest bottleneck in the browser version was never
// the MATH, it was the LANGUAGE — JavaScript in a webview has to interpret
// every multiply-add, and to avoid freezing the UI, training had to be
// sliced into tiny chunks spread across many animation frames. Compiled
// Rust doesn't have that problem, so this same algorithm can run at a
// network size that would be impractical in JS, and actually finish
// training in a reasonable amount of time.
//
// This is intentionally kept SEPARATE from Nova's existing lightweight
// JS router (used for every chat message) rather than replacing it —
// Tauri's JS<->Rust bridge is always asynchronous, and threading that
// through Nova's existing synchronous chat logic would be a much bigger,
// riskier rewrite for no real benefit (routing between a few dozen
// topics doesn't need millions of parameters). This network is instead
// an explicitly user-triggered "train a big model and see it think"
// feature — additive, not a replacement.
//
// Everything here uses only the Rust standard library (Vec<f32>, plain
// loops) — no ndarray/nalgebra/rand dependency — to keep this file's
// compile-time risk as low as possible.

use serde::{Deserialize, Serialize};
use std::sync::Mutex;

const STOP_WORDS: &[&str] = &[
    "what", "is", "the", "a", "an", "who", "how", "do", "does", "are", "was",
    "were", "in", "of", "to", "it", "i", "you", "and", "or", "but", "with",
    "me", "my", "tell", "about", "on", "for", "that", "this", "can", "using",
    "whats", "its", "mean", "means", "meaning", "define", "definition",
    "word", "wassup", "sup",
];

// --- tiny helpers, mirroring the browser version's text processing -----

fn clean(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| !"?!.,;:'\"".contains(*c))
        .collect()
}

fn stem(word: &str) -> String {
    if word.len() > 3 && word.ends_with('s') {
        word[..word.len() - 1].to_string()
    } else {
        word.to_string()
    }
}

fn keywords(text: &str) -> Vec<String> {
    clean(text)
        .split_whitespace()
        .filter(|w| !w.is_empty() && !STOP_WORDS.contains(w))
        .map(stem)
        .collect()
}

/// The exact same rolling hash the browser version uses, so words land
/// in the same bucket whether Nova is thinking in JS or in Rust.
fn hash_word(word: &str, dim: usize) -> usize {
    let mut h: u32 = 0;
    for b in word.bytes() {
        h = h.wrapping_mul(131).wrapping_add(b as u32);
    }
    (h as usize) % dim
}

/// A minimal, dependency-free pseudo-random generator (xorshift) — used
/// only for weight initialization and training-time dropout, so we don't
/// need to pull in the `rand` crate for this.
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

fn featurize(tokens: &[String], dim: usize, dropout: f32, rng: &mut Rng) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    for w in tokens {
        if dropout > 0.0 && rng.next_f32() < dropout {
            continue;
        }
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

// --- the network itself -------------------------------------------------

pub struct Network {
    pub dim: usize,
    pub h1: usize,
    pub h2: usize,
    pub classes: Vec<String>,
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    w3: Vec<Vec<f32>>,
    b3: Vec<f32>,
}

impl Network {
    fn new(dim: usize, h1: usize, h2: usize, classes: Vec<String>, rng: &mut Rng) -> Self {
        let mat = |rows: usize, cols: usize, rng: &mut Rng| -> Vec<Vec<f32>> {
            (0..rows)
                .map(|_| (0..cols).map(|_| rng.signed()).collect())
                .collect()
        };
        let n_classes = classes.len();
        Network {
            dim,
            h1,
            h2,
            classes,
            w1: mat(h1, dim, rng),
            b1: vec![0f32; h1],
            w2: mat(h2, h1, rng),
            b2: vec![0f32; h2],
            w3: mat(n_classes, h2, rng),
            b3: vec![0f32; n_classes],
        }
    }

    fn param_count(&self) -> usize {
        self.h1 * self.dim + self.h1 + self.h2 * self.h1 + self.h2
            + self.classes.len() * self.h2 + self.classes.len()
    }

    /// Forward pass. Returns (hidden1 activations, hidden2 activations,
    /// output probabilities) — the hidden activations are needed by
    /// backprop, so we hand them all back together.
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
        let n = self.classes.len();
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

    /// One step of backpropagation (softmax + cross-entropy at the
    /// output, tanh derivative through both hidden layers), updating
    /// weights in place — same shape as the browser version's nnStep.
    fn train_step(&mut self, x: &[f32], target: usize, lr: f32) {
        let (a1, a2, p) = self.forward(x);
        let n_classes = self.classes.len();

        let mut dz = p.clone();
        dz[target] -= 1.0;

        let mut da2 = vec![0f32; self.h2];
        for c in 0..n_classes {
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

// --- the Tauri-facing API -------------------------------------------------

#[derive(Deserialize)]
pub struct FactInput {
    pub topic: String,
    pub question: String,
    pub answer: String,
}

#[derive(Serialize, Clone)]
pub struct TrainResult {
    pub trained: bool,
    pub params: usize,
    pub dim: usize,
    pub h1: usize,
    pub h2: usize,
    pub classes: usize,
    pub accuracy: u32,
    pub trained_on: usize,
    pub message: String,
}

#[derive(Serialize)]
pub struct Prediction {
    pub topic: String,
    pub confidence: u32,
}

/// Holds the currently-trained native network (if any) for the app's
/// lifetime. Retrained from scratch each time `train_native_network` is
/// called — same "retrain on demand" model the browser version uses.
pub struct NetState(pub Mutex<Option<Network>>);

fn tier_dims(size_mode: &str) -> (usize, usize, usize) {
    // (input_dim, hidden1, hidden2) — deliberately much larger than the
    // browser's JS tiers for the top options, since native code can
    // actually finish training them in a reasonable time.
    match size_mode {
        "large" => (4096, 1024, 512),   // ~4.7M parameters
        "ultra" => (8192, 2048, 1024),  // ~19M parameters
        _ => (2048, 512, 256),          // "standard" — ~1.2M parameters
    }
}

#[tauri::command]
pub fn train_native_network(
    facts: Vec<FactInput>,
    size_mode: String,
    state: tauri::State<NetState>,
) -> TrainResult {
    // Group facts by topic, exactly like the browser version's classifier.
    let mut by_topic: std::collections::BTreeMap<String, Vec<(String, String)>> =
        std::collections::BTreeMap::new();
    for f in facts {
        by_topic
            .entry(f.topic)
            .or_insert_with(Vec::new)
            .push((f.question, f.answer));
    }
    // Only topics with at least 2 facts are worth classifying between.
    by_topic.retain(|_, v| v.len() >= 2);

    let classes: Vec<String> = by_topic.keys().cloned().collect();
    let total: usize = by_topic.values().map(|v| v.len()).sum();

    if classes.len() < 2 || total < 10 {
        return TrainResult {
            trained: false,
            params: 0,
            dim: 0,
            h1: 0,
            h2: 0,
            classes: 0,
            accuracy: 0,
            trained_on: 0,
            message: "Need at least 2 topics with a few facts each before training.".into(),
        };
    }

    let (dim, h1, h2) = tier_dims(&size_mode);
    let mut rng = Rng::new(0xC0FFEE ^ total as u64);
    let mut net = Network::new(dim, h1, h2, classes.clone(), &mut rng);

    // Build the training set: (token list, class index) pairs, capped per
    // topic the same way the browser version caps at 200 to keep this
    // bounded for extremely large brains.
    let mut samples: Vec<(Vec<String>, usize)> = Vec::new();
    for (ti, topic) in classes.iter().enumerate() {
        let facts_for_topic = &by_topic[topic];
        for (q, a) in facts_for_topic.iter().take(200) {
            let mut text = q.clone();
            text.push(' ');
            text.push_str(&a.chars().take(120).collect::<String>());
            samples.push((keywords(&text), ti));
        }
    }

    let epochs = ((60_000usize / samples.len().max(1)).clamp(10, 28)) as f32;
    let epochs_n = epochs as usize;
    for ep in 0..epochs_n {
        let lr = 0.15 * (1.0 - (ep as f32) / epochs) + 0.02;
        // Fisher-Yates shuffle using our own tiny RNG.
        for i in (1..samples.len()).rev() {
            let j = (rng.next_f32() * ((i + 1) as f32)) as usize;
            samples.swap(i, j.min(i));
        }
        for (toks, y) in samples.iter() {
            let x = featurize(toks, dim, 0.3, &mut rng);
            net.train_step(&x, *y, lr);
        }
    }

    // Self-test accuracy on (a sample of, if huge) the training data,
    // without dropout — same honesty check the browser version reports.
    let graded: Vec<&(Vec<String>, usize)> = if samples.len() > 300 {
        samples.iter().step_by(samples.len() / 300).collect()
    } else {
        samples.iter().collect()
    };
    let mut hits = 0usize;
    for (toks, y) in graded.iter() {
        let x = featurize(toks, dim, 0.0, &mut rng);
        let (_, _, p) = net.forward(&x);
        let mut best = 0usize;
        for c in 1..p.len() {
            if p[c] > p[best] {
                best = c;
            }
        }
        if best == *y {
            hits += 1;
        }
    }
    let accuracy = ((hits as f32 / graded.len().max(1) as f32) * 100.0).round() as u32;
    let params = net.param_count();
    let n_classes = net.classes.len();

    *state.0.lock().unwrap() = Some(net);

    TrainResult {
        trained: true,
        params,
        dim,
        h1,
        h2,
        classes: n_classes,
        accuracy,
        trained_on: total,
        message: format!(
            "Trained a native network: {} parameters ({} inputs -> {} -> {} -> {} topics), {}% self-test accuracy on {} facts.",
            params, dim, h1, h2, n_classes, accuracy, total
        ),
    }
}

#[tauri::command]
pub fn predict_native_topic(text: String, state: tauri::State<NetState>) -> Option<Prediction> {
    let guard = state.0.lock().unwrap();
    let net = guard.as_ref()?;
    let toks = keywords(&text);
    if toks.is_empty() {
        return None;
    }
    let mut rng = Rng::new(1); // dropout is 0 here, so the RNG is never actually drawn from
    let x = featurize(&toks, net.dim, 0.0, &mut rng);
    let (_, _, p) = net.forward(&x);
    let mut best = 0usize;
    for c in 1..p.len() {
        if p[c] > p[best] {
            best = c;
        }
    }
    if p[best] < 0.35 {
        return None;
    }
    Some(Prediction {
        topic: net.classes[best].clone(),
        confidence: (p[best] * 100.0).round() as u32,
    })
}
