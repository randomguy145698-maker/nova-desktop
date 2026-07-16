// Nova's NATIVE word embeddings — real semantic understanding
// ==============================================================
//
// Everything Nova has done with words so far (the classifier, the
// generator) treats words as opaque tokens: "car" and "automobile" share
// zero letters in common and are, as far as hash-based bag-of-words is
// concerned, completely unrelated. This module fixes that.
//
// It trains a small skip-gram model (the same core idea behind
// word2vec): for every word, predict the words that tend to appear near
// it. Words that show up in similar contexts end up with similar vector
// representations — Nova starts to understand that "car" and
// "automobile" mean roughly the same thing, purely from how they're
// used in its own reading, without ever being told so directly.
//
// This is what actually lets `semantic_rank` find a fact even when the
// question shares no exact words with it — genuine meaning-based
// matching, not just typo-tolerant string comparison.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

fn clean(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| !"?!.,;:'\"".contains(*c))
        .collect()
}

fn tokenize(text: &str) -> Vec<String> {
    clean(text)
        .split_whitespace()
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .collect()
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
    fn signed(&mut self, scale: f32) -> f32 {
        (self.next_f32() - 0.5) * scale
    }
}

pub struct Embeddings {
    dim: usize,
    vocab: Vec<String>,
    index: HashMap<String, usize>,
    // "in" vectors (what a word IS) and "out" vectors (what a word
    // PREDICTS as context) — standard skip-gram uses two separate
    // matrices; after training we only need the "in" vectors for
    // similarity/search.
    in_vecs: Vec<Vec<f32>>,
    out_vecs: Vec<Vec<f32>>,
}

impl Embeddings {
    fn vector_for(&self, word: &str) -> Option<&Vec<f32>> {
        self.index.get(word).map(|&i| &self.in_vecs[i])
    }

    /// Average embedding of every known word in a piece of text — a
    /// simple, effective way to turn a whole sentence into one vector.
    fn text_vector(&self, text: &str) -> Option<Vec<f32>> {
        let mut sum = vec![0f32; self.dim];
        let mut count = 0;
        for w in tokenize(text) {
            if let Some(v) = self.vector_for(&w) {
                for i in 0..self.dim {
                    sum[i] += v[i];
                }
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        for x in sum.iter_mut() {
            *x /= count as f32;
        }
        Some(sum)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

pub struct EmbedState(pub Mutex<Option<Embeddings>>);

#[derive(Deserialize)]
#[allow(dead_code)] // topic isn't used by this trainer, only question+answer text
pub struct FactInput {
    pub topic: String,
    pub question: String,
    pub answer: String,
}

#[derive(Serialize)]
pub struct EmbedTrainResult {
    pub trained: bool,
    pub vocab_size: usize,
    pub dim: usize,
    pub pairs_trained: usize,
    pub message: String,
}

fn tier_dims(size_mode: &str) -> (usize, usize) {
    // (embedding dimension, vocabulary cap)
    match size_mode {
        "large" => (64, 1500),
        "ultra" => (96, 3000),
        _ => (32, 700), // "standard"
    }
}

#[tauri::command]
pub fn train_native_embeddings(
    facts: Vec<FactInput>,
    size_mode: String,
    state: tauri::State<EmbedState>,
) -> EmbedTrainResult {
    let mut all_text = String::new();
    for f in &facts {
        all_text.push_str(&f.question);
        all_text.push(' ');
        all_text.push_str(&f.answer);
        all_text.push(' ');
    }
    let tokens = tokenize(&all_text);

    if tokens.len() < 100 {
        return EmbedTrainResult {
            trained: false,
            vocab_size: 0,
            dim: 0,
            pairs_trained: 0,
            message: "Not enough reading material yet — teach Nova more, or run a Deep Study, then try again.".into(),
        };
    }

    let (dim, vocab_cap) = tier_dims(&size_mode);
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for w in &tokens {
        *freq.entry(w.as_str()).or_insert(0) += 1;
    }
    let mut by_freq: Vec<(&str, usize)> = freq.into_iter().collect();
    by_freq.sort_by(|a, b| b.1.cmp(&a.1));
    let vocab: Vec<String> = by_freq.into_iter().take(vocab_cap).map(|(w, _)| w.to_string()).collect();
    let index: HashMap<String, usize> = vocab.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect();
    let n = vocab.len();

    let mut rng = Rng::new(0xE33ED ^ tokens.len() as u64);
    let mut in_vecs: Vec<Vec<f32>> = (0..n).map(|_| (0..dim).map(|_| rng.signed(0.6)).collect()).collect();
    let mut out_vecs: Vec<Vec<f32>> = (0..n).map(|_| (0..dim).map(|_| rng.signed(0.6)).collect()).collect();

    // Build skip-gram (center, context) pairs: for every word, its
    // neighbors within a small window are the "context" it should help
    // predict. This is the entire supervision signal — no labels needed,
    // Nova's own text teaches the embeddings.
    let window = 2;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..tokens.len() {
        let center = match index.get(tokens[i].as_str()) {
            Some(&c) => c,
            None => continue,
        };
        let lo = i.saturating_sub(window);
        let hi = (i + window + 1).min(tokens.len());
        for j in lo..hi {
            if j == i {
                continue;
            }
            if let Some(&ctx) = index.get(tokens[j].as_str()) {
                pairs.push((center, ctx));
            }
        }
    }
    if pairs.len() > 40_000 {
        pairs.truncate(40_000);
    }
    if pairs.is_empty() {
        return EmbedTrainResult {
            trained: false,
            vocab_size: n,
            dim,
            pairs_trained: 0,
            message: "Couldn't build any training pairs from what Nova knows yet.".into(),
        };
    }

    // Train with simple negative sampling: for each real (center,context)
    // pair, also push AWAY from a handful of random "negative" words that
    // did NOT appear nearby. This is what keeps unrelated words from
    // drifting toward the same vector, and is dramatically cheaper than
    // a full softmax over the whole vocabulary every step.
    let neg_samples = 5;
    let epochs = ((30_000usize / pairs.len().max(1)).clamp(4, 12)) as f32;
    let epochs_n = epochs as usize;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());

    for ep in 0..epochs_n {
        let lr = 0.05 * (1.0 - (ep as f32) / epochs) + 0.005;
        for i in (1..pairs.len()).rev() {
            let j = (rng.next_f32() * ((i + 1) as f32)) as usize;
            pairs.swap(i, j.min(i));
        }
        for &(center, ctx) in pairs.iter() {
            // Positive example: push center/context vectors together.
            let dot: f32 = (0..dim).map(|d| in_vecs[center][d] * out_vecs[ctx][d]).sum();
            let err = sigmoid(dot) - 1.0; // target label = 1 (they DO co-occur)
            for d in 0..dim {
                let in_val = in_vecs[center][d];
                let out_val = out_vecs[ctx][d];
                in_vecs[center][d] -= lr * err * out_val;
                out_vecs[ctx][d] -= lr * err * in_val;
            }
            // Negative examples: push center away from random unrelated words.
            for _ in 0..neg_samples {
                let neg = (rng.next_f32() * n as f32) as usize;
                if neg == ctx {
                    continue;
                }
                let dot: f32 = (0..dim).map(|d| in_vecs[center][d] * out_vecs[neg][d]).sum();
                let err = sigmoid(dot) - 0.0; // target label = 0 (they don't co-occur)
                for d in 0..dim {
                    let in_val = in_vecs[center][d];
                    let out_val = out_vecs[neg][d];
                    in_vecs[center][d] -= lr * err * out_val;
                    out_vecs[neg][d] -= lr * err * in_val;
                }
            }
        }
    }

    let pairs_trained = pairs.len();
    *state.0.lock().unwrap() = Some(Embeddings { dim, vocab: vocab.clone(), index, in_vecs, out_vecs });

    EmbedTrainResult {
        trained: true,
        vocab_size: n,
        dim,
        pairs_trained,
        message: format!(
            "Trained {}-dimensional embeddings for {} words from {} word-pairs — Nova can now match meaning, not just exact words.",
            dim, n, pairs_trained
        ),
    }
}

#[derive(Deserialize)]
pub struct Candidate {
    pub key: String,
    pub text: String,
}

#[derive(Serialize)]
pub struct RankedCandidate {
    pub key: String,
    pub score: f32,
}

#[tauri::command]
pub fn semantic_rank(
    query: String,
    candidates: Vec<Candidate>,
    state: tauri::State<EmbedState>,
) -> Vec<RankedCandidate> {
    let guard = state.0.lock().unwrap();
    let emb = match guard.as_ref() {
        Some(e) => e,
        None => return Vec::new(),
    };
    let qv = match emb.text_vector(&query) {
        Some(v) => v,
        None => return Vec::new(),
    };
    let mut ranked: Vec<RankedCandidate> = candidates
        .iter()
        .filter_map(|c| {
            emb.text_vector(&c.text).map(|cv| RankedCandidate {
                key: c.key.clone(),
                score: cosine(&qv, &cv),
            })
        })
        .collect();
    ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(5);
    ranked
}

#[derive(Serialize)]
pub struct NearestWord {
    pub word: String,
    pub similarity: f32,
}

#[tauri::command]
pub fn nearest_words(word: String, state: tauri::State<EmbedState>) -> Vec<NearestWord> {
    let guard = state.0.lock().unwrap();
    let emb = match guard.as_ref() {
        Some(e) => e,
        None => return Vec::new(),
    };
    let target = match emb.vector_for(&word.to_lowercase()) {
        Some(v) => v.clone(),
        None => return Vec::new(),
    };
    let mut scored: Vec<NearestWord> = emb
        .vocab
        .iter()
        .filter(|w| **w != word.to_lowercase())
        .map(|w| NearestWord {
            word: w.clone(),
            similarity: cosine(&target, emb.vector_for(w).unwrap()),
        })
        .collect();
    scored.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(8);
    scored
}
