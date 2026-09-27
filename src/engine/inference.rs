//! ONNX モデル (src/train/export.py が書き出したもの) の推論。

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::Context;
use ort::ep;
use ort::session::Session;
use ort::value::Tensor;

use crate::board::{Board, CELLS, SIZE};
use crate::config::{Device, InferenceConfig};
use crate::features::{FEATURE_LEN, PLANES, encode};
use crate::mcts::{Evaluation, Evaluator};

/// CUDA 実行プロバイダが使うライブラリ。依存される側から順に読み込む。
const CUDA_DYLIBS: &[&str] = &[
    "libcudart.so.13",
    "libcublasLt.so.13",
    "libcublas.so.13",
    "libnvrtc.so.13",
    "libcurand.so.10",
    "libcufft.so.12",
    "libcudnn.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
];

/// lib_dirs にある CUDA / cuDNN を先に読み込む (Python の venv 同梱のものを使うため)。
/// 見つからないライブラリはシステムの検索パスに任せる。
pub fn preload_cuda(lib_dirs: &[PathBuf]) -> anyhow::Result<()> {
    for name in CUDA_DYLIBS {
        if let Some(path) = lib_dirs.iter().map(|d| d.join(name)).find(|p| p.exists()) {
            ort::util::preload_dylib(&path).with_context(|| format!("{} を読み込めません", path.display()))?;
        }
    }
    Ok(())
}

const BATCH_ALIGN: usize = 32;

pub struct OnnxModel {
    session: Session,
}

impl OnnxModel {
    pub fn load(path: &Path, device: Device) -> anyhow::Result<Self> {
        let mut builder = Session::builder()?;
        if device == Device::Cuda {
            builder = builder
                .with_execution_providers([ep::CUDA::default().build().error_on_failure()])
                .map_err(|e| anyhow::anyhow!("CUDA を有効にできません: {e}"))?;
        }
        let session = builder.commit_from_file(path).with_context(|| format!("{} を読み込めません", path.display()))?;
        let mut model = Self { session };
        // GPU で実際に動くかは推論してみるまで分からない (例: GPU の世代向けのカーネルがない) ので、ここで一度試す
        model.run(&[0; FEATURE_LEN]).with_context(|| {
            format!(
                "{device:?} での試し推論に失敗しました。onnxruntime の CUDA 版がこの GPU の世代 \
                 (compute capability) に対応していない可能性があります"
            )
        })?;
        Ok(model)
    }

    /// features は n 局面分の特徴平面 (n * FEATURE_LEN)。
    pub fn run(&mut self, features: &[u8]) -> anyhow::Result<Vec<Evaluation>> {
        let n = features.len() / FEATURE_LEN;
        // cuDNN は入力の形ごとに畳み込みアルゴリズムを探索し直すので、バッチを BATCH_ALIGN の倍数に揃えて形の種類を減らす
        let padded = n.max(1).next_multiple_of(BATCH_ALIGN);
        let mut input: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        input.resize(padded * FEATURE_LEN, 0.0);
        let input = Tensor::from_array(([padded, PLANES, SIZE, SIZE], input))?;
        let outputs = self.session.run(ort::inputs!["input" => input])?;
        let (_, policy) = outputs["policy"].try_extract_tensor::<f32>()?;
        let (_, value) = outputs["value"].try_extract_tensor::<f32>()?;
        Ok((0..n)
            .map(|i| Evaluation { policy: policy[i * CELLS..(i + 1) * CELLS].to_vec(), value: value[i] })
            .collect())
    }
}

fn encode_all(boards: &[Board]) -> Vec<u8> {
    let mut features = vec![0; boards.len() * FEATURE_LEN];
    for (board, out) in boards.iter().zip(features.chunks_exact_mut(FEATURE_LEN)) {
        encode(board, out);
    }
    features
}

impl Evaluator for OnnxModel {
    fn evaluate(&mut self, boards: &[Board]) -> Vec<Evaluation> {
        self.run(&encode_all(boards)).expect("推論に失敗しました")
    }
}

struct Request {
    features: Vec<u8>,
    reply: Sender<Vec<Evaluation>>,
}

/// 複数スレッドの探索から来る評価要求を 1 つのバッチにまとめて GPU で推論する。
/// モデルを複数渡すと、あるバッチの推論中に次のバッチを集められる。
pub struct InferenceServer {
    tx: Option<Sender<Request>>,
    handles: Vec<JoinHandle<anyhow::Result<()>>>,
    evaluated: Arc<AtomicU64>,
}

impl InferenceServer {
    pub fn start(models: Vec<OnnxModel>, max_batch: usize) -> Self {
        let (tx, rx) = mpsc::channel();
        let rx = Arc::new(Mutex::new(rx));
        let evaluated = Arc::new(AtomicU64::new(0));
        let handles = models
            .into_iter()
            .map(|model| {
                let rx = Arc::clone(&rx);
                let evaluated = Arc::clone(&evaluated);
                thread::spawn(move || serve(model, &rx, max_batch, &evaluated))
            })
            .collect();
        Self { tx: Some(tx), handles, evaluated }
    }

    /// これまでに推論した局面数
    pub fn evaluated(&self) -> u64 {
        self.evaluated.load(Ordering::Relaxed)
    }

    /// 設定に従ってモデルを inference.threads 個読み込んで起動する。
    pub fn load(path: &Path, config: &InferenceConfig) -> anyhow::Result<Self> {
        static PRELOAD: std::sync::Once = std::sync::Once::new();
        if config.device == Device::Cuda {
            let mut result = Ok(());
            PRELOAD.call_once(|| result = preload_cuda(&config.cuda_lib_dirs));
            result?;
        }
        let models = (0..config.threads)
            .map(|_| OnnxModel::load(path, config.device))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self::start(models, config.max_batch))
    }

    pub fn evaluator(&self) -> RemoteEvaluator {
        RemoteEvaluator { tx: self.tx.clone().expect("停止済み") }
    }

    /// 全ての RemoteEvaluator が破棄された後に呼ぶ。推論スレッドのエラーを返す。
    pub fn shutdown(mut self) -> anyhow::Result<()> {
        self.tx.take();
        for handle in self.handles.drain(..) {
            handle.join().expect("推論スレッドがパニックしました")?;
        }
        Ok(())
    }
}

fn serve(
    mut model: OnnxModel,
    rx: &Mutex<Receiver<Request>>,
    max_batch: usize,
    evaluated: &AtomicU64,
) -> anyhow::Result<()> {
    let positions = |reqs: &[Request]| reqs.iter().map(|r| r.features.len() / FEATURE_LEN).sum::<usize>();
    let mut carry: Option<Request> = None;
    loop {
        // 受信側をロックしている間に 1 バッチ分集め、推論はロックを外してから行う
        let mut requests: Vec<Request> = carry.take().into_iter().collect();
        {
            let rx = rx.lock().expect("推論スレッドがパニックしました");
            if requests.is_empty() {
                let Ok(first) = rx.recv() else { return Ok(()) };
                requests.push(first);
            }
            // 他の探索スレッドの要求が揃うのを少しだけ待ってからまとめて推論する
            while positions(&requests) < max_batch {
                match rx.recv_timeout(Duration::from_micros(200)) {
                    // max_batch を超える要求は次のバッチに回す
                    Ok(req) if positions(&requests) + positions(std::slice::from_ref(&req)) > max_batch => {
                        carry = Some(req);
                        break;
                    }
                    Ok(req) => requests.push(req),
                    Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
                }
            }
        }
        let features: Vec<u8> = requests.iter().flat_map(|r| r.features.iter().copied()).collect();
        let mut evals = match model.run(&features) {
            Ok(evals) => evals.into_iter(),
            Err(e) => {
                // 各探索スレッドが待ったまま巻き添えでパニックしないよう、原因を出してプロセスごと終了する
                eprintln!("推論に失敗しました: {e:#}");
                std::process::exit(1);
            }
        };
        evaluated.fetch_add(evals.len() as u64, Ordering::Relaxed);
        for req in requests {
            let n = req.features.len() / FEATURE_LEN;
            // 要求元が既に終了していても問題ない
            let _ = req.reply.send(evals.by_ref().take(n).collect());
        }
    }
}

/// InferenceServer に評価を依頼する Evaluator。探索スレッドごとに 1 つ持つ。
pub struct RemoteEvaluator {
    tx: Sender<Request>,
}

impl Evaluator for RemoteEvaluator {
    fn evaluate(&mut self, boards: &[Board]) -> Vec<Evaluation> {
        let (reply, rx) = mpsc::channel();
        self.tx.send(Request { features: encode_all(boards), reply }).expect("推論スレッドが停止しています");
        rx.recv().expect("推論スレッドが停止しています")
    }
}
