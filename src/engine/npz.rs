//! numpy の .npz (圧縮) 書き出し。np.load でそのまま読める。

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

pub enum NpyData<'a> {
    U8(&'a [u8]),
    F32(&'a [f32]),
}

pub struct NpyArray<'a> {
    pub name: &'a str,
    pub shape: Vec<usize>,
    pub data: NpyData<'a>,
}

impl NpyData<'_> {
    fn len(&self) -> usize {
        match self {
            NpyData::U8(d) => d.len(),
            NpyData::F32(d) => d.len(),
        }
    }

    fn descr(&self) -> &'static str {
        match self {
            NpyData::U8(_) => "|u1",
            NpyData::F32(_) => "<f4",
        }
    }
}

/// 書き込み途中のファイルを学習側が読まないよう、一時ファイルに書いてから置き換える。
pub fn write_npz(path: &Path, arrays: &[NpyArray]) -> io::Result<()> {
    let tmp = path.with_extension("npz.tmp");
    let mut zip = ZipWriter::new(BufWriter::new(File::create(&tmp)?));
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    for array in arrays {
        assert_eq!(array.shape.iter().product::<usize>(), array.data.len(), "{} の形が合わない", array.name);
        zip.start_file(format!("{}.npy", array.name), options)?;
        write_npy(&mut zip, array)?;
    }
    zip.finish()?.flush()?;
    std::fs::rename(tmp, path)
}

fn write_npy<W: Write>(w: &mut W, array: &NpyArray) -> io::Result<()> {
    let shape = match array.shape.as_slice() {
        [n] => format!("({n},)"),
        dims => format!("({})", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header =
        format!("{{'descr': '{}', 'fortran_order': False, 'shape': {shape}, }}", array.data.descr());
    // マジック(6) + バージョン(2) + ヘッダ長(2) + ヘッダ + 改行 が 64 の倍数になるよう空白で埋める
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat(unpadded.next_multiple_of(64) - unpadded));
    header.push('\n');

    w.write_all(b"\x93NUMPY\x01\x00")?;
    w.write_all(&(header.len() as u16).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    match array.data {
        NpyData::U8(d) => w.write_all(d),
        NpyData::F32(d) => {
            let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
            w.write_all(&bytes)
        }
    }
}
