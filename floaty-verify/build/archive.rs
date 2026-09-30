//! Fetches the pinned release archives into the download cache, checks their
//! SHA-256, and extracts them.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::tools::run_once;

/// How an archive is packed.
#[derive(Clone, Copy)]
pub(super) enum Packing {
    /// A zip file. `unzip` is not a dependency of the build; Python's
    /// `zipfile` module is.
    Zip,
    /// A tar file compressed with gzip.
    GzipTar,
    /// One source file, which the extraction copies under this name.
    File(&'static str),
}

/// A pinned release archive.
pub(super) struct Archive {
    /// The file name in the download cache.
    pub(super) file: &'static str,
    /// How the archive is packed.
    pub(super) packing: Packing,
    /// Where the release is published.
    pub(super) url: &'static str,
    /// The SHA-256 of the archive, in lowercase hexadecimal.
    pub(super) sha256: &'static str,
}

/// Returns the cached archive, and downloads it first when the cache does
/// not hold it. Stops the build when the archive does not match its pinned
/// SHA-256.
pub(super) fn fetch(archive: &Archive, downloads: &Path) -> PathBuf {
    let path = downloads.join(archive.file);
    if path.is_file() {
        let actual = sha256(&path);
        assert!(
            actual == archive.sha256,
            "{} has SHA-256 {actual}, not the pinned {}; delete the file to download it again",
            path.display(),
            archive.sha256
        );
        return path;
    }
    fs::create_dir_all(downloads).expect("the download cache can be created");
    // A partial download never takes the final name, so the cache holds
    // only verified archives.
    let partial = downloads.join(format!("{}.part", archive.file));
    let status = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "5",
            "--retry-all-errors",
            "--output",
        ])
        .arg(&partial)
        .arg(archive.url)
        .status()
        .expect("curl can run");
    assert!(status.success(), "curl cannot download {}", archive.url);
    let actual = sha256(&partial);
    assert!(
        actual == archive.sha256,
        "{} has SHA-256 {actual}, not the pinned {}",
        archive.url,
        archive.sha256
    );
    fs::rename(&partial, &path).expect("the verified download can be renamed");
    path
}

/// Returns the SHA-256 of a file in lowercase hexadecimal.
pub(super) fn sha256(path: &Path) -> String {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .expect("sha256sum can run");
    assert!(
        output.status.success(),
        "sha256sum failed on {}",
        path.display()
    );
    let text = String::from_utf8(output.stdout).expect("sha256sum writes text");
    text.split_whitespace()
        .next()
        .expect("sha256sum writes the digest first")
        .to_owned()
}

/// Extracts an archive into an empty `destination`, unless the destination
/// already holds this archive.
pub(super) fn unpack(path: &Path, archive: &Archive, destination: &Path) {
    run_once(&destination.with_extension("stamp"), archive.sha256, || {
        if destination.exists() {
            fs::remove_dir_all(destination).expect("an old extraction can be removed");
        }
        fs::create_dir_all(destination).expect("the extraction directory can be created");
        let status = match archive.packing {
            Packing::File(name) => {
                fs::copy(path, destination.join(name)).expect("the source file can be copied");
                return;
            }
            Packing::Zip => Command::new("python3")
                .args(["-m", "zipfile", "-e"])
                .arg(path)
                .arg(destination)
                .status()
                .expect("python3 can run"),
            // The QD archive holds macOS extended headers, which GNU tar
            // ignores with a warning.
            Packing::GzipTar => Command::new("tar")
                .arg("--warning=no-unknown-keyword")
                .arg("-xzf")
                .arg(path)
                .arg("-C")
                .arg(destination)
                .status()
                .expect("tar can run"),
        };
        assert!(status.success(), "{} cannot be extracted", path.display());
    });
}
