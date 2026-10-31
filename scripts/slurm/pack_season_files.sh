#!/bin/bash
# Pack finished runs' per-season files (<run>/<crop>/*.csv, *_summary.json) into one <run>/<crop>.tar.zst each,
# then delete the loose files. The project space allows ~1.1 M files (not bytes) and every season writes 1-2
# files; training and the benchmark only use the processed tables and the manifest.
#
#   sbatch scripts/slurm/pack_season_files.sh <season dir> [<season dir> ...]
#
# A folder's loose files are deleted only after the archive lists the same number of files (listing it
# decompresses the whole stream, so a damaged archive fails here). To reprocess or audit a packed run,
# unpack it first (the manifest's daily_path points into the folder):
#   tar -I zstd -xf <run>/<crop>.tar.zst -C <run>

#SBATCH --job-name=pack-seasons
#SBATCH --output=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/pack-seasons_%j.out
#SBATCH --error=/gpfs/home2/vnedungadi/tnp-crop/slurm_logs/pack-seasons_%j.err
#SBATCH --partition=rome
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail
echo "Job started at:  $(date)"
for dir in "$@"; do
    dir=${dir%/}
    archive=$dir.tar.zst
    n=$(find "$dir" -type f | wc -l)
    echo "== $dir: $n files -> $archive"
    tar -C "$(dirname "$dir")" -I "zstd -T16 -3" -cf "$archive" "$(basename "$dir")"
    m=$(tar -I zstd -tf "$archive" | grep -cv '/$')
    if [ "$m" -ne "$n" ]; then
        echo "MISMATCH: archive lists $m files, folder has $n; keeping the folder"
        exit 1
    fi
    rm -rf "$dir"
    echo "   ok: $m files, $(du -h "$archive" | cut -f1); folder removed"
done
echo "Job finished at:  $(date)"
