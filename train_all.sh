#    STAGE TDL validation — all four algorithms:
#      1. CNN-LSTM
#      2. CNN-Sparse
#      3. Sa-DLCS
#      4. NCRN
#  Total: 4 trained models. Approx runtime on Apple M1: ~20 minutes.
#  Usage:
#    bash train_all.sh                   # train all 4 models
#    SKIP_CNN=1 bash train_all.sh        # skip CNN-LSTM and CNN-Sparse
set -e
cd "$(dirname "$0")"

echo "======================================================================"
echo "  TDL CHECKING — Training Pipeline (64x64)"
echo "======================================================================"

START_TIME=$(date +%s)

if [ -z "$SKIP_CNN" ]; then
    echo ""
    echo "─── [1/4] CNN-LSTM TDL 64x64 ───"
    cd stage2_tdl/cnn_lstm; python train_cnn_lstm_tdl.py; cd ../..

    echo ""
    echo "─── [2/4] CNN-Sparse TDL 64x64 ───"
    cd stage2_tdl/cnn_sparse; python train_cnn_sparse_tdl.py; cd ../..
else
    echo ""
    echo "─── [1/4] CNN-LSTM   SKIPPED (SKIP_CNN=1) ───"
    echo "─── [2/4] CNN-Sparse SKIPPED (SKIP_CNN=1) ───"
fi

echo ""
echo "─── [3/4] Sa-DLCS TDL 64x64 ───"
cd stage2_tdl/sa_dlcs; python train_sa_dlcs_tdl.py; cd ../..

echo ""
echo "─── [4/4] NCRN TDL 64x64 ───"
cd stage2_tdl/ncrn; python train_ncrn_tdl.py; cd ../..

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MIN=$((ELAPSED / 60))

echo ""
echo "======================================================================"
echo "  TRAINING COMPLETE — total time: ${MIN} min"
echo "  Models saved in: ./models/"
echo ""
echo "  Next step: generate figures"
echo "    python plotting/plot_all_figures.py"
echo "    python plotting/plot_all_figures.py --quick   # fast preview"
echo "======================================================================"