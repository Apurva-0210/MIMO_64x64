import os, sys, time, pickle
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                  '..', '..'))
from shared.channels import gen_tdl_random, TIME_STEPS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ─── Config ───────────────────────────────────────────────────
EPOCHS       = 25
STEPS_EPOCH  = 50
TRAIN_SNR_DB = 10.0


def build_sa_dlcs(Nr, Nt):
    inp = tf.keras.Input(shape=(TIME_STEPS, Nr, Nt, 2))
    x   = tf.keras.layers.ConvLSTM2D(
              16, (3,3), padding='same', return_sequences=True,
              activation='relu')(inp)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.ConvLSTM2D(
              8, (3,3), padding='same', return_sequences=False,
              activation='relu')(x)
    x   = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Conv2D(2, (3,3), padding='same',
                                  activation='linear')(x)
    m   = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='mse', metrics=['mae'])
    return m


def make_dataset(Nr, Nt, batch_size):
    ns = np.sqrt(10**(-TRAIN_SNR_DB/10))
    def _gen():
        while True:
            X, Y = [], []
            for _ in range(batch_size):
                h = gen_tdl_random(Nr, Nt)
                hn = (h + np.random.normal(0, ns, h.shape)).astype(np.float32)
                X.append(hn); Y.append(h[-1])
            yield np.array(X, dtype=np.float32), \
                  np.array(Y, dtype=np.float32)
    return tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec((batch_size, TIME_STEPS, Nr, Nt, 2), tf.float32),
            tf.TensorSpec((batch_size, Nr, Nt, 2),             tf.float32),
        )
    ).prefetch(tf.data.AUTOTUNE)


def main(Nr, Nt):
    suffix     = '128' if Nt == 128 else '64'
    batch_size = 8 if Nt == 128 else 4

    print(f"\n{'='*64}")
    print(f"  Sa-DLCS Stage II — TDL (10 scenarios), {Nr}x{Nt}")
    print(f"  Training SNR: fixed {TRAIN_SNR_DB} dB | Epochs: {EPOCHS}")
    print(f"  ISTA refinement applied at INFERENCE only")
    print(f"{'='*64}")

    model = build_sa_dlcs(Nr, Nt)
    model.summary()
    print(f"\n  Params: {model.count_params():,}\n")

    ds  = make_dataset(Nr, Nt, batch_size)
    t0  = time.time()
    his = model.fit(ds, steps_per_epoch=STEPS_EPOCH, epochs=EPOCHS,
                    verbose=1)
    elapsed = time.time() - t0

    out_dir = os.path.join(os.path.dirname(__file__),
                            '..', '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    out_path  = os.path.join(out_dir, f'sa_dlcs_tdl_{suffix}.keras')
    hist_path = os.path.join(out_dir, f'sa_dlcs_tdl_{suffix}_history.pkl')
    model.save(out_path)
    with open(hist_path, 'wb') as f:
        pickle.dump({'loss': his.history['loss'],
                      'mae':  his.history['mae']}, f)
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Model:   {out_path}")


if __name__ == '__main__':
    main(64, 64)
