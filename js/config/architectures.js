// Built-in architectures, expressed in the Keras functional API.
//
// Each template receives the input tensor as `x` and must leave its result in
// `x`; the output head is appended separately according to the modality, so no
// template emits its own classifier.
export const ARCHITECTURES = {
    simple_cnn: () => `# Simple CNN
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)`,

    alexnet: () => `# AlexNet
x = layers.Conv2D(96, (11, 11), strides=4, activation='relu')(x)
x = layers.MaxPooling2D((3, 3), strides=2)(x)
x = layers.Conv2D(256, (5, 5), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((3, 3), strides=2)(x)
x = layers.Conv2D(384, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(384, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((3, 3), strides=2)(x)
x = layers.Flatten()(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)`,

    vgg16: () => `# VGG16 backbone, expanded block by block
for block, (n_conv, filters) in enumerate([(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)], start=1):
    for conv in range(1, n_conv + 1):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same',
                          name=f'block{block}_conv{conv}')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=f'block{block}_pool')(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)`,

    lstm: () => `# Stacked LSTM
x = layers.LSTM(128, return_sequences=True)(x)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(128, return_sequences=True)(x)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(64)(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)`,

    gru: () => `# Stacked GRU
x = layers.GRU(128, return_sequences=True)(x)
x = layers.Dropout(0.2)(x)
x = layers.GRU(64, return_sequences=True)(x)
x = layers.Dropout(0.2)(x)
x = layers.GRU(32)(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(16, activation='relu')(x)`,

    // v1.0.x placed MultiHeadAttention inside a Sequential, which Keras rejects:
    // attention takes (query, value), not a single tensor. Written properly, a
    // transformer block is residual by construction, so it needs the functional
    // API regardless.
    transformer: () => `# Transformer encoder
# Learned positional embeddings are added to the token embeddings.
positions = tf.range(start=0, limit=SEQ_LEN, delta=1)
x = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, name='token_embedding')(x)
x = x + layers.Embedding(input_dim=SEQ_LEN, output_dim=EMBED_DIM, name='position_embedding')(positions)

for block in range(N_BLOCKS):
    # Self-attention sublayer, with its residual connection
    attn = layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=EMBED_DIM // N_HEADS,
                                     name=f'block{block}_attention')(x, x)
    attn = layers.Dropout(0.1)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, attn]))

    # Feed-forward sublayer, with its residual connection
    ff = layers.Dense(FF_DIM, activation='relu')(x)
    ff = layers.Dense(EMBED_DIM)(ff)
    ff = layers.Dropout(0.1)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, ff]))

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)`,

    // v1.0.x emitted the encoder and decoder as a straight stack, so the skip
    // connections that define a U-Net were absent. They are the architecture.
    unet: () => `# U-Net
skips = []
for filters in [64, 128, 256]:
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    skips.append(x)                      # kept for the matching decoder stage
    x = layers.MaxPooling2D((2, 2))(x)

# Bridge
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)

for filters, skip in zip([256, 128, 64], reversed(skips)):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip])  # the skip connection
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)`,

    autoencoder: () => `# Autoencoder
# Encoder
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)

# Latent space
x = layers.Dense(LATENT_DIM, activation='relu', name='latent_space')(x)

# Decoder
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)`
};
