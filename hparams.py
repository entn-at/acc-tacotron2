# Mel
num_mels = 80
n_mel_channels = num_mels
n_frames_per_step = 1
num_freq = 1025
sample_rate = 24000
frame_length_ms = 50
frame_shift_ms = 10
preemphasis = 0.97
fmin = 40
min_level_db = -100
ref_level_db = 20
max_iters = 20000
signal_normalization = True
griffin_lim_iters = 60
power = 1.5


# TTS-DNN
max_sep_len = 5000
num_phn = 384
kernel_size = 3
stride = 1
padding = 1
encoder_dim = 384
encoder_n_layer = 3
encoder_head = 8
decoder_dim = 384 + 128
decoder_n_layer = 3
decoder_head = 8
dropout = 0.1
duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3

fft_conv1d_kernel = 3
fft_conv1d_padding = 1
ref_enc_filters = [32, 32, 64, 64, 128, 128]

encoder_conv1d_filter_size = 1536
decoder_conv1d_filter_size = 1536
num_heads = 8
token_num = 3
style_dim = 128

# Tacotron2

# PostNet
postnet_embedding_dim = 512
postnet_kernel_size = 5
postnet_n_convolutions = 5

# PreNet
prenet_dim = 256

# Encoder
encoder_n_convolutions = 3
encoder_embedding_dim = 512
# encoder_embedding_dim = 1024
encoder_kernel_size = 5

# Decoder
attention_rnn_dim = 1024
decoder_rnn_dim = 1024
max_decoder_steps = 1000
gate_threshold = 0.5
p_attention_dropout = 0.1
p_decoder_dropout = 0.1
attention_location_kernel_size = 31
attention_location_n_filters = 32
attention_dim = 128

# Tacotron2
mask_padding = True
n_symbols = num_phn
symbols_embedding_dim = 512
symbols_embedding_dim = 1024

# Train
checkpoint_path = "./model_new"
logger_path = "./logger"

batch_size = 32
epochs = 1000
n_warm_up_step = 4000

learning_rate = 1e-3
weight_decay = 1e-6
grad_clip_thresh = 1.0
decay_step = [500000, 1000000, 2000000]

save_step = 3000
log_step = 5
clear_Time = 20
