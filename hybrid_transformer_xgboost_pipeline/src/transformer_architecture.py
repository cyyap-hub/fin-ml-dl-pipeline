import numpy as np
import pandas as pd
from dataset_generation import build_dataset
from sklearn.preprocessing import StandardScaler
import gc

# ==========================================
# INITIALISE FEATURE EXTRACTION
# ==========================================

# HARDWARE & ENVIRONMENT SETUP
def transformer_extraction(ticker="^GSPC", start_date="2000-01-01", end_date="2026-08-01",model_size="AutonLab/MOMENT-1-small"):
    from momentfm import MOMENTPipeline
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing deep learning on: {device}")

    sequence_length = 512
    batch_size = 64  # Optimised for GPU Parallel Processing

    # Data Fetching & Engineering
    print("\nFetching Ticker data...")
    df = build_dataset(ticker=ticker, start_date=start_date, end_date=end_date)


    # Scale the raw Close prices for stability with MOMENT Foundation Model
    scaler = StandardScaler()
    close_prices_scaled = scaler.fit_transform(df[['Close']].values)

    # ===========================================
    # MOMENT FEATURE EXTRACTION SETUP
    # ===========================================
    def setup_moment_model(model_size):
        print(f"Loading {model_size} in embedding mode...")
        model = MOMENTPipeline.from_pretrained(
            model_size,
            model_kwargs={"task_name": "embedding"}
        )
        model.init()

        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        model.to(device)
        return model

    def extract_embeddings(model, batch_tensor):
        # Gives the correct shape: [batch_size, 1 channel, sequence_len]
        # This acts as our safety net against numpy/pandas dimension swapping
        batch_tensor = batch_tensor.reshape(batch_tensor.shape[0], 1, 512).to(device)

        with torch.no_grad():
            outputs = model(x_enc=batch_tensor)
            embeddings = outputs.embeddings.cpu().numpy()
        return embeddings
    # Specifying the MOMENT model variant
    moment_model = setup_moment_model(model_size)

    # Processing Foundation Model Extraction on GPU in batches windows of 512 days
    print("Generating rolling windows...")
    windows = []
    for i in range(len(close_prices_scaled) - sequence_length):
        windows.append(close_prices_scaled[i: i + sequence_length])

    all_embeddings = []
    print(f"Processing {len(windows)} windows through MOMENT...")
    for i in range(0, len(windows), batch_size):
        batch = windows[i: i + batch_size]
        batch_tensor = torch.tensor(np.array(batch), dtype=torch.float32)

        batch_embeddings = extract_embeddings(moment_model, batch_tensor)
        all_embeddings.append(batch_embeddings)

        if i % (batch_size * 5) == 0:
            print(f"Processed {i} / {len(windows)} windows...")

    final_embeddings_array = np.concatenate(all_embeddings, axis=0)
    #=============================================================
    # MEMORY CLEANING AND MERGING MOMENT EMBEDDINGS INTO DATASET
    #=============================================================
    print("\nClearing GPU memory...")
    del moment_model
    torch.cuda.empty_cache()
    gc.collect()

    print("\nMerging hybrid dataset...")

    valid_dates = df.index[sequence_length:]
    embedding_cols = [f"emb_{j}" for j in range(final_embeddings_array.shape[1])]

    embeddings_df = pd.DataFrame(
        final_embeddings_array,
        index=valid_dates, columns = embedding_cols
    )

    models_df = pd.concat([df, embeddings_df], axis=1).dropna()

    print(f"Models' Dataframe Shape: {models_df.shape}")

    return models_df

if __name__ == "__main__":
    transformer_extraction()