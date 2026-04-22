#another AI tool to check data
import numpy as np
import os

# --- CONFIG ---
action = input("Enter the letter folder to scan (e.g., A): ").upper()
DATA_PATH = os.path.join('ASL_Dataset', action)

if not os.path.exists(DATA_PATH):
    print(f"Directory {DATA_PATH} does not exist. Please check the path.")
    exit()

print(f"\n--- Batch Data Report for {DATA_PATH} ---")

clean_files = []
corrupted_files = []

# Gather and sort files numerically (so 2.npy comes before 10.npy)
files = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
files.sort(key=lambda x: int(x.replace('.npy', '')))

for filename in files:
    filepath = os.path.join(DATA_PATH, filename)
    
    try:
        data = np.load(filepath)
        
        # Check if the shape is correct before anything else
        if data.shape != (16, 63):
            print(f"[{filename:^8}] ERROR - Wrong shape: {data.shape}")
            corrupted_files.append(filename)
            continue
            
        # Find frames that are completely empty
        zero_frames = []
        for frame_num in range(data.shape[0]):
            if np.all(data[frame_num] == 0):
                zero_frames.append(frame_num)
        
        if not zero_frames:
            print(f"[{filename:^8}] PASS - 100% Clean")
            clean_files.append(filename)
        else:
            print(f"[{filename:^8}] FAIL - Empty frames: {zero_frames}")
            corrupted_files.append(filename)
            
    except Exception as e:
        print(f"[{filename:^8}] ERROR - Could not read file: {e}")

# --- SUMMARY STATISTICS ---
print("\n" + "="*45)
print("FINAL BATCH SUMMARY")
print("="*45)
print(f"Total Files Scanned: {len(files)}")
print(f"Clean Files:         {len(clean_files)}")
print(f"Corrupted Files:     {len(corrupted_files)}")

if not corrupted_files:
    print("\nStatus: PERFECT. This dataset is cleared for training!")
else:
    print("\nStatus: ACTION REQUIRED. Run Data_Correction.py on the failed files.")
