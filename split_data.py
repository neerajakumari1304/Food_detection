import os
import random
import shutil
from log_code import setup_logging

# Initialize logging
logger = setup_logging('split_data')


def split_dataset():
    try:
        # 1. PATH CONFIGURATION
        # Use forward slashes or raw strings to avoid path errors
        src_dir = r"D:\projects\Food_detection\archive (5)\Food Classification dataset"
        base_dir = r"D:\projects\Food_detection\data_split"

        # 2. TARGET COUNTS
        N_TRAIN, N_VAL, N_TEST = 200, 50, 10
        TOTAL_NEEDED = N_TRAIN + N_VAL + N_TEST

        # 3. CLEANUP: Delete old split data so counts don't stack up (e.g., the 375 potato images)
        if os.path.exists(base_dir):
            logger.info("Cleaning up old data_split folder...")
            shutil.rmtree(base_dir)
        os.makedirs(base_dir, exist_ok=True)

        if not os.path.exists(src_dir):
            logger.error(f"Source folder not found: {src_dir}")
            return

        # 4. PROCESSING CLASSES
        for cls in os.listdir(src_dir):
            class_path = os.path.join(src_dir, cls)

            # Skip if not a directory
            if not os.path.isdir(class_path):
                continue

            # Filter for images only
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            images.sort()  # Step A: Alphabetize so the starting order is always the same
            random.seed(42)  # Step B: Lock the "shuffle" so it produces the same result every time
            random.shuffle(images)

            count = len(images)

            if count < 10:
                logger.warning(f"Skipping {cls}: Only {count} images (too few).")
                continue

            random.shuffle(images)

            # 5. ADAPTIVE SPLITTING LOGIC
            if count >= TOTAL_NEEDED:
                # Use your exact requested numbers
                train_files = images[:N_TRAIN]
                val_files = images[N_TRAIN: N_TRAIN + N_VAL]
                test_files = images[N_TRAIN + N_VAL: TOTAL_NEEDED]
            else:
                # Fallback: Proportional split for small folders (80/15/5%)
                logger.info(f"Class '{cls}' is small ({count} imgs). Using proportional split.")
                t_idx = int(count * 0.8)
                v_idx = int(count * 0.95)
                train_files = images[:t_idx]
                val_files = images[t_idx:v_idx]
                test_files = images[v_idx:]

            # 6. COPYING FILES
            splits = {
                "train": train_files,
                "valid": val_files,
                "test": test_files
            }

            for split_name, files in splits.items():
                dest_path = os.path.join(base_dir, split_name, cls)
                os.makedirs(dest_path, exist_ok=True)
                for f in files:
                    shutil.copy(os.path.join(class_path, f), os.path.join(dest_path, f))

            logger.info(f"Processed {cls}: {len(train_files)} Train, {len(val_files)} Valid, {len(test_files)} Test")

        logger.info("Dataset successfully split and balanced!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


