import os
from pathlib import Path

import datasets


def main(data_files, save_dir):
    """
    Preprocess json dataset to parquet format
    """
    data_path = os.path.splitext(Path(data_files[0]))[-1][1:]
    raw_data = datasets.load_dataset(data_path, data_files=data_files, num_proc=16, streaming=False)
    
    total_data_size = len(raw_data["train"])
    test_data_size = int(total_data_size * 0.01)

    raw_test_data = raw_data["train"].select(range(test_data_size))
    raw_train_data = raw_data["train"].select(range(test_data_size, total_data_size))
    print(f"train_data size: {len(raw_train_data)}, test_data size: {len(raw_test_data)}")
    
    def make_map_fn(split):
        def process_fn(example, idx):
            query = example['query']
            response = example['response']
            
            data = {
                "data_source": "mcq",
                "prompt": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                "ability": "instruction",
                "reward_model": {"style": "rule", "ground_truth": response},
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data

        return process_fn
    
    train_data = raw_train_data.map(function=make_map_fn("train"), with_indices=True)
    test_data = raw_test_data.map(function=make_map_fn("test"), with_indices=True)
    
    train_data.to_parquet(os.path.join(save_dir, "train.parquet"))
    test_data.to_parquet(os.path.join(save_dir, "test.parquet"))
    
    print("Done!")


if __name__ == "__main__":
    data_files = ["/workspace/verl/examples/cheyujie/mcq.json"]
    save_dir = "datasets/mcq"
    main(data_files, save_dir)