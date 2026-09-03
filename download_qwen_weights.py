"""Download Qwen/Qwen3-VL weights for frozen-backbone experiments."""

import argparse

from frozen_backbones import download_qwen_weights


def main():
    parser = argparse.ArgumentParser(description="Download Qwen/Qwen3-VL weights")
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--local_dir", default="./models/Qwen__Qwen3-VL-8B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    path = download_qwen_weights(
        model_id=args.model_id,
        local_dir=args.local_dir,
        revision=args.revision,
        token=args.hf_token,
        local_files_only=args.local_files_only,
    )
    print(path)


if __name__ == "__main__":
    main()
