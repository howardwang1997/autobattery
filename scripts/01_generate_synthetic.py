"""Generate synthetic training data from PyBaMM simulations."""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from src.simulation.data_generator import SyntheticDataGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic battery data")
    parser.add_argument("--config", type=str, default="configs/lmb.yaml", help="Config file")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generator = SyntheticDataGenerator(args.config, output_dir=args.output)
    output_path = generator.generate(num_workers=args.num_workers, seed=args.seed)
    print(f"Data saved to: {output_path}")


if __name__ == "__main__":
    main()
