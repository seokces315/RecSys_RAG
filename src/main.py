from data import join_dataset, generate_prompts
from utils import set_seed

HF_TOKEN = "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"


def main():

    # 1. Set seed for reproducibility
    set_seed(42)

    # 2. Load dataset
    category = "Toys_and_Games"
    test_size = 0.1
    task_id = [
        "1-1",
        "1-2",
        "1-3",
        "1-4",
        "1-5",
        "1-6",
    ]

    TG_dataset = join_dataset(category)
    generate_prompts(TG_dataset, test_size, task_id)
    # PP_dataset = load_dataset()
    # print(PP_dataset)
    print()

    return


if __name__ == "__main__":
    main()
