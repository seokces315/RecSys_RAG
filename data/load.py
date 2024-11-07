from datasets import load_dataset
import json


# Method to load amazon review data
def amazon_review_load(category):

    # Target category
    target = "raw_review_" + category

    amazon_review_dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        target,
        trust_remote_code=True,
    )

    return amazon_review_dataset


# Method to load amazon meta data
def amazon_meta_load(category):

    # Target category
    target = "raw_meta_" + category

    amazon_meta_dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        target,
        split="full",
        trust_remote_code=True,
    )

    return amazon_meta_dataset


if __name__ == "__main__":

    # Local vars
    category = "Toys_and_Games"

    # Load dataset
    amazon_review_dataset = amazon_review_load(category)

    # Transform into list
    entry_list = [entry for entry in amazon_review_dataset["full"]]

    # Save as json file
    with open("amazon_sports_outdoors.json", "w", encoding="utf-8") as json_file:
        json.dump(entry_list, json_file, ensure_ascii=False, indent=4)
