import os
import sys
from datasets import Dataset

# Add certain path for importing modules
new_path = os.path.join(os.path.dirname(__file__), "../data")
sys.path.append(new_path)

from load import amazon_review_load, amazon_meta_load  # type: ignore


# Define personalized prompts
rating_tasks = {}

# [ 1-1 ] - User ID, Item ID
template = {}
template["source"] = (
    "Which star rating will user_{} give item_{} ? ( 1 being lowest and 5 being highest )"
)
template["target"] = "{}"
rating_tasks["1-1"] = template

# [ 1-2 ] - User ID, Item title
template = {}
template["source"] = (
    "How will user_{} rate this product : ' {} ' ? ( 1 being lowest and 5 being highest )"
)
template["target"] = "{}"
rating_tasks["1-2"] = template

# [ 1-3 ] - User ID, Item ID, Item title
template = {}
template["source"] = (
    "Predict the user_{} 's preference on item_{} ( {} ) \n -1 \n -2 \n -3 \n -4 \n -5"
)
template["target"] = "{}"
rating_tasks["1-3"] = template

# [ 1-4 ] - User Desc., Item ID
template = {}
template["source"] = (
    "What star rating do you think ' {} ' will give item_{} ? ( 1 being lowest and 5 being highest )"
)
template["target"] = "{}"
rating_tasks["1-4"] = template

# [ 1-5 ] - User Desc., Item title
template = {}
template["source"] = (
    "How will ' {} ' rate this product : ' {} ' ? ( 1 being lowest and 5 being highest )"
)
template["target"] = "{}"
rating_tasks["1-5"] = template

# [ 1-6 ] - User Desc., Item title, Item ID
template = {}
template["source"] = (
    "Predict {} 's preference towards {} ( item_{} ) \n -1 \n -2 \n -3 \n -4 \n -5"
)
template["target"] = "{}"
rating_tasks["1-3"] = template

# '[ 2-1 ] - User ID, Item ID, Star rating / Yes_or_No
template = {}
template["source"] = (
    "Will user_{} give item_{} a {}-star rating ? ( 1 being lowest and 5 being highest )"
)
template["target"] = "{}"
rating_tasks["2-1"] = template

# '[ 2-2 ] - User Desc., Star rating, Item title / Yes_or_No
template = {}
template["source"] = (
    "Will ' {} ' give a {}-star rating for ' {} ' ? ( 1 being lowest and 5 being highest )"
)
template["target"] = "{}"
rating_tasks["2-2"] = template

# '[ 3-1 ] - User ID, Item ID / Dislike(1, 2, 3)_or_Like(4, 5)
template = {}
template["source"] = "Does user_{} like or dislike item_{} ?"
template["target"] = "{}"
rating_tasks["3-1"] = template

# '[ 3-2 ] - User Desc, Item title / Dislike(1, 2, 3)_or_Like(4, 5)
template = {}
template["source"] = "Does {} like or dislike {} ?"
template["target"] = "{}"
rating_tasks["3-1"] = template


# Mapping function - Review
def asin_join(data_dict, amazon_meta_dict):

    # 1. Concatenate user review's title/text
    data_dict["review"] = data_dict["title"] + " - " + data_dict["text"]

    # 2. Join data_dict with meta dataset's dictionary
    data_dict["item_title"] = amazon_meta_dict[data_dict["parent_asin"]]

    return data_dict


# Method to join amazon review/meta dataset
def join_dataset(category):

    # Load amazon review/meta dataset
    amazon_review_dataset = amazon_review_load(category)
    amazon_meta_dataset = amazon_meta_load(category)

    # Generate meta dataset's dictionary
    amazon_meta_dict = {}
    for data_dict in amazon_meta_dataset:
        amazon_meta_dict[data_dict["parent_asin"]] = data_dict["title"]

    # Applying certain function to review dataset
    amazon_review_dataset = amazon_review_dataset["full"].map(
        lambda x: asin_join(x, amazon_meta_dict)
    )

    return amazon_review_dataset


# Mapping function - Review
def prompt_transform(data_dict, prompt_format):

    data_dict["prompt"] = prompt_format.format(data_dict["user_id"], data_dict["asin"])

    return data_dict
  

# Method to generate personalized prompts
def generate_prompts(dataset, test_size, rag_size, task_id_list):
  
    # Local vars
    data = dict()
    upper_bound_prompts = []
    upper_bound_targets = []
    base_prompts = []
    base_targets = []
    rag_prompts = []

    # Split given dataset into train/test
    upper_bound_dataset = dataset.train_test_split(test_size=test_size, shuffle=True)
    # Split upper bound dataset into Base/RAG
    base_dataset = upper_bound_dataset["train"].train_test_split(test_size=rag_size, shuffle=True)
    
    # Create personalized prompts based on task_id
    for task_id in task_id_list:
        if rating_tasks[task_id] is not None:
            prompt_format = rating_tasks[task_id]["source"] # Task prompt
            
            # Apply a function to format the prompt
            base_dataset = base_dataset["train"].map(
              lambda x: prompt_transform(x, prompt_format)
            )
            base_prompts = base_dataset["prompt"]
            base_targets = base_dataset["rating"]
            
    # Integrate prompts and targets into one data dictionary
    data["base_prompts"] = base_prompts
    data["base_targets"] = base_targets

    return data


# Method to create a dataset with personalized prompts
def load_dataset(data):
  
    # Define base dataset
    base_dict = {
      "text": data["base_prompts"],
      "target": data["base_targets"]
    }
    base_dataset = Dataset.from_dict(base_dict)
    
    return base_dataset
