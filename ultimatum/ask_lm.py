import os
import pickle as pkl

import ultimatum_prompts
# import ultimatum_prompts_zs as ultimatum_prompts
# import ultimatum_prompts_shorter as ultimatum_prompts
# import ultimatum_prompts_shorter_no_explanation as ultimatum_prompts
# import ultimatum_prompts_shorter_no_rho1 as ultimatum_prompts
from language_models import OpenAIModel
import time


def ask_lm(prompt_type, threshold=None):
    if prompt_type == "low_high_percentage":
        assert threshold is not None
        prompts = ultimatum_prompts.get_low_high_percentage_prompt(threshold=threshold)
    elif prompt_type == "low_high_payoff":
        assert threshold is not None
        prompts = ultimatum_prompts.get_low_high_payoff_prompt(threshold=threshold)
    elif prompt_type == "inequity_aversion":
        prompts = ultimatum_prompts.get_inequity_aversion_prompt()

    api_key = os.environ.get("OPENAI_API_KEY")
    to_save = []
    for i, prompt in enumerate(prompts):
        if i % 20 == 0:
            time.sleep(120)
        lm = OpenAIModel(api_key)
        response = lm.predict_token(prompt)
        to_save.append((prompt, response))
    return to_save


def main(prompt_type, threshold=None):
    to_save = ask_lm(prompt_type, threshold=threshold)
    print(to_save)
    with open(
        f"ultimatum/lm_responses/{prompt_type}_{threshold}.pkl",
        "wb",
    ) as f:
        pkl.dump(to_save, f)


if __name__ == "__main__":
    # prompt_type options: low_high_payoff, low_high_percentage
    prompt_type = "low_high_payoff"
    # threshold for low_high_payoff: 10,100
    # threshold for low_high_percentage: 0.3,0,6
    threshold = 10
    
    main(prompt_type, threshold)
    # with open(f"ultimatum/lm_responses/{prompt_type}_{threshold}.pkl", "rb") as f:
    #    data = pkl.load(f)
